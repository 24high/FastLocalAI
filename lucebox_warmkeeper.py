#!/usr/bin/env python3
"""
Lucebox Warm-Keeper Proxy
=========================

Problem: Luceboxs Disk-KV-Cache restored Folgerequests nur an der Grenze des
ERSTEN gespeicherten Snapshots. Die Grenze rueckt nie nach -- je laenger die
Session, desto mehr Tokens werden bei JEDEM Request neu prefilled (bei ~90
tok/s schnell Minuten).

Loesung: Dieses Proxy sitzt zwischen Client (z.B. OpenCode) und Lucebox.
Es reicht alle Requests unveraendert durch und merkt sich den letzten
Chat-Request. Wenn die Verbindung IDLE_SECONDS ruhig war und der Kontext
seit dem letzten Refresh um mehr als MIN_GROWTH_BYTES gewachsen ist:

  1. loescht es die veralteten Disk-Snapshots im Container,
  2. schickt es den letzten Kontext einmal mit max_tokens=1 als
     Hintergrund-Warming an Lucebox.

Dadurch legt Lucebox einen frischen Snapshot NAHE AM KOPF der Konversation
an, und der naechste echte Request muss nur noch den letzten Turn
prefillen -- die teure Arbeit passiert waehrend der Benutzer liest/denkt.

ACHTUNG: Lucebox bricht laufende Berechnungen nicht ab. Kommt ein echter
Request waehrend des Warmings, wartet er, bis das Warming fertig ist.
Deshalb feuert der Refresh nur nach laengerer Ruhe (Default 120 s).

Start:
    python3 lucebox_warmkeeper.py
    # Client auf http://127.0.0.1:8001/v1 zeigen lassen

Konfiguration (Env):
    LISTEN_PORT       Default 8001
    UPSTREAM          Default 127.0.0.1:8000
    IDLE_SECONDS      Ruhe, bevor ein Refresh starten darf (Default 120)
    MIN_GROWTH_BYTES  Mindest-Wachstum des Request-Bodys seit letztem
                      Refresh (Default 24000 ~ ca. 6k Tokens)
    CONTAINER         Docker-Containername (Default lucebox-qwen36)
    CACHE_DIR         Cache-Pfad im Container (Default /opt/lucebox-hub/cache)
"""
import http.client
import http.server
import json
import os
import subprocess
import threading
import time

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8001"))
UPSTREAM_HOST, UPSTREAM_PORT = (
    os.environ.get("UPSTREAM", "127.0.0.1:8000").rsplit(":", 1)
)
UPSTREAM_PORT = int(UPSTREAM_PORT)
IDLE_SECONDS = float(os.environ.get("IDLE_SECONDS", "120"))
MIN_GROWTH_BYTES = int(os.environ.get("MIN_GROWTH_BYTES", "24000"))
CONTAINER = os.environ.get("CONTAINER", "lucebox-qwen36")
CACHE_DIR = os.environ.get("CACHE_DIR", "/opt/lucebox-hub/cache")

state_lock = threading.Lock()
last_chat_body = None          # letzter /v1/chat/completions Body (bytes)
last_request_ts = 0.0          # Zeitpunkt des letzten echten Requests
last_refresh_len = 0           # Body-Groesse beim letzten Refresh
warming = False                # laeuft gerade ein Warming?


def log(msg):
    print(f"[warmkeeper] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def delete_snapshots():
    """Loescht die .dkv-Snapshots im Container (root-owned, daher docker exec)."""
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c",
         f"rm -f {CACHE_DIR}/*/*.dkv 2>/dev/null; ls {CACHE_DIR}"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0


def send_warming(body_bytes):
    """Schickt den Kontext mit max_tokens=1 an Lucebox (blockiert bis fertig)."""
    body = json.loads(body_bytes)
    body["max_tokens"] = 1
    body["stream"] = False
    body.pop("stream_options", None)
    data = json.dumps(body).encode()
    conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=3600)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", data,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return time.time() - t0, resp.status


def refresh_loop():
    global warming, last_refresh_len
    while True:
        time.sleep(5)
        with state_lock:
            body = last_chat_body
            idle = time.time() - last_request_ts
            grown = body is not None and (len(body) - last_refresh_len) >= MIN_GROWTH_BYTES
        if not grown or idle < IDLE_SECONDS or warming:
            continue
        warming = True
        try:
            size = len(body)
            log(f"Refresh: idle={idle:.0f}s, Kontext {size} bytes "
                f"(+{size - last_refresh_len} seit letztem Refresh)")
            if not delete_snapshots():
                log("WARNUNG: Snapshot-Loeschen fehlgeschlagen, Refresh uebersprungen")
                continue
            dur, status = send_warming(body)
            log(f"Warming fertig nach {dur:.0f}s (HTTP {status})")
            with state_lock:
                last_refresh_len = size
        except Exception as e:
            log(f"Refresh-Fehler: {e}")
        finally:
            warming = False


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _forward(self, method):
        global last_chat_body, last_request_ts
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        if method == "POST" and self.path.rstrip("/").endswith("/chat/completions") and body:
            with state_lock:
                last_chat_body = body
                last_request_ts = time.time()
            if warming:
                log("Echter Request waehrend Warming eingetroffen -- er wartet, "
                    "bis Lucebox das Warming abgearbeitet hat")

        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=3600)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "connection", "transfer-encoding")}
        try:
            conn.request(method, self.path, body, headers)
            resp = conn.getresponse()
        except Exception as e:
            self.send_error(502, f"Upstream-Fehler: {e}")
            return

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            with state_lock:
                last_request_ts = time.time()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


if __name__ == "__main__":
    threading.Thread(target=refresh_loop, daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Proxy)
    log(f"lausche auf 127.0.0.1:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT} "
        f"(idle={IDLE_SECONDS:.0f}s, min_growth={MIN_GROWTH_BYTES}B)")
    srv.serve_forever()
