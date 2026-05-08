"""LAN TCP server: receives ECG/PPG stream and emits data to GUI."""
from __future__ import annotations
import socket, threading, json, struct, time
from typing import Callable
import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_PORT = 50505
HEADER_FMT = "!4sIHH"  # magic(4) | payload_len(4) | version(2) | msg_type(2)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"MFPX"

MSG_DATA   = 0x01   # ECG+PPG window
MSG_HELLO  = 0x02
MSG_STATUS = 0x03
MSG_ACK    = 0x80


def _pack(msg_type: int, payload: bytes) -> bytes:
    hdr = struct.pack(HEADER_FMT, MAGIC, len(payload), PROTOCOL_VERSION, msg_type)
    return hdr + payload


def _unpack_header(buf: bytes) -> tuple[int, int, int] | None:
    if len(buf) < HEADER_SIZE:
        return None
    magic, plen, ver, mtype = struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    return plen, ver, mtype


class ClientHandler(threading.Thread):
    def __init__(self, sock: socket.socket, addr: tuple, on_data: Callable, on_disconnect: Callable) -> None:
        super().__init__(daemon=True)
        self.sock = sock
        self.addr = addr
        self.on_data = on_data
        self.on_disconnect = on_disconnect
        self.alive = True
        self.client_id = f"{addr[0]}:{addr[1]}"
        self.frames_received = 0
        self.connected_at = time.time()

    def run(self) -> None:
        buf = b""
        try:
            while self.alive:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    header = _unpack_header(buf)
                    if header is None:
                        break
                    plen, ver, mtype = header
                    total = HEADER_SIZE + plen
                    if len(buf) < total:
                        break
                    payload = buf[HEADER_SIZE:total]
                    buf = buf[total:]
                    self._dispatch(mtype, payload)
        except (ConnectionResetError, OSError):
            pass
        finally:
            self.alive = False
            try:
                self.sock.close()
            except Exception:
                pass
            self.on_disconnect(self.client_id)

    def _dispatch(self, mtype: int, payload: bytes) -> None:
        if mtype == MSG_HELLO:
            try:
                info = json.loads(payload.decode())
                self._send_ack(MSG_HELLO, {"status": "ok", "server": "MetaFingerprint"})
            except Exception:
                pass
        elif mtype == MSG_DATA:
            try:
                meta_len = struct.unpack("!I", payload[:4])[0]
                meta = json.loads(payload[4:4 + meta_len].decode())
                body = payload[4 + meta_len:]
                n = meta.get("n_samples", 1250)
                fs = meta.get("fs", 125.0)
                arr = np.frombuffer(body, dtype=np.float32).reshape(2, n)
                self.frames_received += 1
                self.on_data({
                    "client_id": self.client_id,
                    "patient_id": meta.get("patient_id", ""),
                    "fs": fs,
                    "ecg": arr[0].copy(),
                    "ppg": arr[1].copy(),
                    "abp": None,
                    "frame": self.frames_received,
                })
                self._send_ack(MSG_DATA, {"ok": True, "frame": self.frames_received})
            except Exception:
                pass

    def _send_ack(self, ref_type: int, data: dict) -> None:
        try:
            body = json.dumps(data).encode()
            self.sock.sendall(_pack(MSG_ACK, struct.pack("!H", ref_type) + body))
        except Exception:
            pass

    def status(self) -> dict:
        return {
            "client_id": self.client_id,
            "ip": self.addr[0],
            "port": self.addr[1],
            "frames": self.frames_received,
            "uptime_s": round(time.time() - self.connected_at, 1),
            "alive": self.alive,
        }


class LANServer:
    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, ClientHandler] = {}
        self._lock = threading.Lock()
        self.running = False
        # callbacks
        self.on_data: Callable | None = None
        self.on_client_connect: Callable | None = None
        self.on_client_disconnect: Callable | None = None
        self.on_error: Callable | None = None

    def start(self) -> bool:
        if self.running:
            return True
        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(8)
            self._server.settimeout(1.0)
            self.running = True
            self._thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._thread.start()
            return True
        except OSError as e:
            if self.on_error:
                self.on_error(str(e))
            return False

    def stop(self) -> None:
        self.running = False
        with self._lock:
            for h in list(self._clients.values()):
                h.alive = False
                try:
                    h.sock.close()
                except Exception:
                    pass
            self._clients.clear()
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

    def _accept_loop(self) -> None:
        while self.running:
            try:
                sock, addr = self._server.accept()
                handler = ClientHandler(sock, addr, self._on_data, self._on_disconnect)
                with self._lock:
                    self._clients[handler.client_id] = handler
                handler.start()
                if self.on_client_connect:
                    self.on_client_connect(handler.client_id, addr)
            except socket.timeout:
                continue
            except OSError:
                break

    def _on_data(self, frame: dict) -> None:
        if self.on_data:
            self.on_data(frame)

    def _on_disconnect(self, client_id: str) -> None:
        with self._lock:
            self._clients.pop(client_id, None)
        if self.on_client_disconnect:
            self.on_client_disconnect(client_id)

    def client_list(self) -> list[dict]:
        with self._lock:
            return [h.status() for h in self._clients.values()]

    @property
    def local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close(); return ip
        except Exception:
            return "127.0.0.1"
