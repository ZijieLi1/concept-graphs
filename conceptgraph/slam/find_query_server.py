"""Unix-socket JSON find() server. Numpy/pickle only on the wire."""

from __future__ import annotations

import os
from threading import Thread
from typing import Optional

from conceptgraph.slam.frame_ipc import bind_unix_server, read_msg, write_msg
from conceptgraph.slam.grounding import GroundingService


class FindQueryServer:
    def __init__(self, grounding: GroundingService, sock_path: str):
        self.grounding = grounding
        self.sock_path = sock_path
        self._server = None
        self._thread: Optional[Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._server = bind_unix_server(self.sock_path)
        self._server.settimeout(0.5)
        self._running = True
        self._thread = Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"Find query server on {self.sock_path}")

    def _accept_loop(self) -> None:
        while self._running and self._server is not None:
            try:
                conn, _ = self._server.accept()
            except OSError:
                continue
            Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn) -> None:
        try:
            while self._running:
                request = read_msg(conn)
                if not isinstance(request, dict):
                    write_msg(conn, {"ok": False, "message": "expected a dict", "hits": []})
                    continue
                write_msg(conn, self.grounding.handle(request))
        except (ConnectionError, OSError, EOFError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._running = False
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self.sock_path:
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass
