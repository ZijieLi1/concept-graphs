"""Unix-socket Frame transport. Numpy only — safe in conda 3.10 and Jazzy 3.12.

Do not import torch, rclpy, or record3d here.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import subprocess
import time
from threading import Event, Lock
from typing import Any, Optional

import numpy as np

JAZZY_SETUP = "/opt/ros/jazzy/setup.bash"
JAZZY_PYTHON = "/usr/bin/python3"
_HEADER = struct.Struct("!Q")


class LatestFrameSlot:
    """Depth-1 queue: put() overwrites; get() takes the newest item or times out."""

    def __init__(self):
        self._lock = Lock()
        self._frame: Optional[Any] = None
        self._dropped = 0
        self._event = Event()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def put(self, frame: Any) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
                if hasattr(frame, "dropped_before"):
                    frame.dropped_before = self._dropped
                elif isinstance(frame, dict):
                    frame["dropped_before"] = self._dropped
            self._frame = frame
            self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[Any]:
        if not self._event.wait(timeout=timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
            return frame


def rotation_matrix_to_quaternion(R) -> np.ndarray:
    R = np.asarray(R)
    q = np.empty((4,), dtype=np.float32)
    q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
    q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
    q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
    return q


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def write_msg(conn: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(_HEADER.pack(len(payload)) + payload)


def read_msg(conn: socket.socket) -> Any:
    header = _recvall(conn, _HEADER.size)
    if header is None:
        raise ConnectionError("socket closed")
    (n,) = _HEADER.unpack(header)
    if n > 64 * 1024 * 1024:
        raise ValueError(f"frame message too large: {n}")
    payload = _recvall(conn, n)
    if payload is None:
        raise ConnectionError("socket closed")
    return pickle.loads(payload)


def _recvall(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def bind_unix_server(path: str) -> socket.socket:
    if os.path.exists(path):
        os.unlink(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    return sock


def connect_unix(path: str, timeout: float = 15.0) -> socket.socket:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
            return sock
        except OSError as exc:
            last_err = exc
            sock.close()
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to {path}: {last_err}")


def load_jazzy_env(setup_bash: str = JAZZY_SETUP) -> dict:
    """Env for /usr/bin/python3 + Jazzy. No conda PYTHONHOME/PATH."""
    if not os.path.isfile(setup_bash):
        raise FileNotFoundError(f"ROS setup not found: {setup_bash}")
    passthrough = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", ""),
        "DISPLAY": os.environ.get("DISPLAY", ""),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", ""),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "TERM": os.environ.get("TERM", "xterm"),
    }
    for key, val in os.environ.items():
        if key.startswith(("ROS_", "RMW_")):
            passthrough[key] = val
    bootstrap = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TERM": passthrough["TERM"],
        "HOME": passthrough["HOME"],
        "LANG": passthrough["LANG"],
    }
    raw = subprocess.check_output(
        ["/bin/bash", "--noprofile", "--norc", "-c", f"source {setup_bash} && env -0"],
        env=bootstrap,
    )
    env = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        env[k.decode()] = v.decode()
    env.update({k: v for k, v in passthrough.items() if v})
    return env


def spawn_jazzy_python(script: str, args: list[str]) -> subprocess.Popen:
    env = load_jazzy_env()
    cmd = [JAZZY_PYTHON, script, *args]
    return subprocess.Popen(cmd, env=env)
