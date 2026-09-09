"""Mapper-side ROS ingest: conda never imports rclpy.

A Jazzy sidecar (`ros_topics_to_ipc.py`) subscribes and serves the newest
synced frame over a unix socket. This process only pulls.

Phone vs robot: docs/ros-ingest.md.
"""

from __future__ import annotations

import os
from typing import Optional

from conceptgraph.slam.frame_ipc import connect_unix, read_msg, spawn_jazzy_python, write_msg
from conceptgraph.slam.frame_source import Frame

_SIDECAR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "ros_topics_to_ipc.py",
)


def _payload_to_frame(payload: dict) -> Frame:
    return Frame(
        rgb=payload["rgb"],
        depth=payload["depth"],
        K=payload["K"],
        pose=payload["pose"],
        t=float(payload["t"]),
        dropped_before=int(payload.get("dropped_before", 0)),
    )


class RosRgbDPoseSource:
    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        pose_topic: str,
        node_name: str = "conceptgraph_rgbd_sub",
    ):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.pose_topic = pose_topic
        self.node_name = node_name
        self._proc = None
        self._conn = None
        self._sock_path = ""
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._sock_path = f"/tmp/conceptgraph_ros_sub_{os.getpid()}.sock"
        args = [
            "--socket",
            self._sock_path,
            "--rgb-topic",
            self.rgb_topic,
            "--depth-topic",
            self.depth_topic,
            "--info-topic",
            self.camera_info_topic,
            "--pose-topic",
            self.pose_topic,
            "--node-name",
            self.node_name,
        ]
        print(
            "Starting Jazzy ROS sidecar (/usr/bin/python3). "
            "Conda does not import rclpy.\n"
            f"  rgb   {self.rgb_topic}\n"
            f"  depth {self.depth_topic}\n"
            f"  info  {self.camera_info_topic}\n"
            f"  pose  {self.pose_topic}"
        )
        self._proc = spawn_jazzy_python(_SIDECAR, args)
        self._conn = connect_unix(self._sock_path, timeout=20.0)
        self._started = True

    def next(self, timeout: float = 1.0) -> Optional[Frame]:
        if self._conn is None:
            return None
        try:
            write_msg(self._conn, "get")
            payload = read_msg(self._conn)
        except (ConnectionError, OSError):
            return None
        if payload is None:
            return None
        return _payload_to_frame(payload)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._sock_path:
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass
        self._started = False
