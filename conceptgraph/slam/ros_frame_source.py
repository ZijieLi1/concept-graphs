"""In-process ROS2 RGB-D + pose ingest. Callbacks only fill LatestFrameSlot."""

from __future__ import annotations

from typing import Optional

import numpy as np

from conceptgraph.slam.frame_ipc import LatestFrameSlot, quaternion_to_rotation_matrix
from conceptgraph.slam.frame_source import Frame
from conceptgraph.slam.ros_runtime import add_node, ensure_rclpy, remove_node


def sensor_qos_profile():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def image_msg_to_bgr(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("bgr8", "8UC3"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
    if msg.encoding in ("rgb8",):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return rgb[:, :, ::-1].copy()
    raise ValueError(f"unsupported color encoding {msg.encoding}")


def image_msg_to_depth_m(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("32FC1", "32FC"):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def camera_info_to_K(msg) -> np.ndarray:
    return np.asarray(msg.k, dtype=np.float32).reshape(3, 3)


def pose_msg_to_T(msg) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w]).astype(np.float32)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def header_stamp_sec(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


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
        self.slot = LatestFrameSlot()
        self._node = None
        self._started = False
        self._subs = []
        self._sync = None

    def start(self) -> None:
        if self._started:
            return
        from message_filters import Subscriber, TimeSynchronizer
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image

        rclpy = ensure_rclpy()
        qos = sensor_qos_profile()
        self._node = rclpy.create_node(self.node_name)
        rgb_sub = Subscriber(self._node, Image, self.rgb_topic, qos_profile=qos)
        depth_sub = Subscriber(self._node, Image, self.depth_topic, qos_profile=qos)
        info_sub = Subscriber(self._node, CameraInfo, self.camera_info_topic, qos_profile=qos)
        pose_sub = Subscriber(self._node, PoseStamped, self.pose_topic, qos_profile=qos)
        sync = TimeSynchronizer([rgb_sub, depth_sub, info_sub, pose_sub], queue_size=2)
        sync.registerCallback(self._on_sync)
        self._subs = [rgb_sub, depth_sub, info_sub, pose_sub]
        self._sync = sync
        add_node(self._node)
        self._started = True
        print(
            "ROS FrameSource (in-process):\n"
            f"  rgb   {self.rgb_topic}\n"
            f"  depth {self.depth_topic}\n"
            f"  info  {self.camera_info_topic}\n"
            f"  pose  {self.pose_topic}"
        )

    def _on_sync(self, rgb_msg, depth_msg, info_msg, pose_msg) -> None:
        try:
            rgb = image_msg_to_bgr(rgb_msg)
            depth = image_msg_to_depth_m(depth_msg)
            if depth.shape[:2] != rgb.shape[:2]:
                import cv2

                depth = cv2.resize(
                    depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            self.slot.put(
                Frame(
                    rgb=rgb,
                    depth=depth,
                    K=camera_info_to_K(info_msg),
                    pose=pose_msg_to_T(pose_msg),
                    t=header_stamp_sec(rgb_msg.header),
                )
            )
        except Exception as exc:
            print(f"ROS FrameSource skip frame: {exc}")

    def next(self, timeout: float = 1.0) -> Optional[Frame]:
        return self.slot.get(timeout=timeout)

    def close(self) -> None:
        if self._node is not None:
            remove_node(self._node)
            self._node.destroy_node()
            self._node = None
        self._subs = []
        self._sync = None
        self._started = False
