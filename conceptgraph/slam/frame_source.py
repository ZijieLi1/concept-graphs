"""Frame ingest for live mapping: one latest slot, never block the producer."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Protocol

import numpy as np
import torch


@dataclass
class Frame:
    """One RGB-D observation in mapper-native types (BGR uint8, depth meters, T_wc)."""

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    t: float
    dropped_before: int = 0

    def intrinsics_4x4(self) -> torch.Tensor:
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = torch.from_numpy(np.asarray(self.K, dtype=np.float32))
        return mat


class FrameSource(Protocol):
    def start(self) -> None: ...

    def next(self, timeout: float = 1.0) -> Optional[Frame]: ...

    def close(self) -> None: ...


class UsbRecord3DFrameSource:
    """Direct Record3D USB ingest (no ROS)."""

    def __init__(self, app, dev_idx: int = 0):
        self.app = app
        self.dev_idx = int(dev_idx)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.app.connect_to_device(dev_idx=self.dev_idx)
            self._started = True

    def next(self, timeout: float = 1.0) -> Optional[Frame]:
        rgb, depth, K4, pose = self.app.get_frame_data()
        if rgb is None:
            return None
        K = K4.cpu().numpy()[:3, :3] if torch.is_tensor(K4) else np.asarray(K4)[:3, :3]
        return Frame(
            rgb=rgb,
            depth=depth,
            K=np.asarray(K, dtype=np.float32),
            pose=np.asarray(pose, dtype=np.float32),
            t=float(time.time()),
        )

    def close(self) -> None:
        return None
