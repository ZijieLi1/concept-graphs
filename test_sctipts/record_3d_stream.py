"""USB Record3D smoke test. pip record3d is enough; OpenCV here has no imshow GUI."""

from threading import Event

import cv2
import matplotlib.pyplot as plt
import numpy as np
from record3d import Record3DStream

# --- edit these ---
DEV_IDX = 0
PRINT_EVERY = 30
# --- end edit ---


class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_to_device(self, dev_idx):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        print(f"{len(devs)} device(s) found")
        for dev in devs:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}\n")
        if len(devs) <= dev_idx:
            raise RuntimeError(f"Cannot connect to device #{dev_idx}, try a different index.")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[dev_idx])

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [
                [coeffs.fx, 0, coeffs.tx],
                [0, coeffs.fy, coeffs.ty],
                [0, 0, 1],
            ]
        )

    def start_processing_stream(self):
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        rgb_im = depth_im = None
        n = 0

        while True:
            self.event.wait()
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            pose = self.session.get_camera_pose()

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            n += 1
            if n == 1 or n % PRINT_EVERY == 0:
                print(
                    f"frame {n}  rgb={rgb.shape}  depth={depth.shape}\n"
                    f"K=\n{intrinsic_mat}\n"
                    f"pose t=({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})  "
                    f"q=({pose.qx:.3f}, {pose.qy:.3f}, {pose.qz:.3f}, {pose.qw:.3f})"
                )

            depth_vis = np.nan_to_num(depth, nan=0.0)
            dmax = float(depth_vis.max()) if depth_vis.size else 1.0
            if dmax > 0:
                depth_vis = depth_vis / dmax

            if rgb_im is None:
                rgb_im = axes[0].imshow(rgb)
                depth_im = axes[1].imshow(depth_vis, cmap="turbo")
                axes[0].set_title("RGB")
                axes[1].set_title("Depth")
                axes[0].axis("off")
                axes[1].axis("off")
                fig.tight_layout()
            else:
                rgb_im.set_data(rgb)
                depth_im.set_data(depth_vis)
            fig.canvas.draw_idle()
            plt.pause(0.001)

            self.event.clear()


if __name__ == "__main__":
    app = DemoApp()
    app.connect_to_device(dev_idx=DEV_IDX)
    app.start_processing_stream()
