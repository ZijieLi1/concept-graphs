"""USB Record3D smoke test. pip record3d is enough; OpenCV here has no imshow GUI.

Prints RGB/depth FPS. Use --headless to skip matplotlib (otherwise the plot
refresh is usually slower than the USB stream).

    python3 test_sctipts/record_3d_stream.py --headless
"""

from threading import Event
import sys
import time

import cv2
import numpy as np
from record3d import Record3DStream

# --- edit these ---
DEV_IDX = 0
PRINT_EVERY = 30
# --- end edit ---

HEADLESS = "--headless" in sys.argv or "-n" in sys.argv


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

    def connect_to_device(self, dev_idx, timeout_s=20.0):
        print("Searching for devices")
        deadline = time.monotonic() + float(timeout_s)
        devs = []
        while True:
            devs = Record3DStream.get_connected_devices()
            if len(devs) > int(dev_idx):
                break
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            print(
                f"0 device(s) found — waiting for usbmuxd ({remain:.0f}s). "
                "Plug in, unlock, trust, Record3D USB streaming."
            )
            time.sleep(min(0.5, remain))
        print(f"{len(devs)} device(s) found")
        for dev in devs:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}\n")
        if len(devs) <= dev_idx:
            raise RuntimeError(
                f"Cannot connect to device #{dev_idx}. "
                "usbmuxd has no iPhone — replug, unlock, trust, start USB streaming. "
                "In Docker, recreate the container after compose changes (/host-run)."
            )
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
        rgb_im = depth_im = fig = axes = None
        if not HEADLESS:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        n = 0
        t0 = time.perf_counter()
        t_window = t0
        print(f"measuring FPS  headless={HEADLESS}  print every {PRINT_EVERY} frames")

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
            now = time.perf_counter()
            if n == 1 or n % PRINT_EVERY == 0:
                elapsed = now - t0
                fps = n / elapsed if elapsed > 0 else 0.0
                window_n = 1 if n == 1 else PRINT_EVERY
                window_fps = window_n / (now - t_window) if now > t_window else 0.0
                t_window = now
                print(
                    f"frame {n}  rgb={tuple(rgb.shape)}  depth={tuple(depth.shape)}  "
                    f"fps={fps:.1f}  window={window_fps:.1f}\n"
                    f"K=\n{intrinsic_mat}\n"
                    f"pose t=({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})  "
                    f"q=({pose.qx:.3f}, {pose.qy:.3f}, {pose.qz:.3f}, {pose.qw:.3f})"
                )

            if not HEADLESS:
                import matplotlib.pyplot as plt

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
    try:
        app.start_processing_stream()
    except KeyboardInterrupt:
        print("\nstopped")
