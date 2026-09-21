"""Live USB Record3D → YOLO/YOLOE detection. Edit the constants; no CLI.

OpenCV in this env has no imshow GUI, so the overlay uses matplotlib.
Ultralytics forces the Agg backend on import, so we switch back to TkAgg.
Prints det counts at several conf cutoffs so you can tune CONF without remapping.
"""

from pathlib import Path
from threading import Event
import os
import time

# Wayland + Tk/Qt: prefer X11 so a window actually appears.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("GDK_BACKEND", "x11")

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE
from record3d import Record3DStream

import matplotlib

matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

# --- edit these ---
DEV_IDX = 0
# yoloe | yoloe_pf | yolo11 | yolo26_seg
DETECTOR = "yolo26_seg"
CONF = 0.1
DEVICE = "cuda"
PRINT_THRESHOLDS = (0.10, 0.2, 0.5, 0.70)
REPO_ROOT = Path("/ws") #/home/blinky/ZJ-WS/concept-graphs")
CG_DIR = REPO_ROOT / "conceptgraph"
CLASSES_FILE = CG_DIR / "scannet200_classes.txt"
WEIGHTS = {
    "yoloe": CG_DIR / "yoloe-11l-seg.pt",
    "yoloe_pf": CG_DIR / "yoloe-11m-seg-pf.pt",
    "yolo11": CG_DIR / "yolo11l.pt",
    "yolo26_seg": CG_DIR / "yolo26s-seg.pt",
}
# --- end edit ---


class Stream:
    def __init__(self):
        self.event = Event()
        self.session = None

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect(self, dev_idx):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        print(f"{len(devs)} device(s) found")
        for dev in devs:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}")
        if len(devs) <= dev_idx:
            raise RuntimeError(f"Cannot connect to device #{dev_idx}")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[dev_idx])

    def get_rgb(self):
        """Block until a frame arrives, same as record_3d_stream.py."""
        if not self.event.wait(timeout=2.0):
            return None
        rgb = self.session.get_rgb_frame()
        _ = self.session.get_depth_frame()
        if self.session.get_device_type() == 0:  # TrueDepth
            rgb = cv2.flip(rgb, 1)
        self.event.clear()
        return rgb


def load_class_names(path: Path):
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise SystemExit(f"no class names in {path}")
    return names


def resolve_weights(path: Path):
    if path.exists():
        return str(path)
    return path.name


def load_model(detector: str):
    weights = resolve_weights(WEIGHTS[detector])
    if detector in ("yolo11", "yolo26_seg"):
        model = YOLO(weights)
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    else:
        model = YOLOE(weights)
        model.eval()
        if detector == "yoloe":
            names = load_class_names(CLASSES_FILE)
            if hasattr(model, "get_text_pe"):
                model.set_classes(names, model.get_text_pe(names))
            else:
                model.set_classes(names)
        else:
            names = (
                [model.names[i] for i in range(len(model.names))]
                if isinstance(model.names, dict)
                else list(model.names)
            )
    model.to(DEVICE)
    print(f"detector={detector}  weights={weights}  classes={len(names)}  conf={CONF}")
    return model


def counts_at_thresholds(confidences: np.ndarray):
    parts = []
    for t in PRINT_THRESHOLDS:
        n = int((confidences >= t).sum()) if confidences.size else 0
        parts.append(f"@{t:.2f}={n}")
    return "  ".join(parts)


def open_window():
    print(f"matplotlib backend={plt.get_backend()}  DISPLAY={os.environ.get('DISPLAY')!r}")
    if plt.get_backend().lower() == "agg":
        raise SystemExit(
            "Matplotlib backend is Agg (no window). Install python3-tk or set MPLBACKEND=TkAgg."
        )
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_title("waiting for Record3D frame...")
    ax.axis("off")
    fig.tight_layout()
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.2)
    return fig, ax


def main():
    if DETECTOR not in WEIGHTS:
        raise SystemExit(f"DETECTOR must be one of {list(WEIGHTS)}")

    torch.set_grad_enabled(False)
    print("Loading detector (window opens after this)...")
    model = load_model(DETECTOR)
    predict_conf = min(CONF, min(PRINT_THRESHOLDS))

    fig, ax = open_window()
    im_artist = None

    stream = Stream()
    stream.connect(DEV_IDX)
    print(
        f"Ctrl+C to stop. Overlay conf>={predict_conf:.2f}; "
        "printed counts are how many boxes remain at each cutoff."
    )
    print("Waiting for USB frames (keep Record3D open on the phone)...")

    n = 0
    waits = 0
    try:
        while True:
            rgb = stream.get_rgb()
            if rgb is None:
                waits += 1
                print(f"no frame yet ({waits}); is Record3D streaming?")
                plt.pause(0.05)
                continue
            n += 1
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model.predict(bgr, conf=predict_conf, device=DEVICE, verbose=False)[0]
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            boxes = result.boxes
            if boxes is None or boxes.xyxy.numel() == 0:
                confidences = np.array([], dtype=np.float32)
                vis_rgb = rgb
            else:
                confidences = boxes.conf.cpu().numpy()
                vis_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            n_all = int(confidences.size)
            conf_stats = ""
            if n_all:
                conf_stats = (
                    f"  conf[min={confidences.min():.2f} mean={confidences.mean():.2f} "
                    f"max={confidences.max():.2f}]"
                )
            print(
                f"frame {n:4d}  dets={n_all}{conf_stats}  "
                f"{counts_at_thresholds(confidences)}  yolo={dt_ms:6.1f} ms"
            )

            title = f"{DETECTOR}  conf>={predict_conf:.2f}  n={n_all}"
            if im_artist is None:
                im_artist = ax.imshow(vis_rgb)
                ax.set_title(title)
            else:
                im_artist.set_data(vis_rgb)
                im_artist.set_extent((0, vis_rgb.shape[1], vis_rgb.shape[0], 0))
                ax.set_title(title)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
