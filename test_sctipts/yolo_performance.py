"""Local YOLO + CLIP crop timing. Edit the constants below; nothing is CLI."""

from pathlib import Path
import time

import numpy as np
import open_clip
import torch
from PIL import Image
from ultralytics import YOLO, YOLOE

# --- edit these ---
IMAGE_DIR = Path(
    "/ws/Record3D/apartment_living_room_preprocessed/rgb"
)
MODEL_PATH = Path("/ws/conceptgraph/yoloe-11m-seg-pf.pt")
# Optional text prompts for YOLOE (leave empty for prompt-free / YOLO11).
CLASSES = []
DEVICE = "cuda"
CONF = 0.1
WARMUP = 3
MAX_IMAGES = 100
STRIDE = 10
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
RUN_CLIP = True
CLIP_MODEL = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
CLIP_CROP_PAD = 20
# --- end edit ---


def load_yolo(path: Path):
    if "yoloe" in path.name.lower():
        model = YOLOE(str(path))
        model.eval()
        if CLASSES:
            model.set_classes(list(CLASSES))
    else:
        model = YOLO(str(path))
    return model


def list_images(folder: Path):
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    imgs.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return imgs[::STRIDE][:MAX_IMAGES]


def sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def summarize(name, times):
    t = np.array(times, dtype=np.float64)
    if len(t) == 0:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: n={len(t)}  mean={t.mean()*1000:.1f} ms  std={t.std()*1000:.1f} ms  "
        f"p50={np.percentile(t, 50)*1000:.1f} ms  p95={np.percentile(t, 95)*1000:.1f} ms  "
        f"fps={1.0 / t.mean():.2f}"
    )


def clip_encode_boxes(image: Image.Image, xyxy, clip_model, clip_preprocess):
    if xyxy is None or len(xyxy) == 0:
        return 0
    w, h = image.size
    batch = []
    for x_min, y_min, x_max, y_max in xyxy:
        pad_l = min(CLIP_CROP_PAD, x_min)
        pad_t = min(CLIP_CROP_PAD, y_min)
        pad_r = min(CLIP_CROP_PAD, w - x_max)
        pad_b = min(CLIP_CROP_PAD, h - y_max)
        crop = image.crop((x_min - pad_l, y_min - pad_t, x_max + pad_r, y_max + pad_b))
        batch.append(clip_preprocess(crop).unsqueeze(0))
    x = torch.cat(batch, dim=0).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return int(feats.shape[0])


def main():
    images = list_images(IMAGE_DIR)
    if not images:
        raise SystemExit(f"no images in {IMAGE_DIR}")

    print(f"yolo:   {MODEL_PATH}")
    print(f"images: {len(images)} from {IMAGE_DIR} (stride={STRIDE}, max={MAX_IMAGES})")
    print(f"device: {DEVICE}  conf={CONF}  clip={RUN_CLIP}")

    yolo = load_yolo(MODEL_PATH)
    clip_model = clip_preprocess = None
    if RUN_CLIP:
        print(f"clip:   {CLIP_MODEL} {CLIP_PRETRAINED}")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, CLIP_PRETRAINED
        )
        clip_model = clip_model.eval().to(DEVICE)

    print(f"warmup {WARMUP} ...")
    for p in images[:WARMUP]:
        results = yolo.predict(str(p), conf=CONF, device=DEVICE, verbose=False)
        if RUN_CLIP:
            boxes = results[0].boxes
            xyxy = None if boxes is None else boxes.xyxy.detach().cpu().numpy()
            clip_encode_boxes(Image.open(p).convert("RGB"), xyxy, clip_model, clip_preprocess)
    sync()

    yolo_times, clip_times, total_times, n_dets = [], [], [], []
    for i, p in enumerate(images):
        image = Image.open(p).convert("RGB")

        sync()
        t0 = time.perf_counter()
        results = yolo.predict(str(p), conf=CONF, device=DEVICE, verbose=False)
        sync()
        t_yolo = time.perf_counter() - t0

        boxes = results[0].boxes
        n = 0 if boxes is None else int(boxes.xyxy.shape[0])
        xyxy = None if boxes is None else boxes.xyxy.detach().cpu().numpy()

        t_clip = 0.0
        if RUN_CLIP:
            sync()
            t1 = time.perf_counter()
            clip_encode_boxes(image, xyxy, clip_model, clip_preprocess)
            sync()
            t_clip = time.perf_counter() - t1

        yolo_times.append(t_yolo)
        clip_times.append(t_clip)
        total_times.append(t_yolo + t_clip)
        n_dets.append(n)
        print(
            f"[{i+1:4d}/{len(images)}] {p.name:20s}  "
            f"yolo={t_yolo*1000:7.1f} ms  clip={t_clip*1000:7.1f} ms  "
            f"sum={((t_yolo + t_clip)*1000):7.1f} ms  dets={n}"
        )

    print()
    summarize("yolo ", yolo_times)
    if RUN_CLIP:
        summarize("clip ", clip_times)
        summarize("total", total_times)
        clip_arr = np.array(clip_times)
        det_arr = np.array(n_dets, dtype=np.float64)
        per_det = clip_arr[det_arr > 0] / det_arr[det_arr > 0]
        if len(per_det):
            print(f"clip per det: mean={per_det.mean()*1000:.1f} ms")
    print(f"dets/image mean={np.mean(n_dets):.1f}")


if __name__ == "__main__":
    main()
