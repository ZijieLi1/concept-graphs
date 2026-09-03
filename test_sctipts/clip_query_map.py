"""Headless CLIP search on a saved concept-graph map. No Open3D."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

# --- edit these ---
MAP_PATH = Path(
    "/home/blinky/ZJ-WS/concept-graphs/Record3D/apartment_living_room_preprocessed/exps/yoloe_pf_exp_apt/pcd_yoloe_pf_exp_apt.pkl.gz"
)
DEVICE = "cuda"
MIN_OBS = 3
TOP_K = 8
# If non-empty, run these then exit. If empty, prompt in a loop.
QUERIES = ["couch", "chair", "tv", "backpack"]
# --- end edit ---


def bbox_center(obj):
    corners = obj.get("bbox_np")
    if corners is not None:
        return np.asarray(corners).mean(axis=0)
    pts = obj.get("pcd_np")
    if pts is not None:
        return np.asarray(pts).mean(axis=0)
    return np.full(3, np.nan)


def load_map(path: Path):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    kept = []
    for obj in data["objects"]:
        if int(obj.get("num_detections", 0)) < MIN_OBS:
            continue
        if obj.get("clip_ft") is None:
            continue
        kept.append(obj)
    clip = np.stack([np.asarray(o["clip_ft"], dtype=np.float32) for o in kept], axis=0)
    clip = clip / np.linalg.norm(clip, axis=1, keepdims=True).clip(min=1e-8)
    return kept, clip


def main():
    print(f"map: {MAP_PATH}")
    objects, clip_np = load_map(MAP_PATH)
    print(f"objects with num_detections >= {MIN_OBS}: {len(objects)}")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_ft = torch.from_numpy(clip_np).to(DEVICE)

    def search(text: str):
        with torch.no_grad():
            tokens = tokenizer([text]).to(DEVICE)
            q = clip_model.encode_text(tokens)
            q = F.normalize(q, dim=-1).squeeze(0)
            sim = F.cosine_similarity(q.unsqueeze(0), clip_ft, dim=-1)
        vals, idx = torch.topk(sim, k=min(TOP_K, len(objects)))
        print(f"\nquery: {text!r}")
        for rank, (i, s) in enumerate(zip(idx.tolist(), vals.tolist()), start=1):
            obj = objects[i]
            c = bbox_center(obj)
            print(
                f"  {rank}. sim={s:.3f}  n={obj['num_detections']:3d}  "
                f"{obj['class_name']:20s}  xyz=({c[0]:6.2f}, {c[1]:6.2f}, {c[2]:6.2f})"
            )

    if QUERIES:
        for q in QUERIES:
            search(q)
        return

    while True:
        q = input("query (empty to quit): ").strip()
        if not q:
            break
        search(q)


if __name__ == "__main__":
    main()
