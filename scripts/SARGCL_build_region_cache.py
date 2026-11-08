#!/usr/bin/env python3
"""
Build a REGION-FEATURE cache for SARGCL.

The existing cache (./sargcl_feature_cache) represents each detected region by
the CLIP *text* embedding of its BLIP caption. Diagnostics showed these nodes
carry less information than the global CLIP embedding. This script keeps
everything in the existing cache (global CLIP features, textual nodes,
hyperedges, labels, captions) and only replaces the visual nodes by the CLIP
*image* embeddings of the detected region crops.

Detection uses the same settings as the original pipeline (torchvision
Faster R-CNN ResNet-50 FPN, score >= 0.10, at most 12 regions), so the number
of regions should match the existing cache; mismatches are reported.

Usage (from /home/nagendra/TMM):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 SARGCL_build_region_cache.py
Output: ./sargcl_feature_cache_region/{train,validation,test}_cache.pt
"""
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import clip

SRC_DIR = "./sargcl_feature_cache"
DST_DIR = "./sargcl_feature_cache_region"
SPLITS = ["train", "validation", "test"]
SCORE_THRESHOLD = 0.10
MAX_NODES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def detect(detector, img):
    pred = detector([transforms.functional.to_tensor(img).to(DEVICE)])[0]
    boxes, scores = pred["boxes"], pred["scores"]
    keep = scores >= SCORE_THRESHOLD
    boxes, scores = boxes[keep], scores[keep]
    if boxes.size(0) > MAX_NODES:
        boxes = boxes[torch.topk(scores, k=MAX_NODES).indices]
    return boxes.cpu()


def crops_from(img, boxes):
    crops, kept = [], []
    for b in boxes:
        x1, y1, x2, y2 = b.tolist()
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        w, h = crop.size
        if w < 2 or h < 2:
            crop = crop.resize((max(2, w), max(2, h)), Image.Resampling.BILINEAR)
        crops.append(crop)
        kept.append((int(x1), int(y1), int(x2), int(y2)))
    return crops, kept


@torch.no_grad()
def encode_crops(clip_model, preprocess, crops):
    if not crops:
        return torch.zeros(0, clip_model.visual.output_dim)
    x = torch.stack([preprocess(c) for c in crops]).to(DEVICE)
    f = clip_model.encode_image(x).float()
    return F.normalize(f, dim=-1).cpu()


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    clip_model.float().eval()

    for split in SPLITS:
        src = os.path.join(SRC_DIR, f"{split}_cache.pt")
        dst = os.path.join(DST_DIR, f"{split}_cache.pt")
        if os.path.exists(dst):
            print(f"[skip] {dst} already exists")
            continue
        obj = torch.load(src, map_location="cpu")
        samples = obj["samples"]
        new_samples, mismatch, zero = [], 0, 0
        t0 = time.time()
        for s in tqdm(samples, desc=f"Region features {split}"):
            img = Image.open(s["image_path"]).convert("RGB")
            boxes = detect(detector, img)
            crops, kept = crops_from(img, boxes)
            feats = encode_crops(clip_model, preprocess, crops)
            if feats.size(0) != s["visual_nodes"].size(0):
                mismatch += 1
            zero += int(feats.size(0) == 0)
            ns = dict(s)
            ns["caption_visual_nodes"] = s["visual_nodes"]   # keep the old nodes
            ns["visual_nodes"] = feats                        # CLIP image features of crops
            ns["visual_boxes"] = kept
            new_samples.append(ns)
        out = dict(obj)
        out["samples"] = new_samples
        out["visual_node_type"] = "clip_image_region_crops"
        torch.save(out, dst)
        n = len(samples)
        print(f"{split}: {n} samples | zero-region samples {100 * zero / n:.2f}% | "
              f"region-count mismatch vs. caption cache {mismatch} ({100 * mismatch / n:.2f}%) | "
              f"{(time.time() - t0) / 60:.1f} min -> {dst}")


if __name__ == "__main__":
    main()
