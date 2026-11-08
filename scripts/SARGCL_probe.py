#!/usr/bin/env python3
"""
SARGCL representation probe (no training of SARGCL).

Question: does the structured branch carry label information that the
classifier ignores, or is there little information in it to begin with?

For every feature set below, a logistic-regression probe is trained on the
TRAIN split and evaluated on VALIDATION and TEST (positive-class F1, %):

  raw_V        mean of cached visual node embeddings (CLIP text of BLIP captions)
  raw_T        mean of cached textual node embeddings (CLIP text of tokens)
  zV, zT       mean-pooled hypergraph (HGAT) embeddings of the trained model
  zV+zT        both hypergraph embeddings
  gV, gT       global CLIP image / text embeddings
  gV+gT        both global embeddings
  zV+zT+gV+gT  everything

Samples without visual (or textual) nodes get a zero vector for that part;
the zero-node rate is reported.

Usage (from /home/nagendra/TMM):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python3 SARGCL_probe.py sargcl_results_v5
Output: <run dir>/probe/probe_results.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

import SARGCL_v5_EVAL as E   # model classes, cache loading, collate

CACHES = {
    "train": os.path.join(E.CACHE_DIR, "train_cache.pt"),
    "validation": os.path.join(E.CACHE_DIR, "validation_cache.pt"),
    "test": os.path.join(E.CACHE_DIR, "test_cache.pt"),
}


@torch.no_grad()
def extract(model, path):
    ds = E.CachedDataset(path)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=E.collate)
    out = {k: [] for k in ["raw_V", "raw_T", "zV", "zT", "gV", "gT", "y", "zeroV"]}
    d = ds.clip_dim
    for b in tqdm(loader, desc=f"Extract {os.path.basename(path)}"):
        _, zV, zT, zVn, _ = model(b)
        out["zV"].append(zV.cpu())
        out["zT"].append(zT.cpu())
        out["gV"].append(b["gV_clip"])
        out["gT"].append(b["gT_clip"])
        out["raw_V"].append(torch.stack([v.mean(0) if v.size(0) else torch.zeros(d)
                                         for v in b["visual_nodes"]]))
        out["raw_T"].append(torch.stack([t.mean(0) if t.size(0) else torch.zeros(d)
                                         for t in b["text_nodes"]]))
        out["zeroV"].append(torch.tensor([v.size(0) == 0 for v in b["visual_nodes"]]))
        out["y"].append(b["label"])
    return {k: torch.cat(v).float().numpy() if k not in ("y", "zeroV") else torch.cat(v).numpy()
            for k, v in out.items()}


FEATURE_SETS = [
    ("raw_V", ["raw_V"]), ("raw_T", ["raw_T"]),
    ("zV", ["zV"]), ("zT", ["zT"]), ("zV+zT", ["zV", "zT"]),
    ("gV", ["gV"]), ("gT", ["gT"]), ("gV+gT", ["gV", "gT"]),
    ("zV+zT+gV+gT", ["zV", "zT", "gV", "gT"]),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1]
    ds = E.CachedDataset(CACHES["test"])
    model = E.EvalSARGCL(ds.clip_dim).to(E.DEVICE)
    ckpt = torch.load(os.path.join(run_dir, E.CKPT_NAME), map_location=E.DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    data = {split: extract(model, path) for split, path in CACHES.items()}
    for split in data:
        print(f"{split}: {len(data[split]['y'])} samples, "
              f"zero visual nodes = {100 * data[split]['zeroV'].mean():.2f}%")

    rows = []
    for name, parts in FEATURE_SETS:
        X = {s: np.concatenate([data[s][p] for p in parts], axis=1) for s in data}
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        clf.fit(X["train"], data["train"]["y"])
        row = {"features": name, "dim": X["train"].shape[1]}
        for s in ["validation", "test"]:
            p = clf.predict(X[s])
            row[f"{s}_acc"] = 100 * accuracy_score(data[s]["y"], p)
            row[f"{s}_f1"] = 100 * f1_score(data[s]["y"], p, average="binary")
        rows.append(row)
        print(f"{name:14s} val F1={row['validation_f1']:.2f}  test F1={row['test_f1']:.2f}")

    out_dir = os.path.join(run_dir, "probe")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "probe_results.csv"), index=False)
    print("\n" + df.round(2).to_string(index=False))
    print(f"\nSaved {os.path.join(out_dir, 'probe_results.csv')}")


if __name__ == "__main__":
    main()
