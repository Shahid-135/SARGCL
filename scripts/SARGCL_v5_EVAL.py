#!/usr/bin/env python3
"""
SARGCL v5 evaluation (no training).

Usage (from /home/nagendra/TMM):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 SARGCL_v5_EVAL.py \
        sargcl_results_v5 \
        sargcl_50epoch_results_monotau_fixedtau:fixed \
        sargcl_results_v5_bs4 sargcl_results_v5_bs16 sargcl_results_v5_bs32

Each argument is a results directory containing best_sargcl_clip_model_cached.pth.
Append ":fixed" for a model trained with a fixed temperature (w/o DRA).
The FIRST directory is the reference model: its OT costs define the
easy/medium/hard bins used for every model in the difficulty table.

Outputs (per directory, in <dir>/eval_v5/):
    test_metrics.csv          Acc, Prec, Rec, F1 (positive class) + macro metrics
    test_per_sample.csv       label, prediction, confidence, OT, tau, pos/hard-neg sim,
                              margin, per-anchor contrastive penalty, valid flag
Outputs (combined, in ./v5_eval_summary/):
    all_test_metrics.csv / .tex
    difficulty_bins.csv / .tex            (Table V)
    tau_representatives.csv / .tex        (Table VII, reference model)
    contrastive_examples.csv              (candidates for Table VIII)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.environ.get("SARGCL_CACHE_DIR", "./sargcl_feature_cache")
TEST_CACHE = os.path.join(CACHE_DIR, "test_cache.pt")
CKPT_NAME = "best_sargcl_clip_model_cached.pth"
SUMMARY_DIR = "./v5_eval_summary"

EVAL_BATCH_SIZE = 8        # batch-wise OT normalization uses the training batch size
TAU_MIN, TAU_MAX, TAU_BASE = 0.05, 0.10, 0.07
OT_EPSILON, SINKHORN_ITERS = 0.07, 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATA
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu")
        self.samples = obj["samples"]
        self.clip_dim = int(obj["clip_dim"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    return {
        "gV_clip": torch.stack([x["gV_clip"] for x in batch]),
        "gT_clip": torch.stack([x["gT_clip"] for x in batch]),
        "visual_nodes": [x["visual_nodes"] for x in batch],
        "text_nodes": [x["text_nodes"] for x in batch],
        "text_hyperedges": [x["text_hyperedges"] for x in batch],
        "label": torch.stack([x["label"] for x in batch]),
        "image_paths": [x["image_path"] for x in batch],
        "raw_texts": [x["raw_text"] for x in batch],
    }


# ============================================================
# MODEL (identical trainable modules to SARGCL v4/v5 CachedSARGCL)
# ============================================================
def cosine_matrix(x, y):
    return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).t()


class VisualHypergraphRefiner(nn.Module):
    def __init__(self, clip_hidden, top_k_edges=8):
        super().__init__()
        self.top_k_edges = top_k_edges
        self.scorer = nn.Sequential(
            nn.Linear(clip_hidden * 2, clip_hidden), nn.ReLU(),
            nn.Linear(clip_hidden, 1), nn.Sigmoid())

    def forward(self, visual_node_feats, gT):
        refined = []
        for i, V in enumerate(visual_node_feats):
            n = V.size(0)
            if n <= 1:
                refined.append((V, [list(range(n))]))
                continue
            with torch.no_grad():
                sims = cosine_matrix(V, V)
                knn = 3 if n > 3 else max(1, n - 1)
                neighbors = torch.topk(sims, k=knn + 1, dim=1).indices[:, 1:]
            cands = set()
            for u in range(n):
                for v in neighbors[u]:
                    cands.add(tuple(sorted((u, int(v)))))
            scored = []
            for pair in cands:
                idxs = list(pair)
                score = self.scorer(torch.cat([V[idxs].mean(dim=0), gT[i]], dim=-1))
                scored.append((float(score.item()), idxs))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [idxs for _, idxs in scored[:self.top_k_edges]] or [list(range(n))]
            refined.append((V, top))
        return refined


class HGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, heads=2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.lin_node = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_edge = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, out_dim))
        self.dropout, self.act = nn.Dropout(dropout), nn.GELU()

    def forward(self, X, H):
        N, E = H.shape
        deg_e = H.sum(dim=0).clamp(min=1.0)
        E_feat = (H.t() @ X) / deg_e.unsqueeze(-1)
        X_h = self.lin_node(X).view(N, self.heads, self.out_dim)
        E_h = self.lin_edge(E_feat).view(E, self.heads, self.out_dim)
        HX = torch.einsum("en,nhd->ehd", H.t(), X_h) / deg_e.view(E, 1, 1)
        M = HX + E_h
        alpha = self.dropout(torch.softmax(torch.einsum("ehd,hd->eh", M, self.attn), dim=0))
        out = torch.einsum("ne,eh,ehd->nhd", H, alpha, M)
        return self.act(out.reshape(N, self.heads * self.out_dim))


class HGATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layers=2, heads=2):
        super().__init__()
        mods = [HGATLayer(in_dim, hid_dim, heads=heads)]
        for _ in range(layers - 2):
            mods.append(HGATLayer(hid_dim * heads, hid_dim, heads=heads))
        mods.append(HGATLayer(hid_dim * heads, out_dim // heads, heads=heads))
        self.layers = nn.ModuleList(mods)

    def forward(self, X, H):
        for layer in self.layers:
            X = layer(X, H)
        return X


def sinkhorn_knopp(C, epsilon=OT_EPSILON, n_iters=SINKHORN_ITERS):
    K = torch.exp(-C / epsilon)
    n, m = K.shape
    r = torch.ones(n, device=K.device) / n
    c = torch.ones(m, device=K.device) / m
    u = torch.ones(n, device=K.device) / n
    v = torch.ones(m, device=K.device) / m
    for _ in range(n_iters):
        u = r / (K @ v + 1e-8)
        v = c / (K.t() @ u + 1e-8)
    return torch.diag(u) @ K @ torch.diag(v)


def ot_cost(Vn, Tn):
    C = 1.0 - cosine_matrix(Vn, Tn).clamp(-1, 1)
    return (sinkhorn_knopp(C) * C).sum()


class EvalSARGCL(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.visual_refiner = VisualHypergraphRefiner(d)
        self.hgat_visual = HGATEncoder(d, d, d, 2)
        self.hgat_text = HGATEncoder(d, d, d, 2)
        self.uncert_mlp = nn.Sequential(
            nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, 4), nn.Softplus())
        self.classifier = nn.Sequential(
            nn.Linear(d * 3, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, 2))

    def incidence(self, n, hyperedges):
        E = len(hyperedges)
        H = torch.zeros(n, max(E, 1), device=DEVICE)
        if E == 0:
            H[:, 0] = 1.0
            return H
        for e, nodes in enumerate(hyperedges):
            for k in nodes:
                if 0 <= int(k) < n:
                    H[int(k), e] = 1.0
        return H

    def forward(self, batch):
        gV = batch["gV_clip"].to(DEVICE)
        gT = batch["gT_clip"].to(DEVICE)
        vis = [x.to(DEVICE) for x in batch["visual_nodes"]]
        txt = [x.to(DEVICE) for x in batch["text_nodes"]]
        refined = self.visual_refiner(vis, gT)
        zV_nodes, zT_nodes, zV, zT = [], [], [], []
        for i in range(len(vis)):
            V_i, edges = refined[i]
            zv = (self.hgat_visual(V_i, self.incidence(V_i.size(0), edges))
                  if V_i.size(0) > 0 else torch.zeros(0, self.d, device=DEVICE))
            T_i = txt[i]
            zt = (self.hgat_text(T_i, self.incidence(T_i.size(0), batch["text_hyperedges"][i]))
                  if T_i.size(0) > 0 else torch.zeros(0, self.d, device=DEVICE))
            zV_nodes.append(zv)
            zT_nodes.append(zt)
            zV.append(zv.mean(0) if zv.size(0) else torch.zeros(self.d, device=DEVICE))
            zT.append(zt.mean(0) if zt.size(0) else torch.zeros(self.d, device=DEVICE))
        zV, zT = torch.stack(zV), torch.stack(zT)
        alphas = self.uncert_mlp(torch.cat([zV, zT], -1)) + 1e-3
        w = alphas / alphas.sum(-1, keepdim=True)
        zh = w[:, 0:1] * zV + w[:, 1:2] * zT + w[:, 2:3] * gV + w[:, 3:4] * gT
        logits = self.classifier(torch.cat([zh, gV, gT], -1))
        return logits, zV, zT, zV_nodes, zT_nodes


def temperature(ot, valid, fixed):
    if fixed:
        return torch.full_like(ot, TAU_BASE)
    if int(valid.sum()) >= 2:
        v = ot[valid]
        d = ((ot - v.min()) / (v.max() - v.min() + 1e-6)).clamp(0, 1)
    else:
        d = torch.full_like(ot, 0.5)
    d = torch.where(valid, d, torch.ones_like(d))
    return TAU_MIN + (TAU_MAX - TAU_MIN) * d


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(run_dir, fixed, loader, clip_dim):
    ckpt_path = os.path.join(run_dir, CKPT_NAME)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    model = EvalSARGCL(clip_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{run_dir}: missing keys {missing}")
    if unexpected:
        print(f"[warn] {run_dir}: unexpected keys ignored: {unexpected}")
    model.eval()
    print(f"\n[{run_dir}] checkpoint epoch={ckpt.get('epoch')} val_f1={ckpt.get('val_f1')} fixed_tau={fixed}")

    rows, offset = [], 0
    for batch in tqdm(loader, desc=f"Eval {run_dir}"):
        logits, zV, zT, zVn, zTn = model(batch)
        B = logits.size(0)
        probs = torch.softmax(logits, -1)
        valid = torch.tensor([zVn[i].size(0) > 0 and zTn[i].size(0) > 0 for i in range(B)],
                             dtype=torch.bool, device=DEVICE)
        ot = torch.zeros(B, device=DEVICE)
        for i in range(B):
            if valid[i]:
                ot[i] = ot_cost(zVn[i], zTn[i])
        tau = temperature(ot, valid, fixed)
        sim = cosine_matrix(zV, zT)
        logit_rows = sim / tau.unsqueeze(1)
        logit_rows = logit_rows.masked_fill((~valid).unsqueeze(0), -1e9)
        penalty = -torch.log_softmax(logit_rows, dim=1).diagonal()
        for i in range(B):
            vi = bool(valid[i])
            negs = [j for j in range(B) if j != i and bool(valid[j])]
            pos = float(sim[i, i])
            hard = max([float(sim[i, j]) for j in negs], default=float("nan"))
            rows.append({
                "sample_index": offset + i,
                "image_path": batch["image_paths"][i],
                "text": batch["raw_texts"][i],
                "label": int(batch["label"][i]),
                "prediction": int(probs[i].argmax()),
                "confidence": float(probs[i].max()),
                "valid_ot": int(vi),
                "visual_nodes": int(zVn[i].size(0)),
                "text_nodes": int(zTn[i].size(0)),
                "ot_cost": float(ot[i]) if vi else np.nan,
                "tau": float(tau[i]) if vi else np.nan,
                "positive_similarity": pos if vi else np.nan,
                "hardest_negative_similarity": hard if vi else np.nan,
                "contrastive_margin": (pos - hard) if vi else np.nan,
                "penalty": float(penalty[i]) if (vi and len(negs) > 0) else np.nan,
            })
        offset += B

    df = pd.DataFrame(rows)
    out_dir = os.path.join(run_dir, "eval_v5")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "test_per_sample.csv"), index=False)

    y, p = df["label"].values, df["prediction"].values
    P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    mP, mR, mF, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    sv = df[df.valid_ot == 1]
    metrics = {
        "run": run_dir, "fixed_tau": int(fixed),
        "best_epoch": ckpt.get("epoch"), "best_val_f1": ckpt.get("val_f1"),
        "test_samples": len(df), "valid_ot_samples": len(sv),
        "acc": accuracy_score(y, p), "prec": P, "rec": R, "f1": F,
        "macro_prec": mP, "macro_rec": mR, "macro_f1": mF,
        "mean_tau": sv["tau"].mean(), "min_tau": sv["tau"].min(), "max_tau": sv["tau"].max(),
        "corr_ot_tau": sv["ot_cost"].corr(sv["tau"]) if sv["tau"].std() > 0 else np.nan,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    print(f"  Acc={metrics['acc']:.4f} P={P:.4f} R={R:.4f} F1={F:.4f} | macro-F1={mF:.4f}")
    return df, metrics


def f1_pos(df):
    if len(df) == 0:
        return np.nan
    return precision_recall_fscore_support(df.label, df.prediction, average="binary",
                                           zero_division=0)[2]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    runs = [(a.split(":")[0], a.endswith(":fixed")) for a in args]
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    ds = CachedDataset(TEST_CACHE)
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate)

    results, metrics = {}, []
    for run_dir, fixed in runs:
        df, m = evaluate(run_dir, fixed, loader, ds.clip_dim)
        results[run_dir] = df
        metrics.append(m)

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(os.path.join(SUMMARY_DIR, "all_test_metrics.csv"), index=False)
    t = mdf[["run", "best_epoch", "acc", "prec", "rec", "f1", "macro_f1"]].copy()
    for c in ["acc", "prec", "rec", "f1", "macro_f1"]:
        t[c] = (100 * t[c]).map(lambda x: f"{x:.2f}")
    t.to_latex(os.path.join(SUMMARY_DIR, "all_test_metrics.tex"), index=False)

    # ---------------- Table V: difficulty bins (reference OT) ----------------
    ref_name = runs[0][0]
    ref = results[ref_name]
    rv = ref[ref.valid_ot == 1].copy()
    q = rv["ot_cost"].rank(method="first", pct=True)
    rv["bin"] = np.where(q <= 1 / 3, "Easy (Low OT)", np.where(q <= 2 / 3, "Medium", "Hard (High OT)"))
    bin_of = dict(zip(rv.sample_index, rv["bin"]))
    rows = []
    for name in ["Easy (Low OT)", "Medium", "Hard (High OT)", "Overall (valid)", "Overall (all)"]:
        row = {"Difficulty": name}
        for run_dir, _ in runs:
            df = results[run_dir]
            if name == "Overall (all)":
                sub = df
            elif name == "Overall (valid)":
                sub = df[df.sample_index.isin(bin_of.keys())]
            else:
                sub = df[df.sample_index.map(bin_of) == name]
            row[run_dir] = 100 * f1_pos(sub)
            row["n"] = len(sub)
        rows.append(row)
    bins = pd.DataFrame(rows)
    bins.to_csv(os.path.join(SUMMARY_DIR, "difficulty_bins.csv"), index=False)
    bt = bins.copy()
    for run_dir, _ in runs:
        bt[run_dir] = bt[run_dir].map(lambda x: f"{x:.2f}")
    bt.to_latex(os.path.join(SUMMARY_DIR, "difficulty_bins.tex"), index=False)
    print("\nDifficulty bins (F1, %):")
    print(bins.to_string(index=False))

    # sanity: overall F1 must lie between the bin F1 values
    for run_dir, _ in runs:
        b = bins.set_index("Difficulty")[run_dir]
        lo = b[["Easy (Low OT)", "Medium", "Hard (High OT)"]].min()
        hi = b[["Easy (Low OT)", "Medium", "Hard (High OT)"]].max()
        ok = lo - 1e-9 <= b["Overall (valid)"] <= hi + 1e-9
        print(f"  [check] {run_dir}: overall(valid) within bin range -> {'OK' if ok else 'FAIL'}")

    # ---------------- Table VII: tau representatives (reference) ----------------
    reps = []
    for name in ["Hard (High OT)", "Medium", "Easy (Low OT)"]:
        g = rv[rv["bin"] == name].copy()
        if g.empty:
            continue
        g["dist"] = (g.ot_cost - g.ot_cost.median()).abs()
        reps.append(g.sort_values("dist").head(2))
    rep = pd.concat(reps, ignore_index=True)
    keep = ["bin", "sample_index", "image_path", "text", "label", "prediction", "confidence",
            "ot_cost", "tau", "positive_similarity", "hardest_negative_similarity",
            "contrastive_margin", "penalty"]
    rep[keep].to_csv(os.path.join(SUMMARY_DIR, "tau_representatives.csv"), index=False)
    rt = rep[["bin", "confidence", "ot_cost", "tau", "positive_similarity",
              "hardest_negative_similarity", "contrastive_margin"]].copy()
    rt.columns = ["Difficulty", "Conf.", "OT Cost", r"$\tau_i$", "Pos Sim", "Hard Neg.", "C. Margin"]
    for c in rt.columns[1:]:
        rt[c] = rt[c].map(lambda x: f"{x:.4f}")
    rt.to_latex(os.path.join(SUMMARY_DIR, "tau_representatives.tex"), index=False, escape=False)
    print("\nTau representatives (reference model):")
    print(rep[["bin", "ot_cost", "tau", "positive_similarity", "hardest_negative_similarity"]].to_string(index=False))

    # ---------------- Table VIII candidates ----------------
    cand = pd.concat([
        rv.sort_values("contrastive_margin").head(3),       # hard negative beats positive
        rv.sort_values("penalty", ascending=False).head(3),  # largest penalties
        rv.sort_values("penalty").head(3),                   # easiest pairs
    ]).drop_duplicates("sample_index")
    cand[keep].to_csv(os.path.join(SUMMARY_DIR, "contrastive_examples.csv"), index=False)

    print(f"\nSaved summary files in {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
