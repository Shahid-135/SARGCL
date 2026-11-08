#!/usr/bin/env python3
"""
Complementarity analysis (no GPU, no training).

Question: when the global-CLIP model is wrong, how often is the structured
(hypergraph-only) model right? This bounds how much ANY fusion of the two
branches could gain.

Inputs: test_per_sample.csv written by SARGCL_v5_EVAL.py for
    sargcl_results_v5                   (full SARGCL; also provides OT bins)
    sargcl_results_v5_global_only
    sargcl_results_v5_structured_only

Usage (from /home/nagendra/TMM):
    python3 SARGCL_complementarity.py
Output: complementarity/complementarity_summary.txt (+ CSVs)
"""
import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

RUNS = {
    "full": "sargcl_results_v5",
    "global": "sargcl_results_v5_global_only",
    "structured": "sargcl_results_v5_structured_only",
}
OUT = os.environ.get("COMP_OUT", "complementarity")


def load(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "eval_v5", "test_per_sample.csv"))
    df = df.sort_values("sample_index").reset_index(drop=True)
    # probability of the positive class from the predicted-class confidence
    df["p1"] = np.where(df.prediction == 1, df.confidence, 1.0 - df.confidence)
    df["correct"] = (df.prediction == df.label).astype(int)
    return df


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def metrics(y, p):
    return 100 * accuracy_score(y, p), 100 * f1_score(y, p, average="binary")


def main():
    import sys
    if len(sys.argv) == 4:   # optional: full_dir global_dir structured_dir
        RUNS["full"], RUNS["global"], RUNS["structured"] = sys.argv[1:4]
    os.makedirs(OUT, exist_ok=True)
    d = {k: load(v) for k, v in RUNS.items()}
    y = d["full"].label.values
    for k in d:
        assert (d[k].label.values == y).all(), f"label order mismatch in {k}"

    g, s, f = d["global"], d["structured"], d["full"]
    lines = []
    def out(x=""):
        print(x)
        lines.append(x)

    out("=" * 72)
    out("COMPLEMENTARITY: global-only vs structured-only (CrisisMMD test)")
    out("=" * 72)
    for k in ["full", "global", "structured"]:
        a, f1 = metrics(y, d[k].prediction.values)
        out(f"{k:11s} Acc={a:6.2f}  F1={f1:6.2f}  errors={int((1 - d[k].correct).sum())}")

    # 2x2 contingency of correctness
    gc, sc = g.correct.values, s.correct.values
    both_right = int(((gc == 1) & (sc == 1)).sum())
    g_only = int(((gc == 1) & (sc == 0)).sum())
    s_only = int(((gc == 0) & (sc == 1)).sum())
    both_wrong = int(((gc == 0) & (sc == 0)).sum())
    out("\nCorrectness contingency (rows: global, cols: structured)")
    out(f"  both right          : {both_right}")
    out(f"  global right only   : {g_only}")
    out(f"  structured right only: {s_only}   <- errors of global that structured fixes")
    out(f"  both wrong          : {both_wrong}")
    g_err = s_only + both_wrong
    rescue = 100 * s_only / max(1, g_err)
    out(f"\nGlobal errors: {g_err}; structured is right on {s_only} of them ({rescue:.1f}%)")

    # Oracle: per-sample choose whichever model is right
    oracle_pred = np.where(gc == 1, g.prediction.values, s.prediction.values)
    oa, of1 = metrics(y, oracle_pred)
    ga, gf1 = metrics(y, g.prediction.values)
    out(f"Oracle selector (upper bound): Acc={oa:.2f}  F1={of1:.2f}  "
        f"(+{oa - ga:.2f} Acc / +{of1 - gf1:.2f} F1 over global)")

    # Simple late fusion, fixed weight (no tuning on test)
    for w in [0.5, 0.7, 0.9]:
        pf = (w * g.p1 + (1 - w) * s.p1 >= 0.5).astype(int).values
        a, f1 = metrics(y, pf)
        out(f"Late fusion  p = {w:.1f}*global + {1 - w:.1f}*structured : Acc={a:.2f}  F1={f1:.2f}")
    out("  (weights 0.7/0.9 are shown for reference only; choosing a weight on the test set is not valid)")

    # Is full SARGCL different from global-only? McNemar
    fc = f.correct.values
    b = int(((fc == 1) & (gc == 0)).sum())
    c = int(((fc == 0) & (gc == 1)).sum())
    out(f"\nFull vs global-only: full right & global wrong = {b}, global right & full wrong = {c}, "
        f"exact McNemar p = {mcnemar_exact(b, c):.3f}")
    agree = 100 * (f.prediction.values == g.prediction.values).mean()
    out(f"Prediction agreement full vs global-only: {agree:.2f}%")

    # Rescue rate by OT difficulty (bins from the full model's OT cost)
    v = f.valid_ot == 1
    q = f.loc[v, "ot_cost"].rank(method="first", pct=True)
    bins = pd.Series("No visual nodes", index=f.index)
    bins.loc[v] = np.where(q <= 1 / 3, "Easy", np.where(q <= 2 / 3, "Medium", "Hard"))
    rows = []
    for name in ["Easy", "Medium", "Hard", "No visual nodes"]:
        m = (bins == name).values
        ge = (gc == 0) & m
        rows.append({
            "bin": name, "n": int(m.sum()),
            "global_acc": 100 * gc[m].mean() if m.any() else np.nan,
            "structured_acc": 100 * sc[m].mean() if m.any() else np.nan,
            "global_errors": int(ge.sum()),
            "structured_fixes": int((ge & (sc == 1)).sum()),
        })
    tb = pd.DataFrame(rows)
    tb["rescue_rate_%"] = 100 * tb.structured_fixes / tb.global_errors.clip(lower=1)
    out("\nBy OT difficulty (bins from full model):")
    out(tb.round(2).to_string(index=False))
    tb.to_csv(os.path.join(OUT, "complementarity_by_bin.csv"), index=False)

    # Examples that structured fixes (for qualitative inspection)
    ex = g.loc[(gc == 0) & (sc == 1), ["sample_index", "image_path", "text", "label"]].copy()
    ex["global_conf"] = g.confidence[(gc == 0) & (sc == 1)].values
    ex["structured_conf"] = s.confidence[(gc == 0) & (sc == 1)].values
    ex.to_csv(os.path.join(OUT, "structured_fixes_examples.csv"), index=False)

    with open(os.path.join(OUT, "complementarity_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    out(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
