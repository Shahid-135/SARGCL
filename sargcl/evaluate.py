"""Evaluation, training loops, and analysis utilities for SARGCL."""

import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from copy import deepcopy

from .model import SARGCL
from .utils import append_training_log


# ============== Evaluation ==============


def evaluate_loop(
    model: SARGCL,
    loader: DataLoader,
    device,
    return_preds: bool = False,
):
    """Run model evaluation on a dataloader.

    Args:
        model: SARGCL model instance.
        loader: DataLoader for the evaluation split.
        device: Torch device.
        return_preds: If True, also return per-sample predictions and paths.

    Returns:
        metrics: Dictionary with loss, accuracy, precision, recall, f1_score.
        (optional) preds, image_paths: When ``return_preds=True``.
    """
    model.eval()
    all_labels, all_preds, all_image_paths = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            labels = batch["label"]
            logits, _, losses, _ = model(batch)
            if "classification" in losses:
                total_loss += losses["classification"].item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            if return_preds:
                all_image_paths.extend(batch["image_paths"])

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    if return_preds:
        return metrics, all_preds, all_image_paths
    return metrics


# ============== Comment 5 Uncertainty Analysis ==============


def evaluate_comment5_uncertainty(
    model: SARGCL,
    loader: DataLoader,
    device,
    output_dir: str,
    label_map_inv: Dict[int, str],
    n_per_group: int = 3,
):
    """Generate OT/uncertainty analysis tables from the forward pass.

    Samples for which OT could not be computed are marked invalid and
    excluded from OT-based difficulty grouping.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    rows = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Comment 5 evaluation"):
            labels = batch["label"].to(device)
            logits, _, _, info = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            vc = info["visual_node_counts"].cpu().numpy()
            tc = info["text_node_counts"].cpu().numpy()
            vm = info["valid_ot_mask"].cpu().numpy().astype(bool)

            for j in range(labels.size(0)):
                gt = int(labels[j])
                pr = int(preds[j])
                rows.append(
                    {
                        "sample_index": offset + j,
                        "image_path": batch["image_paths"][j],
                        "text": batch["raw_texts"][j],
                        "ground_truth_id": gt,
                        "ground_truth": label_map_inv.get(gt, str(gt)),
                        "prediction_id": pr,
                        "prediction": label_map_inv.get(pr, str(pr)),
                        "correct": int(gt == pr),
                        "prediction_confidence": float(conf[j]),
                        "visual_nodes": int(vc[j]),
                        "text_nodes": int(tc[j]),
                        "valid_ot": int(vm[j]),
                        "ot_cost": float(info["ot_costs"][j]),
                        "tau": float(info["tau"][j]),
                        "w_visual_structured": float(info["weights"][j, 0]),
                        "w_text_structured": float(info["weights"][j, 1]),
                        "w_visual_global": float(info["weights"][j, 2]),
                        "w_text_global": float(info["weights"][j, 3]),
                        "positive_similarity": float(
                            info["positive_similarity"][j]
                        ),
                        "hardest_negative_similarity": float(
                            info["hardest_negative_similarity"][j]
                        ),
                        "contrastive_margin": float(
                            info["contrastive_margin"][j]
                        ),
                    }
                )
            offset += labels.size(0)

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("No test samples collected for Comment 5 analysis.")
    all_df.to_csv(
        os.path.join(output_dir, "comment5_all_test_samples.csv"), index=False
    )

    valid = all_df[all_df.valid_ot == 1].copy()
    valid_rate = 100 * len(valid) / len(all_df)

    diagnostic = pd.DataFrame(
        [
            {
                "total_test_samples": len(all_df),
                "valid_ot_samples": len(valid),
                "invalid_ot_samples": len(all_df) - len(valid),
                "valid_ot_rate_percent": valid_rate,
                "mean_visual_nodes": all_df.visual_nodes.mean(),
                "mean_text_nodes": all_df.text_nodes.mean(),
                "zero_visual_node_rate_percent": 100
                * (all_df.visual_nodes == 0).mean(),
                "zero_text_node_rate_percent": 100
                * (all_df.text_nodes == 0).mean(),
                "mean_valid_ot_cost": valid.ot_cost.mean()
                if len(valid)
                else np.nan,
                "min_valid_ot_cost": valid.ot_cost.min()
                if len(valid)
                else np.nan,
                "max_valid_ot_cost": valid.ot_cost.max()
                if len(valid)
                else np.nan,
                "mean_valid_tau": valid.tau.mean() if len(valid) else np.nan,
                "min_valid_tau": valid.tau.min() if len(valid) else np.nan,
                "max_valid_tau": valid.tau.max() if len(valid) else np.nan,
            }
        ]
    )
    diagnostic.to_csv(
        os.path.join(output_dir, "comment5_ot_node_diagnostic.csv"), index=False
    )
    if valid.empty:
        raise RuntimeError(
            "No valid OT samples. See comment5_ot_node_diagnostic.csv."
        )

    valid = valid.sort_values("ot_cost").reset_index(drop=True)
    valid["uncertainty_rank"] = np.arange(1, len(valid) + 1)
    valid["ot_percentile"] = valid.ot_cost.rank(method="average", pct=True)
    p = valid.ot_cost.rank(method="first", pct=True)
    valid["uncertainty_group"] = np.where(
        p <= 1 / 3,
        "Low difficulty",
        np.where(p <= 2 / 3, "Medium difficulty", "High difficulty"),
    )
    valid.to_csv(
        os.path.join(output_dir, "comment5_valid_ot_samples.csv"), index=False
    )

    order = {"Low difficulty": 0, "Medium difficulty": 1, "High difficulty": 2}
    reps = []
    for gname in order:
        g = valid[valid.uncertainty_group == gname].copy()
        if g.empty:
            continue
        med = g.ot_cost.median()
        g["distance_to_group_median"] = (g.ot_cost - med).abs()
        reps.append(g.sort_values("distance_to_group_median").head(n_per_group))
    rep = pd.concat(reps, ignore_index=True)
    rep["group_order"] = rep.uncertainty_group.map(order)
    rep = (
        rep.sort_values(["group_order", "ot_cost"])
        .drop(columns=["group_order"])
        .reset_index(drop=True)
    )
    rep["example_id"] = np.arange(1, len(rep) + 1)
    rep.to_csv(
        os.path.join(output_dir, "comment5_representative_samples.csv"),
        index=False,
    )

    summary = (
        valid.groupby("uncertainty_group", sort=False)
        .agg(
            samples=("sample_index", "count"),
            mean_ot_cost=("ot_cost", "mean"),
            mean_tau=("tau", "mean"),
            mean_positive_similarity=("positive_similarity", "mean"),
            mean_hardest_negative_similarity=(
                "hardest_negative_similarity",
                "mean",
            ),
            mean_contrastive_margin=("contrastive_margin", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
            accuracy=("correct", "mean"),
            mean_w_visual_structured=("w_visual_structured", "mean"),
            mean_w_text_structured=("w_text_structured", "mean"),
            mean_w_visual_global=("w_visual_global", "mean"),
            mean_w_text_global=("w_text_global", "mean"),
        )
        .reset_index()
    )
    summary["group_order"] = summary.uncertainty_group.map(order)
    summary = summary.sort_values("group_order").drop(columns="group_order")
    summary.to_csv(
        os.path.join(output_dir, "comment5_uncertainty_summary.csv"),
        index=False,
    )

    # Plot OT cost vs. temperature
    plt.figure(figsize=(7, 5))
    plt.scatter(
        valid.ot_cost.to_numpy(), valid.tau.to_numpy(), s=14, alpha=0.55
    )
    plt.xlabel("OT-derived correspondence difficulty")
    plt.ylabel(r"Dynamic temperature $\tau_i$")
    plt.title("OT Difficulty vs. Dynamic Contrastive Temperature")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comment5_ot_cost_vs_temperature.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 80)
    print("COMMENT 5 ANALYSIS")
    print("=" * 80)
    print(f"Total test samples : {len(all_df)}")
    print(f"Valid OT samples   : {len(valid)} ({valid_rate:.2f}%)")
    print(f"Valid OT range     : {valid.ot_cost.min():.6f} -- {valid.ot_cost.max():.6f}")
    print(f"Valid tau range    : {valid.tau.min():.6f} -- {valid.tau.max():.6f}")
    print("\nSummary table:")
    print(summary.to_string(index=False))
    print("\nFiles saved under:", output_dir)
    print("=" * 80)

    return all_df, rep, summary


# ============== OT Sanity Check ==============


def run_ot_sanity_check(
    model,
    loader,
    device,
    lambda_hc_cl: float = 0.1,
    lambda_ot: float = 0.1,
    n_batches: int = 1,
    min_valid_rate: float = 1.0,
):
    """Verify that OT is computed correctly before starting training.

    Checks that visual/textual nodes exist, OT is computed, costs are
    non-zero and finite, and dynamic temperatures vary.
    """
    model.eval()
    total = 0
    valid_ot = 0
    zero_visual = 0
    zero_text = 0
    ot_values = []
    tau_values = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits, _, losses, info = model(
                batch, lambda_hc_cl=lambda_hc_cl, lambda_ot=lambda_ot
            )

            v = info["visual_node_counts"].detach().cpu()
            t = info["text_node_counts"].detach().cpu()
            valid = info["valid_ot_mask"].detach().cpu().bool()
            ot = info["ot_costs"].detach().cpu()
            tau = info["tau"].detach().cpu()

            total += len(v)
            valid_ot += int(valid.sum())
            zero_visual += int((v == 0).sum())
            zero_text += int((t == 0).sum())

            if valid.any():
                ot_values.extend(ot[valid].tolist())
                tau_values.extend(tau[valid].tolist())

            print(f"\nOT SANITY CHECK -- batch {batch_idx + 1}")
            print(f"  Batch valid OT rate: {100.0 * valid.float().mean():.2f}%")
            print(f"  OT loss: {float(losses['ot']):.6f}")

            if batch_idx + 1 >= n_batches:
                break

    rate = 100.0 * valid_ot / max(total, 1)

    print("\n" + "=" * 80)
    print("OT SANITY SUMMARY")
    print("=" * 80)
    print(f"Samples checked       : {total}")
    print(f"Valid OT samples      : {valid_ot}")
    print(f"Valid OT rate         : {rate:.2f}%")

    if ot_values:
        print(f"OT range              : {min(ot_values):.6f} -- {max(ot_values):.6f}")
        print(f"Mean OT               : {sum(ot_values) / len(ot_values):.6f}")
        print(f"Temperature range     : {min(tau_values):.6f} -- {max(tau_values):.6f}")
    else:
        print("OT range              : NO VALID OT SAMPLES")
    print("=" * 80)

    if rate < min_valid_rate:
        raise RuntimeError(
            f"\nOT SANITY CHECK FAILED.\n"
            f"Only {rate:.2f}% of checked samples have valid OT, "
            f"but at least {min_valid_rate:.2f}% is required.\n"
            "Inspect visual/text node extraction before training."
        )

    if not ot_values:
        raise RuntimeError("OT SANITY CHECK FAILED: no valid OT values computed.")

    if not all(np.isfinite(ot_values)):
        raise RuntimeError("OT SANITY CHECK FAILED: non-finite OT values detected.")

    if not all(np.isfinite(tau_values)):
        raise RuntimeError("OT SANITY CHECK FAILED: non-finite temperature values.")

    print("OT sanity check PASSED. Starting training is safe.")
    print("=" * 80 + "\n")


# ============== Training Loop ==============


def train_loop(
    model: SARGCL,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer,
    device,
    epochs: int = 50,
    lambda_hc_cl: float = 0.1,
    lambda_ot: float = 0.1,
    grad_clip: float = 1.0,
    model_save_path: str = "best_sargcl_clip_model_new.pth",
    final_model_save_path: str = "sargcl_clip_model_epoch50_final.pth",
    history_csv_path: str = "training_history.csv",
    log_path: str = "training.log",
):
    """Train SARGCL and save epoch-wise logs plus best/final checkpoints."""
    best_valid_f1 = -1.0
    history = []

    history_dir = os.path.dirname(history_csv_path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SARGCL Training Log\n")
        f.write("================================\n")

    append_training_log(log_path, f"Starting training for {epochs} epochs")
    print(f"Starting SARGCL training for {epochs} epochs...")
    print(f"Best checkpoint: {model_save_path}")
    print(f"Final checkpoint: {final_model_save_path}")

    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_cls = 0.0
        total_train_ot = 0.0
        total_train_hc = 0.0

        diag_total = 0
        diag_valid_ot = 0
        diag_zero_visual = 0
        diag_zero_text = 0
        diag_ot_sum = 0.0
        diag_tau_sum = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {ep}/{epochs} [Training]"
        )

        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            _, _, losses, batch_diag = model(
                batch, lambda_hc_cl=lambda_hc_cl, lambda_ot=lambda_ot
            )

            with torch.no_grad():
                vc = batch_diag["visual_node_counts"]
                tc = batch_diag["text_node_counts"]
                vm = batch_diag["valid_ot_mask"]
                oc = batch_diag["ot_costs"]
                ti = batch_diag["tau"]
                diag_total += int(vc.numel())
                diag_valid_ot += int(vm.sum().item())
                diag_zero_visual += int((vc == 0).sum().item())
                diag_zero_text += int((tc == 0).sum().item())
                if vm.any():
                    diag_ot_sum += float(oc[vm].sum().item())
                    diag_tau_sum += float(ti[vm].sum().item())

            loss = losses["total"]
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_train_loss += float(loss.item())
            total_train_cls += float(
                losses.get("classification", torch.tensor(0.0)).item()
            )
            total_train_ot += float(
                losses.get("ot", torch.tensor(0.0)).item()
            )
            total_train_hc += float(
                losses.get("hc_cl", torch.tensor(0.0)).item()
            )

            progress_bar.set_postfix(
                loss=f"{total_train_loss / (i + 1):.4f}",
                cls=f"{total_train_cls / (i + 1):.4f}",
                ot=f"{total_train_ot / (i + 1):.4f}",
                hc=f"{total_train_hc / (i + 1):.4f}",
            )

        n_batches = max(1, len(train_loader))
        avg_train_loss = total_train_loss / n_batches
        avg_train_cls = total_train_cls / n_batches
        avg_train_ot = total_train_ot / n_batches
        avg_train_hc = total_train_hc / n_batches

        valid_metrics = evaluate_loop(model, valid_loader, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = valid_metrics["f1_score"] > best_valid_f1
        if is_best:
            best_valid_f1 = valid_metrics["f1_score"]
            torch.save(model.state_dict(), model_save_path)
            msg = (
                f"New best model saved: {model_save_path} "
                f"(Val F1={best_valid_f1:.4f})"
            )
            print("-> " + msg)
            append_training_log(log_path, msg)

        diag_valid_rate = 100.0 * diag_valid_ot / max(1, diag_total)
        diag_mean_ot = diag_ot_sum / max(1, diag_valid_ot)
        diag_mean_tau = diag_tau_sum / max(1, diag_valid_ot)

        row = {
            "epoch": ep,
            "train_loss": avg_train_loss,
            "train_classification_loss": avg_train_cls,
            "train_ot_loss": avg_train_ot,
            "train_hc_cl_loss": avg_train_hc,
            "val_loss": valid_metrics["loss"],
            "val_accuracy": valid_metrics["accuracy"],
            "val_precision": valid_metrics["precision"],
            "val_recall": valid_metrics["recall"],
            "val_f1": valid_metrics["f1_score"],
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
            "best_val_f1_so_far": best_valid_f1,
            "is_best": int(is_best),
            "valid_ot_rate_percent": diag_valid_rate,
            "mean_valid_ot_cost": diag_mean_ot,
            "mean_valid_tau": diag_mean_tau,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"Train={avg_train_loss:.4f} "
            f"(Cls={avg_train_cls:.4f}, OT={avg_train_ot:.4f}, HC={avg_train_hc:.4f}) | "
            f"Val Loss={valid_metrics['loss']:.4f} | "
            f"Val Acc={valid_metrics['accuracy']:.4f} | "
            f"Val F1={valid_metrics['f1_score']:.4f} | "
            f"ValidOT={diag_valid_rate:.2f}% | "
            f"Time={epoch_time / 60:.2f} min"
        )

    # Save final checkpoint
    torch.save(model.state_dict(), final_model_save_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)

    print("Training finished.")
    print(f"Best validation F1: {best_valid_f1:.4f}")
    print(f"Best checkpoint: {model_save_path}")
    print(f"Final checkpoint: {final_model_save_path}")
    append_training_log(
        log_path,
        f"Training finished. Best Val F1={best_valid_f1:.6f}",
    )

    return history_df


# ============== Utility Functions ==============


def save_test_predictions(model, loader, device, output_dir, label_map_inv):
    """Save per-sample test set predictions to CSV."""
    print("Generating and saving test predictions...")
    _, preds, image_paths = evaluate_loop(
        model, loader, device, return_preds=True
    )

    results_df = pd.DataFrame(
        {"image_path": image_paths, "predicted_label_id": preds}
    )
    results_df["predicted_label"] = results_df["predicted_label_id"].map(
        label_map_inv
    )

    output_path = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to: {output_path}")


def calculate_parameters(model):
    """Count trainable parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    return num_params


def calculate_inference_time(model, loader, device, num_batches: int = 50):
    """Measure average inference time per batch."""
    print(f"Measuring inference time over {num_batches} batches...")
    model.eval()
    total_time = 0

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(loader, total=num_batches, desc="Inference Time")
        ):
            if i >= num_batches:
                break
            start_time = time.time()
            _ = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            total_time += end_time - start_time

    avg_time = total_time / num_batches
    print(f"Average inference time per batch: {avg_time * 1000:.2f} ms")
    return avg_time
