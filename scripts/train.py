#!/usr/bin/env python
"""SARGCL Training Script.

Train the State-Aware Hypergraph Representation Learning model with
Difficulty-Regulated Contrastive Alignment for multimodal classification.

Example usage:
    python scripts/train.py \
        --train_csv data/task_informative_text_img_agreed_lab_train.tsv \
        --valid_csv data/task_informative_text_img_agreed_lab_dev.tsv \
        --test_csv  data/task_informative_text_img_agreed_lab_test.tsv \
        --image_dir /path/to/data_image \
        --output_dir outputs/sargcl_run1 \
        --epochs 50 \
        --batch_size 8
"""

import os
import sys
import argparse

import pandas as pd
import torch
import clip
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargcl.utils import set_seed
from sargcl.dataset import DisasterDataset, collate_fn
from sargcl.model import SARGCL
from sargcl.evaluate import (
    evaluate_loop,
    evaluate_comment5_uncertainty,
    run_ot_sanity_check,
    train_loop,
    save_test_predictions,
    calculate_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SARGCL for multimodal classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to the training TSV file.",
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        required=True,
        help="Path to the validation TSV file.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to the test TSV file.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory containing tweet images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for checkpoints, logs, and results.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_hc_cl", type=float, default=0.1)
    parser.add_argument("--lambda_ot", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model config
    parser.add_argument(
        "--clip_model", type=str, default="ViT-B/32",
        help="CLIP architecture variant.",
    )
    parser.add_argument(
        "--max_visual_nodes", type=int, default=12,
        help="Maximum visual entity nodes per image.",
    )
    parser.add_argument(
        "--visual_conf", type=float, default=0.5,
        help="Minimum detection confidence for visual nodes.",
    )

    # Workflow control
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and only evaluate the best checkpoint.",
    )
    parser.add_argument(
        "--ot_sanity_check", action="store_true", default=True,
        help="Run OT sanity check before training.",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU device index to use.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # GPU setup
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Label mapping
    LABEL_MAP = {"not_informative": 0, "informative": 1}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
    NUM_CLASSES = len(LABEL_MAP)

    # Load CLIP preprocessing
    _, clip_preprocess = clip.load(args.clip_model, device=device)

    # Load datasets
    print("Loading data...")
    df_train = pd.read_csv(args.train_csv, sep="\t")
    df_valid = pd.read_csv(args.valid_csv, sep="\t")
    df_test = pd.read_csv(args.test_csv, sep="\t")

    for df in (df_train, df_valid, df_test):
        df["label"] = df["label"].map(LABEL_MAP)
        df.dropna(subset=["label"], inplace=True)

    train_dataset = DisasterDataset(
        df_train, clip_preprocess=clip_preprocess, image_base_path=args.image_dir
    )
    valid_dataset = DisasterDataset(
        df_valid, clip_preprocess=clip_preprocess, image_base_path=args.image_dir
    )
    test_dataset = DisasterDataset(
        df_test, clip_preprocess=clip_preprocess, image_base_path=args.image_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = SARGCL(
        device=device,
        num_classes=NUM_CLASSES,
        clip_model_name=args.clip_model,
        max_visual_nodes=args.max_visual_nodes,
        visual_conf=args.visual_conf,
    ).to(device)

    calculate_parameters(model)

    # Checkpoint paths
    best_path = os.path.join(args.output_dir, "best_sargcl_model.pth")
    final_path = os.path.join(args.output_dir, "sargcl_model_final.pth")
    history_path = os.path.join(args.output_dir, "training_history.csv")
    log_path = os.path.join(args.output_dir, "training.log")

    # Training
    if not args.skip_training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        if args.ot_sanity_check:
            run_ot_sanity_check(
                model,
                train_loader,
                device,
                lambda_hc_cl=args.lambda_hc_cl,
                lambda_ot=args.lambda_ot,
            )

        train_loop(
            model,
            train_loader,
            valid_loader,
            optimizer,
            device,
            epochs=args.epochs,
            lambda_hc_cl=args.lambda_hc_cl,
            lambda_ot=args.lambda_ot,
            grad_clip=args.grad_clip,
            model_save_path=best_path,
            final_model_save_path=final_path,
            history_csv_path=history_path,
            log_path=log_path,
        )

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATING BEST CHECKPOINT")
    print("=" * 80)

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

    best_model = SARGCL(
        device=device,
        num_classes=NUM_CLASSES,
        clip_model_name=args.clip_model,
    ).to(device)
    checkpoint = torch.load(best_path, map_location=device)
    best_model.load_state_dict(checkpoint)
    best_model.eval()

    print(f"Loaded best checkpoint: {best_path}")

    # Test evaluation
    test_metrics = evaluate_loop(best_model, test_loader, device)
    metrics_df = pd.DataFrame(
        [
            {
                "checkpoint": "best_validation_f1",
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1_score"],
            }
        ]
    )
    metrics_csv = os.path.join(args.output_dir, "final_test_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1:       {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:   {test_metrics['recall']:.4f}")

    # Save predictions
    save_test_predictions(best_model, test_loader, device, args.output_dir, LABEL_MAP_INV)

    # Uncertainty analysis
    evaluate_comment5_uncertainty(
        model=best_model,
        loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        label_map_inv=LABEL_MAP_INV,
        n_per_group=3,
    )

    print("\n" + "=" * 80)
    print("ALL RESULTS SAVED")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
