#!/usr/bin/env python
"""Standalone SARGCL Evaluation Script.

Evaluate a trained SARGCL checkpoint on a test set and generate
predictions, metrics, and uncertainty analysis.

Example usage:
    python scripts/evaluate.py \
        --test_csv data/task_informative_text_img_agreed_lab_test.tsv \
        --image_dir /path/to/data_image \
        --checkpoint outputs/best_sargcl_model.pth \
        --output_dir outputs/eval_results
"""

import os
import sys
import argparse

import pandas as pd
import torch
import clip
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sargcl.utils import set_seed
from sargcl.dataset import DisasterDataset, collate_fn
from sargcl.model import SARGCL
from sargcl.evaluate import (
    evaluate_loop,
    evaluate_comment5_uncertainty,
    save_test_predictions,
    calculate_parameters,
    calculate_inference_time,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SARGCL checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--run_uncertainty_analysis",
        action="store_true",
        default=True,
        help="Run OT-based uncertainty analysis.",
    )
    parser.add_argument(
        "--measure_inference_time",
        action="store_true",
        help="Measure average inference time per batch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    LABEL_MAP = {"not_informative": 0, "informative": 1}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
    NUM_CLASSES = len(LABEL_MAP)

    _, clip_preprocess = clip.load(args.clip_model, device=device)

    # Load test data
    df_test = pd.read_csv(args.test_csv, sep="\t")
    df_test["label"] = df_test["label"].map(LABEL_MAP)
    df_test.dropna(subset=["label"], inplace=True)

    test_dataset = DisasterDataset(
        df_test, clip_preprocess=clip_preprocess, image_base_path=args.image_dir
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model = SARGCL(
        device=device,
        num_classes=NUM_CLASSES,
        clip_model_name=args.clip_model,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    calculate_parameters(model)

    # Evaluate
    test_metrics = evaluate_loop(model, test_loader, device)
    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1:        {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")

    # Save predictions
    save_test_predictions(model, test_loader, device, args.output_dir, LABEL_MAP_INV)

    # Uncertainty analysis
    if args.run_uncertainty_analysis:
        evaluate_comment5_uncertainty(
            model=model,
            loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            label_map_inv=LABEL_MAP_INV,
        )

    # Inference time
    if args.measure_inference_time:
        calculate_inference_time(model, test_loader, device)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
