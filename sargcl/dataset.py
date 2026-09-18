"""Dataset and data-loading utilities for SARGCL."""

import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import clip


class DisasterDataset(Dataset):
    """Multimodal disaster-tweet dataset.

    Each sample consists of a tweet image, tweet text, and a binary label
    (informative / not_informative).

    Args:
        df: DataFrame with columns ``image``, ``tweet_text``, ``label``.
        clip_preprocess: CLIP image preprocessing transform.
        image_base_path: Root directory containing the tweet images.
        max_len: Maximum CLIP token length (default 77).
    """

    def __init__(self, df, clip_preprocess, image_base_path: str, max_len: int = 77):
        self.df = df.reset_index(drop=True)
        self.clip_preprocess = clip_preprocess
        self.max_len = max_len
        self.image_base_path = image_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        relative_img_path = row["image"]
        if relative_img_path.startswith("/"):
            relative_img_path = relative_img_path[1:]
        img_path = os.path.join(self.image_base_path, relative_img_path)

        text = row["tweet_text"]
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError):
            # Create a black image if the original is not found
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Process image and text for CLIP
        image_input = self.clip_preprocess(image)
        text_input = clip.tokenize([text], truncate=True).squeeze(0)

        item = {
            "pixel_values": image_input,
            "input_ids": text_input,
            "raw_image": image,
            "raw_text": text,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": img_path,
        }
        return item


def collate_fn(batch):
    """Custom collate function that stacks tensors and keeps raw data as lists."""
    keys = ["pixel_values", "input_ids", "label"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["raw_images"] = [b["raw_image"] for b in batch]
    out["raw_texts"] = [b["raw_text"] for b in batch]
    out["image_paths"] = [b["image_path"] for b in batch]
    return out
