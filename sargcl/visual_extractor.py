"""State-Aware Visual Node Extraction using Faster R-CNN + BLIP captioning."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image


class VisualNodeExtractor(nn.Module):
    """Extract state-aware visual entity nodes from images.

    Uses Faster R-CNN for object detection and BLIP for region-level
    captioning, producing textual descriptions that encode the *state*
    of each detected entity (e.g., "a damaged building" vs. "a building").

    Args:
        device: Torch device.
        blip_model: HuggingFace model identifier for BLIP captioner.
        max_nodes: Maximum number of entity nodes per image.
        min_conf: Minimum detection confidence threshold.
    """

    def __init__(
        self,
        device,
        blip_model: str = "Salesforce/blip-image-captioning-base",
        max_nodes: int = 12,
        min_conf: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
        self.captioner = BlipForConditionalGeneration.from_pretrained(
            blip_model, use_safetensors=True
        ).to(device).eval()
        self.blip_proc = BlipProcessor.from_pretrained(blip_model)
        self.max_nodes = max_nodes
        self.min_conf = min_conf

    @torch.no_grad()
    def forward(
        self, pil_images: List[Image.Image]
    ) -> Tuple[List[List[Tuple[int, int, int, int]]], List[List[str]]]:
        """Detect objects and caption each region.

        Returns:
            results_boxes: Per-image list of bounding boxes ``(x1, y1, x2, y2)``.
            captions: Per-image list of BLIP-generated captions for each box.
        """
        self.detector.eval()
        results_boxes = []
        captions = []

        for img in pil_images:
            img_t = transforms.functional.to_tensor(img).to(self.device)
            pred = self.detector([img_t])[0]
            boxes, scores = pred["boxes"], pred["scores"]
            keep = scores >= self.min_conf
            boxes, scores = boxes[keep], scores[keep]

            if boxes.size(0) > self.max_nodes:
                topk = torch.topk(scores, k=self.max_nodes).indices
                boxes = boxes[topk]

            img_captions, img_boxes = [], []
            for b in boxes:
                x1, y1, x2, y2 = b.tolist()
                crop = img.crop((x1, y1, x2, y2))
                inputs = self.blip_proc(
                    images=crop, text=None, return_tensors="pt"
                ).to(self.device)
                out = self.captioner.generate(**inputs, max_new_tokens=30)
                caption = self.blip_proc.decode(out[0], skip_special_tokens=True)
                img_captions.append(caption)
                img_boxes.append((int(x1), int(y1), int(x2), int(y2)))

            results_boxes.append(img_boxes)
            captions.append(img_captions)

        return results_boxes, captions
