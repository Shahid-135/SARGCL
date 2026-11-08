#!/usr/bin/env python
# coding: utf-8

# # SARG-CL: Notebook Version
# 
# Cleaned and organized version of the supplied SARG-CL training and evaluation code. The original command-line argument section has been converted into notebook configuration variables so the notebook can be run interactively.

# In[1]:


# ============================================================
# GPU Configuration
# ============================================================

import os

# Use physical GPU 2
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # v5: command-line value takes precedence

import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# In[2]:


import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_convert
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

# Transformers and CLIP
from transformers import BlipForConditionalGeneration, BlipProcessor
import clip

# spaCy for dependency parsing
import spacy

# --- NEW: Added for metrics calculation ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from thop import profile # For FLOPs calculation
from copy import deepcopy

from torchvision import transforms


# In[3]:


# ============== Utilities ==============

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def exists(x):
    return x is not None

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.matmul(a, b.transpose(-1, -2))


# In[4]:


# ============== Dataset ==============

class DisasterDataset(Dataset):
    def __init__(self, df, clip_preprocess, max_len=77): # CLIP's context length is 77
        self.df = df.reset_index(drop=True)
        self.clip_preprocess = clip_preprocess
        self.max_len = max_len
        # Make sure this path is correct for your environment
        self.image_base_path = "/home/nagendra/cafunet/data_image"

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_image_path = str(row["image"]).strip()

        # ------------------------------------------------------------
        # Robust image-path resolution.
        #
        # Supported:
        #   data_image/foo/bar.jpg
        #   /data_image/foo/bar.jpg
        #   foo/bar.jpg
        #   /home/nagendra/cafunet/data_image/foo/bar.jpg
        # ------------------------------------------------------------
        if os.path.isabs(raw_image_path):
            img_path = os.path.normpath(raw_image_path)
        else:
            clean_path = raw_image_path.replace("\\", "/").lstrip("/")

            # image_base_path already ends in "data_image".
            # Therefore remove the duplicated prefix from TSV entries.
            if clean_path.startswith("data_image/"):
                clean_path = clean_path[len("data_image/"):]

            img_path = os.path.normpath(
                os.path.join(
                    self.image_base_path,
                    clean_path,
                )
            )

        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                "IMAGE FILE NOT FOUND.\n"
                f"TSV image value : {raw_image_path}\n"
                f"Resolved path   : {img_path}\n"
                f"Base path       : {self.image_base_path}\n"
            )

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                "IMAGE FILE COULD NOT BE OPENED.\n"
                f"TSV image value : {raw_image_path}\n"
                f"Resolved path   : {img_path}\n"
            ) from exc

        image_tensor = (
            torchvision.transforms.functional
            .pil_to_tensor(image)
            .float()
            .div(255.0)
        )

        if not torch.isfinite(image_tensor).all():
            raise ValueError(
                f"Non-finite image tensor: {img_path}"
            )

        # Preserve the original dataset return structure.
        text = row["tweet_text"]
        label = int(row["label"])

        return {
            "image": image_tensor,
            "text": text,
            "label": label,
            "image_path": img_path,
        }


def collate_fn(batch):
    """
    Convert the actual DisasterDataset output into the tensors/objects
    expected by the original SARGCL model.

    DisasterDataset returns:
        image       : [3,H,W] float tensor in [0,1]
        text        : raw tweet string
        label       : integer
        image_path  : resolved path

    We intentionally do NOT stack the raw images because images can have
    different spatial sizes. CLIP preprocessing is applied individually.
    """

    raw_images = []
    pixel_values = []
    raw_texts = []
    input_ids = []
    labels = []
    image_paths = []

    for sample in batch:
        # Recover PIL image for Faster R-CNN/BLIP.
        image_tensor = sample["image"].clamp(0.0, 1.0)
        pil_image = transforms.ToPILImage()(image_tensor)

        raw_images.append(pil_image)

        # CLIP expects the original CLIP preprocessing pipeline.
        pixel_values.append(
            clip_preprocess(pil_image)
        )

        text = str(sample["text"])
        raw_texts.append(text)

        # OpenAI CLIP tokenization uses context length 77.
        input_ids.append(
            clip.tokenize(
                [text],
                truncate=True,
            )[0]
        )

        labels.append(int(sample["label"]))
        image_paths.append(sample["image_path"])

    return {
        "pixel_values": torch.stack(
            pixel_values,
            dim=0,
        ),
        "input_ids": torch.stack(
            input_ids,
            dim=0,
        ),
        "label": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "raw_images": raw_images,
        "raw_texts": raw_texts,
        "image_paths": image_paths,
    }


# In[5]:


# ============== Foundational Feature Extractors (CLIP) ==============

class CLIPEncoders(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cpu"):
        super().__init__()
        self.clip, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip.float() # Use float32 for stability
        self.text_hidden = self.clip.text_projection.shape[-1]
        self.vit_hidden = self.clip.visual.proj.shape[-1]

    @torch.no_grad()
    def encode_image_global(self, pixel_values: torch.Tensor) -> torch.Tensor:
        img_feats = self.clip.encode_image(pixel_values.to(self.clip.dtype))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats.float()

    @torch.no_grad()
    def encode_text_global(self, input_ids: torch.Tensor) -> torch.Tensor:
        txt_feats = self.clip.encode_text(input_ids)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        return txt_feats.float()
        
    @torch.no_grad()
    def encode_text_for_nodes(self, texts: List[str], device) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.text_hidden, device=device)
        tokens = clip.tokenize(texts, truncate=True).to(device)
        text_feats = self.clip.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats.float()


# In[6]:


# ============== State-Aware Visual Node Extraction ==============

class VisualNodeExtractor(nn.Module):
    def __init__(self, device, blip_model="Salesforce/blip-image-captioning-base", max_nodes=12, min_conf=0.5):
        super().__init__()
        self.device = device
        self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
        self.captioner = BlipForConditionalGeneration.from_pretrained( blip_model, use_safetensors=True).to(device).eval()
        self.blip_proc = BlipProcessor.from_pretrained(blip_model)
        self.max_nodes = max_nodes
        self.min_conf = min_conf

    @torch.no_grad()
    def forward(self, pil_images: List[Image.Image]) -> Tuple[List[torch.Tensor], List[List[Tuple[int,int,int,int]]], List[List[str]]]:
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
                crop = img.crop((x1, y1, x2, y2)).convert("RGB")

                # Faster R-CNN can occasionally produce a very thin box, e.g.
                # an image crop with shape (1, W, 3).  Transformers' BLIP
                # processor cannot reliably infer the channel dimension for
                # such an array.  Enforce a valid 2-D RGB crop before BLIP.
                crop_w, crop_h = crop.size
                if crop_w < 2 or crop_h < 2:
                    crop = crop.resize(
                        (max(2, crop_w), max(2, crop_h)),
                        Image.Resampling.BILINEAR,
                    )

                inputs = self.blip_proc(
                    images=crop,
                    text=None,
                    return_tensors="pt",
                ).to(self.device)
                out = self.captioner.generate(**inputs, max_new_tokens=30)
                caption = self.blip_proc.decode(out[0], skip_special_tokens=True)
                img_captions.append(caption)
                img_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            results_boxes.append(img_boxes)
            captions.append(img_captions)
        return results_boxes, captions


# In[7]:


# ============== spaCy-based Textual Node Extraction ==============

class TextNodeExtractor:
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(f"spaCy model '{model}' not found. Please run: python -m spacy download {model}")

    def extract_nodes_and_hyperedges(self, texts: List[str]) -> Tuple[List[List[str]], List[List[List[int]]]]:
        all_nodes, all_hyperedges = [], []
        for s in texts:
            doc = self.nlp(s)
            nodes, node_idx = [], {}
            for tok in doc:
                if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"} and not tok.is_stop and tok.text not in node_idx:
                    node_idx[tok.text] = len(nodes)
                    nodes.append(tok.text)

            hedges = []
            for tok in doc:
                if tok.pos_ == "VERB":
                    group = set()
                    if tok.text in node_idx:
                        group.add(node_idx[tok.text])
                    for child in tok.children:
                        if child.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"} and child.text in node_idx:
                            group.add(node_idx[child.text])
                    if len(group) >= 2:
                        hedges.append(sorted(list(group)))
            
            if not hedges and len(nodes) >= 2:
                hedges = [list(range(len(nodes)))]

            all_nodes.append(nodes)
            all_hyperedges.append(hedges)
        return all_nodes, all_hyperedges


# In[8]:


# ============== Dynamic Cross-Modal Visual Hypergraph Refinement ==============

class VisualHypergraphRefiner(nn.Module):
    def __init__(self, clip_hidden: int, top_k_edges: int = 8):
        super().__init__()
        self.top_k_edges = top_k_edges
        self.scorer = nn.Sequential(
            nn.Linear(clip_hidden * 2, clip_hidden), nn.ReLU(),
            nn.Linear(clip_hidden, 1), nn.Sigmoid()
        )

    def forward(self, visual_node_feats: List[torch.Tensor], gT: torch.Tensor) -> List[Tuple[torch.Tensor, List[List[int]]]]:
        B = len(visual_node_feats)
        refined = []
        for i in range(B):
            V = visual_node_feats[i]
            n = V.size(0)
            if n <= 1:
                refined.append((V, [list(range(n))], False))
                continue

            with torch.no_grad():
                sims = cosine_sim(V, V)
                knn = 3 if n > 3 else max(1, n-1)
                neighbors = torch.topk(sims, k=knn+1, dim=1).indices[:, 1:]

            candidates = set()
            for u in range(n):
                for v in neighbors[u]:
                    candidates.add(tuple(sorted((u, int(v)))))
            
            ees = []
            for tup in candidates:
                idxs = list(tup)
                he = V[idxs].mean(dim=0)
                score = self.scorer(torch.cat([he, gT[i]], dim=-1))
                ees.append((score.item(), idxs))

            ees.sort(key=lambda x: x[0], reverse=True)
            fallback_used = False
            top = [idxs for _, idxs in ees[:self.top_k_edges]]
            if not top:
                top = [list(range(n))]
                fallback_used = True
            refined.append((V, top, fallback_used))
        return refined


# In[9]:


# ============== Hypergraph Attention Network (HGAT) ==============

class HGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, heads=2):
        super().__init__()
        self.in_dim, self.out_dim, self.heads = in_dim, out_dim, heads
        self.lin_node = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_edge = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, out_dim))
        self.dropout, self.act = nn.Dropout(dropout), nn.GELU()

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        if X.size(0) != H.size(0):
            raise RuntimeError(f"Node feature matrix X and incidence matrix H have inconsistent number of nodes: {X.size(0)} vs {H.size(0)}")
        
        N, E = H.shape
        deg_e = H.sum(dim=0).clamp(min=1.0)

        E_feat = torch.matmul(H.t(), X) / deg_e.unsqueeze(-1)
        X_h = self.lin_node(X).view(N, self.heads, self.out_dim)
        E_h = self.lin_edge(E_feat).view(E, self.heads, self.out_dim)
        
        HX = torch.einsum("en,nhd->ehd", H.t(), X_h) / deg_e.view(E, 1, 1)

        M = HX + E_h
        alpha = torch.softmax(torch.einsum("ehd,hd->eh", M, self.attn), dim=0)
        alpha = self.dropout(alpha)

        out = torch.einsum("ne,eh,ehd->nhd", H, alpha, M).reshape(N, self.heads * self.out_dim)
        return self.act(out)

class HGATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=256, out_dim=256, layers=2):
        super().__init__()
        mods = []
        heads = 2
        
        if layers == 1:
            if out_dim % heads != 0: raise ValueError(f"out_dim ({out_dim}) must be divisible by heads ({heads})")
            mods.append(HGATLayer(in_dim, out_dim // heads, heads=heads))
        else:
            mods.append(HGATLayer(in_dim, hid_dim, heads=heads))
            for _ in range(layers - 2):
                mods.append(HGATLayer(hid_dim * heads, hid_dim, heads=heads))
            if out_dim % heads != 0: raise ValueError(f"out_dim ({out_dim}) must be divisible by heads ({heads})")
            mods.append(HGATLayer(hid_dim * heads, out_dim // heads, heads=heads))
        self.layers = nn.ModuleList(mods)

    def forward(self, X, H):
        for layer in self.layers:
            X = layer(X, H)
        return X


# In[10]:


# ============== Sinkhorn OT Loss ==============

def sinkhorn_knopp(C: torch.Tensor, epsilon: float = 0.07, n_iters: int = 50) -> torch.Tensor:
    K = torch.exp(-C / epsilon)
    n, m = K.shape
    r, c = torch.ones(n, device=K.device)/n, torch.ones(m, device=K.device)/m
    u, v = torch.ones(n, device=K.device)/n, torch.ones(m, device=K.device)/m
    for _ in range(n_iters):
        u = r / (K @ v + 1e-8)
        v = c / (K.t() @ u + 1e-8)
    return torch.diag(u) @ K @ torch.diag(v)

def ot_loss(V_nodes: torch.Tensor, T_nodes: torch.Tensor, epsilon: float = 0.07, n_iters: int = 50) -> torch.Tensor:
    S = cosine_sim(V_nodes, T_nodes)
    C = 1.0 - S.clamp(-1, 1)
    P = sinkhorn_knopp(C, epsilon=epsilon, n_iters=n_iters)
    return (P * C).sum()


# In[11]:


# ============== SARG-CL Full Model ==============

class SARGCL(nn.Module):
    def __init__(self, device, num_classes: int, clip_model_name="ViT-B/32", blip_model="Salesforce/blip-image-captioning-base", max_visual_nodes=12, visual_conf=0.10, top_k_visual_edges=8, hgat_layers=2, tau_min=0.05, tau_max=0.1, tau_base=0.07):
        super().__init__()
        self.device = device
        self.enc = CLIPEncoders(clip_model_name=clip_model_name, device=device)
        self.clip_dim = self.enc.text_hidden # 512 for ViT-B/32
        
        self.visual_nodes = VisualNodeExtractor(device=device, blip_model=blip_model, max_nodes=max_visual_nodes, min_conf=visual_conf)
        self.text_nodes = TextNodeExtractor()
        
        self.visual_refiner = VisualHypergraphRefiner(clip_hidden=self.clip_dim, top_k_edges=top_k_visual_edges)
        
        self.hgat_visual = HGATEncoder(in_dim=self.clip_dim, hid_dim=self.clip_dim, out_dim=self.clip_dim, layers=hgat_layers)
        self.hgat_text = HGATEncoder(in_dim=self.clip_dim, hid_dim=self.clip_dim, out_dim=self.clip_dim, layers=hgat_layers)
        
        self.uncert_mlp = nn.Sequential(nn.Linear(self.clip_dim * 2, self.clip_dim), nn.ReLU(), nn.Linear(self.clip_dim, 4), nn.Softplus())
        
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim * 3, self.clip_dim), nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(self.clip_dim, num_classes)
        )
        
        # --- NEW: Parameters for Hierarchical Consistency Contrastive Loss ---
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_base = tau_base
        self.temp_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def build_incidence(self, num_nodes: int, hyperedges: List[List[int]]) -> torch.Tensor:
        if num_nodes == 0: return torch.zeros(0, 0, device=self.device)
        E = len(hyperedges)
        H = torch.zeros(num_nodes, max(E,1), device=self.device)
        if E == 0:
            H[:, 0] = 1.0; return H
        for e_idx, nodes in enumerate(hyperedges):
            for n in nodes:
                if n < num_nodes: H[n, e_idx] = 1.0
        return H

    # --- NEW: Hierarchical Consistency Contrastive Loss (LHC-CL) ---
    def compute_dynamic_temperature(self, ot_costs: torch.Tensor) -> torch.Tensor:
        """Compute the sample-specific temperature used by LHC-CL."""
        with torch.no_grad():
            ot_costs_unsqueezed = ot_costs.unsqueeze(-1)
        temp_factor = self.temp_mlp(ot_costs_unsqueezed)
        return (
            self.tau_min
            + (self.tau_max - self.tau_min)
            * temp_factor.squeeze(-1)
        )

    def lhc_cl_loss(self, zV: torch.Tensor, zT: torch.Tensor, ot_costs: torch.Tensor) -> torch.Tensor:
        B = zV.size(0)
        if B == 0:
            return torch.tensor(0.0, device=self.device)
        # Calculate dynamic temperature tau_i for each positive pair.
        tau_i = self.compute_dynamic_temperature(ot_costs)

        # Normalize embeddings
        zV_norm = F.normalize(zV, dim=-1)
        zT_norm = F.normalize(zT, dim=-1)

        # Similarities
        sim_matrix = zV_norm @ zT_norm.t() # Shape: (B, B)

        # Positive pair similarities (diagonal)
        sim_positive = torch.diag(sim_matrix) / tau_i

        # Negative pair similarities (off-diagonal)
        # Use tau_base for all negative pairs
        sim_negatives = sim_matrix / self.tau_base
        
        # Create a mask to zero out the diagonal for negative sums
        mask = torch.eye(B, device=self.device, dtype=torch.bool)
        sim_negatives.masked_fill_(mask, -1e9) # Fill diagonal with large negative number
        
        # Denominator: sum of exp(negative similarities)
        log_sum_exp_neg = torch.logsumexp(sim_negatives, dim=1)
        
        # LHC-CL loss for each item in the batch
        # log [ exp(pos/tau_i) / (exp(pos/tau_i) + sum_j!=i exp(neg_ij/tau_base)) ] is unstable
        # A more stable form is: - (pos/tau_i) + log(exp(pos/tau_i) + sum_j!=i exp(neg_ij/tau_base))
        # But based on the formula: L = -log(exp(pos/tau_i) / (exp(pos/tau_i) + sum_j!=i exp(neg/tau_base)))
        # This simplifies to - (sim_positive - log(exp(sim_positive) + exp(log_sum_exp_neg)))
        # The formula given is actually a simplification of InfoNCE where denominator for positive pair is different
        
        # Denominator: exp(sim_positive) + sum_j!=i exp(sim_neg_ij)
        denominator = torch.exp(sim_positive) + torch.sum(torch.exp(sim_negatives), dim=1)

        # Numerator
        numerator = torch.exp(sim_positive)

        loss = -torch.log(numerator / (denominator + 1e-8))

        return loss.mean()

    def forward(self, batch: Dict[str, Any], lambda_hc_cl=0.1, lambda_ot=0.1):
        pixel_values, input_ids = batch["pixel_values"].to(self.device), batch["input_ids"].to(self.device)
        B = pixel_values.size(0)

        gV_clip = self.enc.encode_image_global(pixel_values)
        gT_clip = self.enc.encode_text_global(input_ids)
        
        _, captions_list = self.visual_nodes(batch["raw_images"])
        vis_node_feats = [self.enc.encode_text_for_nodes(caps, self.device) for caps in captions_list]

        text_nodes_list, text_hyperedges = self.text_nodes.extract_nodes_and_hyperedges(batch["raw_texts"])
        txt_node_feats = [self.enc.encode_text_for_nodes(nodes, self.device) for nodes in text_nodes_list]

        refined_visual = self.visual_refiner(vis_node_feats, gT_clip)
        zV_nodes, zT_nodes, zV_summ, zT_summ = [], [], [], []
        
        for i in range(B):
            V_i, edges_i, fallback_i = refined_visual[i]
            zVi_nodes = torch.zeros(0, self.clip_dim, device=self.device) if V_i.size(0) == 0 else self.hgat_visual(V_i, self.build_incidence(V_i.size(0), edges_i))
            zV_nodes.append(zVi_nodes)
            zV_summ.append(zVi_nodes.mean(dim=0) if zVi_nodes.size(0) > 0 else torch.zeros(self.clip_dim, device=self.device))
            
            T_i, HTi = txt_node_feats[i], self.build_incidence(txt_node_feats[i].size(0), text_hyperedges[i])
            zTi_nodes = torch.zeros(0, self.clip_dim, device=self.device) if T_i.size(0) == 0 else self.hgat_text(T_i, HTi)
            zT_nodes.append(zTi_nodes)
            zT_summ.append(zTi_nodes.mean(dim=0) if zTi_nodes.size(0) > 0 else torch.zeros(self.clip_dim, device=self.device))

        zV, zT = torch.stack(zV_summ), torch.stack(zT_summ)
        
        zcontext = torch.cat([zV, zT], dim=-1)
        alphas = self.uncert_mlp(zcontext) + 1e-3
        weights = alphas / alphas.sum(dim=-1, keepdim=True)
        
        zhybrid = weights[:,0:1]*zV + weights[:,1:2]*zT + weights[:,2:3]*gV_clip + weights[:,3:4]*gT_clip
        
        final_representation = torch.cat([zhybrid, gV_clip, gT_clip], dim=-1)
        logits = self.classifier(final_representation)

        losses = {}
        if "label" in batch: losses["classification"] = F.cross_entropy(logits, batch["label"].to(self.device))
        
        # Compute OT only for samples with both visual and textual nodes.
        # IMPORTANT: keep the differentiable OT mean for the training loss.
        # The previous implementation copied OT values into a fresh tensor via
        # in-place indexing and then optimized that tensor, which severed the
        # gradient path from the OT loss to the HGAT/node representations.
        valid_ot_indices = [
            i for i, (v, t) in enumerate(zip(zV_nodes, zT_nodes))
            if v.size(0) > 0 and t.size(0) > 0
        ]
        ot_vals = [
            ot_loss(zV_nodes[i], zT_nodes[i])
            for i in valid_ot_indices
        ]

        # Differentiable scalar used in the objective.
        if ot_vals:
            losses["ot"] = torch.stack(ot_vals).mean()
        else:
            losses["ot"] = torch.zeros((), device=self.device)

        # Detached per-sample OT values are used only for diagnostics and
        # difficulty/temperature computation.
        ot_costs_tensor = torch.zeros(B, device=self.device)
        if ot_vals:
            with torch.no_grad():
                ot_costs_tensor[valid_ot_indices] = torch.stack(ot_vals).detach()
        losses["hc_cl"] = self.lhc_cl_loss(
            zV, zT, ot_costs_tensor.detach()
        )

        # ============================================================
        # Reviewer Comment 5 diagnostics
        # ============================================================
        tau_i = self.compute_dynamic_temperature(
            ot_costs_tensor.detach()
        )

        with torch.no_grad():
            zV_norm = F.normalize(zV, dim=-1)
            zT_norm = F.normalize(zT, dim=-1)
            sim_matrix = zV_norm @ zT_norm.t()

            positive_similarity = torch.diag(sim_matrix)

            batch_sim = sim_matrix.size(0)
            if batch_sim > 1:
                negative_matrix = sim_matrix.clone()
                negative_mask = torch.eye(
                    batch_sim,
                    device=sim_matrix.device,
                    dtype=torch.bool
                )
                negative_matrix.masked_fill_(negative_mask, -1e9)
                hardest_negative_similarity = negative_matrix.max(dim=1).values
            else:
                hardest_negative_similarity = torch.zeros(
                    batch_sim,
                    device=sim_matrix.device
                )

            contrastive_margin = (
                positive_similarity
                - hardest_negative_similarity
            )

        losses["total"] = (
            losses.get("classification", 0.0)
            + lambda_hc_cl * losses["hc_cl"]
            + lambda_ot * losses["ot"]
        )

        visual_node_counts = torch.tensor(
            [x.size(0) for x in zV_nodes], dtype=torch.long, device=self.device
        )
        text_node_counts = torch.tensor(
            [x.size(0) for x in zT_nodes], dtype=torch.long, device=self.device
        )
        valid_ot_mask = (visual_node_counts > 0) & (text_node_counts > 0)

        fallback_flags = torch.tensor(
            [bool(x[2]) for x in refined_visual],
            dtype=torch.bool,
            device=self.device,
        )
        visual_edge_counts = torch.tensor(
            [len(x[1]) for x in refined_visual],
            dtype=torch.long,
            device=self.device,
        )

        diagnostics = {
            "weights": weights.detach(),
            "alphas": alphas.detach(),
            "ot_costs": ot_costs_tensor.detach(),
            "tau": tau_i.detach(),
            "positive_similarity": positive_similarity.detach(),
            "hardest_negative_similarity": hardest_negative_similarity.detach(),
            "contrastive_margin": contrastive_margin.detach(),
            "visual_node_counts": visual_node_counts.detach(),
            "text_node_counts": text_node_counts.detach(),
            "valid_ot_mask": valid_ot_mask.detach(),
            "fallback_used": fallback_flags.detach(),
            "visual_edge_counts": visual_edge_counts.detach(),
        }

        return logits, final_representation, losses, diagnostics


# In[12]:


# ============== Evaluation and Training Loops ==============

def evaluate_loop(model: SARGCL, loader: DataLoader, device, return_preds=False):
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

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1
    }

    if return_preds:
        return metrics, all_preds, all_image_paths
    return metrics

def evaluate_comment5_uncertainty(
    model: SARGCL,
    loader: DataLoader,
    device,
    output_dir: str,
    label_map_inv: Dict[int, str],
    n_per_group: int = 3,
):
    """Generate a trustworthy OT/uncertainty table from the actual forward pass.

    Samples for which OT could not be computed are marked invalid_for_ot and are
    excluded from OT-based low/medium/high grouping. They are never treated as
    OT cost == 0 / low uncertainty.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    rows=[]; offset=0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Comment 5 evaluation"):
            labels=batch["label"].to(device)
            logits,_,_,info=model(batch)
            probs=torch.softmax(logits,dim=1)
            preds=probs.argmax(dim=1)
            conf=probs.max(dim=1).values
            vc=info["visual_node_counts"].cpu().numpy()
            tc=info["text_node_counts"].cpu().numpy()
            vm=info["valid_ot_mask"].cpu().numpy().astype(bool)
            for j in range(labels.size(0)):
                gt=int(labels[j]); pr=int(preds[j])
                rows.append({
                    "sample_index":offset+j,
                    "image_path":batch["image_paths"][j],
                    "text":batch["raw_texts"][j],
                    "ground_truth_id":gt,
                    "ground_truth":label_map_inv.get(gt,str(gt)),
                    "prediction_id":pr,
                    "prediction":label_map_inv.get(pr,str(pr)),
                    "correct":int(gt==pr),
                    "prediction_confidence":float(conf[j]),
                    "visual_nodes":int(vc[j]),
                    "text_nodes":int(tc[j]),
                    "valid_ot":int(vm[j]),
                    "ot_cost":float(info["ot_costs"][j]),
                    "tau":float(info["tau"][j]),
                    "w_visual_structured":float(info["weights"][j,0]),
                    "w_text_structured":float(info["weights"][j,1]),
                    "w_visual_global":float(info["weights"][j,2]),
                    "w_text_global":float(info["weights"][j,3]),
                    "positive_similarity":float(info["positive_similarity"][j]),
                    "hardest_negative_similarity":float(info["hardest_negative_similarity"][j]),
                    "contrastive_margin":float(info["contrastive_margin"][j]),
                })
            offset+=labels.size(0)

    all_df=pd.DataFrame(rows)
    if all_df.empty: raise RuntimeError("No test samples collected for Comment 5 analysis.")
    all_df.to_csv(os.path.join(output_dir,"comment5_all_test_samples.csv"),index=False)

    valid=all_df[all_df.valid_ot==1].copy()
    valid_rate=100*len(valid)/len(all_df)
    diagnostic=pd.DataFrame([{
        "total_test_samples":len(all_df),
        "valid_ot_samples":len(valid),
        "invalid_ot_samples":len(all_df)-len(valid),
        "valid_ot_rate_percent":valid_rate,
        "mean_visual_nodes":all_df.visual_nodes.mean(),
        "mean_text_nodes":all_df.text_nodes.mean(),
        "zero_visual_node_rate_percent":100*(all_df.visual_nodes==0).mean(),
        "zero_text_node_rate_percent":100*(all_df.text_nodes==0).mean(),
        "mean_valid_ot_cost":valid.ot_cost.mean() if len(valid) else np.nan,
        "min_valid_ot_cost":valid.ot_cost.min() if len(valid) else np.nan,
        "max_valid_ot_cost":valid.ot_cost.max() if len(valid) else np.nan,
        "mean_valid_tau":valid.tau.mean() if len(valid) else np.nan,
        "min_valid_tau":valid.tau.min() if len(valid) else np.nan,
        "max_valid_tau":valid.tau.max() if len(valid) else np.nan,
    }])
    diagnostic.to_csv(os.path.join(output_dir,"comment5_ot_node_diagnostic.csv"),index=False)
    if valid.empty:
        raise RuntimeError("No valid OT samples. See comment5_ot_node_diagnostic.csv; do not use a zero-OT table.")

    valid=valid.sort_values("ot_cost").reset_index(drop=True)
    valid["uncertainty_rank"]=np.arange(1,len(valid)+1)
    valid["ot_percentile"]=valid.ot_cost.rank(method="average",pct=True)
    # Rank-based thirds are robust to repeated OT values.
    p=valid.ot_cost.rank(method="first",pct=True)
    valid["uncertainty_group"]=np.where(p<=1/3,"Low difficulty",np.where(p<=2/3,"Medium difficulty","High difficulty"))
    valid.to_csv(os.path.join(output_dir,"comment5_valid_ot_samples.csv"),index=False)

    order={"Low difficulty":0,"Medium difficulty":1,"High difficulty":2}
    reps=[]
    for gname in order:
        g=valid[valid.uncertainty_group==gname].copy()
        if g.empty: continue
        med=g.ot_cost.median()
        g["distance_to_group_median"]=(g.ot_cost-med).abs()
        reps.append(g.sort_values("distance_to_group_median").head(n_per_group))
    rep=pd.concat(reps,ignore_index=True)
    rep["group_order"]=rep.uncertainty_group.map(order)
    rep=rep.sort_values(["group_order","ot_cost"]).drop(columns=["group_order"]).reset_index(drop=True)
    rep["example_id"]=np.arange(1,len(rep)+1)
    rep.to_csv(os.path.join(output_dir,"comment5_representative_samples.csv"),index=False)

    summary=(valid.groupby("uncertainty_group",sort=False).agg(
        samples=("sample_index","count"),
        mean_ot_cost=("ot_cost","mean"),
        mean_tau=("tau","mean"),
        mean_positive_similarity=("positive_similarity","mean"),
        mean_hardest_negative_similarity=("hardest_negative_similarity","mean"),
        mean_contrastive_margin=("contrastive_margin","mean"),
        mean_confidence=("prediction_confidence","mean"),
        accuracy=("correct","mean"),
        mean_w_visual_structured=("w_visual_structured","mean"),
        mean_w_text_structured=("w_text_structured","mean"),
        mean_w_visual_global=("w_visual_global","mean"),
        mean_w_text_global=("w_text_global","mean"),
    ).reset_index())
    summary["group_order"]=summary.uncertainty_group.map(order)
    summary=summary.sort_values("group_order").drop(columns="group_order")
    summary.to_csv(os.path.join(output_dir,"comment5_uncertainty_summary.csv"),index=False)

    # Representative LaTeX table
    tex=rep[["example_id","uncertainty_group","visual_nodes","text_nodes","ground_truth","prediction","ot_cost","tau","w_visual_structured","w_text_structured","w_visual_global","w_text_global","positive_similarity","hardest_negative_similarity","contrastive_margin"]].copy()
    tex.columns=["ID","Difficulty","$N_V$","$N_T$","GT","Pred.","OT Cost",r"$\\tau_i$",r"$w_V$",r"$w_T$",r"$w_{gV}$",r"$w_{gT}$",r"$s^+$",r"$s^-_{\\max}$","Margin"]
    for c in ["OT Cost",r"$\\tau_i$",r"$w_V$",r"$w_T$",r"$w_{gV}$",r"$w_{gT}$",r"$s^+$",r"$s^-_{\\max}$","Margin"]:
        tex[c]=pd.to_numeric(tex[c],errors="coerce").map(lambda x:f"{x:.4f}" if pd.notna(x) else "")
    for c in ["Difficulty","GT","Pred."]:
        tex[c]=tex[c].astype(str).str.replace("&",r"\\&",regex=False).str.replace("_",r"\\_",regex=False).str.replace("%",r"\\%",regex=False)
    tex.to_latex(os.path.join(output_dir,"comment5_representative_samples.tex"),index=False,escape=False,column_format="c c c c c c c c c c c c c c c")

    # Compact manuscript summary LaTeX table
    st=summary[["uncertainty_group","samples","mean_ot_cost","mean_tau","mean_positive_similarity","mean_hardest_negative_similarity","mean_contrastive_margin","accuracy"]].copy()
    st.columns=["Difficulty","Samples","OT Cost",r"$\\tau_i$",r"$s^+$",r"$s^-_{\\max}$","Margin","Accuracy"]
    for c in ["OT Cost",r"$\\tau_i$",r"$s^+$",r"$s^-_{\\max}$","Margin","Accuracy"]:
        st[c]=pd.to_numeric(st[c],errors="coerce").map(lambda x:f"{x:.4f}" if pd.notna(x) else "")
    st.to_latex(os.path.join(output_dir,"comment5_uncertainty_summary.tex"),index=False,escape=False,column_format="c c c c c c c c")

    # Plot only valid OT samples.
    plt.figure(figsize=(7,5))
    plt.scatter(valid.ot_cost.to_numpy(),valid.tau.to_numpy(),s=14,alpha=.55)
    plt.xlabel("OT-derived correspondence difficulty")
    plt.ylabel(r"Dynamic temperature $\\tau_i$")
    plt.title("OT Difficulty vs. Dynamic Contrastive Temperature")
    plt.grid(alpha=.2); plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"comment5_ot_cost_vs_temperature.png"),dpi=300,bbox_inches="tight")
    plt.close()

    print("\\n"+"="*80)
    print("COMMENT 5 ANALYSIS")
    print("="*80)
    print(f"Total test samples : {len(all_df)}")
    print(f"Valid OT samples   : {len(valid)} ({valid_rate:.2f}%)")
    print(f"Zero visual nodes  : {100*(all_df.visual_nodes==0).mean():.2f}%")
    print(f"Zero text nodes    : {100*(all_df.text_nodes==0).mean():.2f}%")
    print(f"Valid OT range     : {valid.ot_cost.min():.6f} -- {valid.ot_cost.max():.6f}")
    print(f"Valid tau range    : {valid.tau.min():.6f} -- {valid.tau.max():.6f}")
    print("\\nSummary table:")
    print(summary.to_string(index=False))
    print("\\nFiles saved under:",output_dir)
    print("  comment5_ot_node_diagnostic.csv")
    print("  comment5_valid_ot_samples.csv")
    print("  comment5_uncertainty_summary.csv")
    print("  comment5_uncertainty_summary.tex")
    print("  comment5_representative_samples.csv")
    print("  comment5_representative_samples.tex")
    print("  comment5_ot_cost_vs_temperature.png")
    print("="*80)
    return all_df,rep,summary



def append_training_log(log_path, message):
    """
    Append one timestamped message to the persistent training log.

    This helper is intentionally defined before train_loop() because the
    training loop calls it at startup, after saving checkpoints, and when
    training finishes.
    """
    log_path = str(log_path)

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def train_loop(
    model: SARGCL,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer,
    device,
    epochs=50,
    lambda_hc_cl=0.1,
    lambda_ot=0.1,
    grad_clip=1.0,
    model_save_path="best_sargcl_clip_model_new.pth",
    final_model_save_path="sargcl_clip_model_epoch50_final.pth",
    history_csv_path="training_history.csv",
    log_path="training.log",
):
    """Train SARGCL and save complete epoch-wise logs plus best/final checkpoints."""
    best_valid_f1 = -1.0
    history = []

    history_dir = os.path.dirname(history_csv_path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    # Start a persistent human-readable training log.
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SARGCL 50-Epoch Training Log\n")
        f.write("================================\n")

    append_training_log(log_path, f"Starting training for {epochs} epochs")
    print("Starting full SARGCL training...")
    print(f"Epochs: {epochs}")
    print(f"Best checkpoint: {model_save_path}")
    print(f"Final checkpoint: {final_model_save_path}")
    print(f"History CSV: {history_csv_path}")

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
            train_loader,
            desc=f"Epoch {ep}/{epochs} [Training]"
        )

        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            _, _, losses, batch_diag = model(
                batch,
                lambda_hc_cl=lambda_hc_cl,
                lambda_ot=lambda_ot,
            )

            with torch.no_grad():
                vc=batch_diag["visual_node_counts"]
                tc=batch_diag["text_node_counts"]
                vm=batch_diag["valid_ot_mask"]
                oc=batch_diag["ot_costs"]
                ti=batch_diag["tau"]
                diag_total += int(vc.numel())
                diag_valid_ot += int(vm.sum().item())
                diag_zero_visual += int((vc==0).sum().item())
                diag_zero_text += int((tc==0).sum().item())
                if vm.any():
                    diag_ot_sum += float(oc[vm].sum().item())
                    diag_tau_sum += float(ti[vm].sum().item())

            loss = losses["total"]
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_train_loss += float(loss.item())
            total_train_cls += float(losses.get("classification", 0.0).item())
            total_train_ot += float(losses.get("ot", 0.0).item())
            total_train_hc += float(losses.get("hc_cl", 0.0).item())

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
        diag_zero_visual_rate = 100.0 * diag_zero_visual / max(1, diag_total)
        diag_zero_text_rate = 100.0 * diag_zero_text / max(1, diag_total)
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
            "diagnostic_samples": diag_total,
            "valid_ot_samples": diag_valid_ot,
            "valid_ot_rate_percent": diag_valid_rate,
            "zero_visual_node_rate_percent": diag_zero_visual_rate,
            "zero_text_node_rate_percent": diag_zero_text_rate,
            "mean_valid_ot_cost": diag_mean_ot,
            "mean_valid_tau": diag_mean_tau,
            # Batch-size analysis requested by reviewer: each anchor has B-1
            # nominal in-batch negatives for a batch of size B.
            "batch_size": train_loader.batch_size,
            "nominal_negatives_per_anchor": max(0, train_loader.batch_size - 1),
            "optimizer_updates_per_epoch": len(train_loader),
            "samples_per_epoch": len(train_loader.dataset),
            "effective_ot_samples_per_epoch": diag_valid_ot,
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
            f"MeanOT={diag_mean_ot:.4f} | "
            f"Time={epoch_time/60:.2f} min"
        )


    # Explicit final checkpoint after all 50 epochs.
    torch.save(model.state_dict(), final_model_save_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)

    print("Training finished.")
    print(f"Best validation F1: {best_valid_f1:.4f}")
    print(f"Best checkpoint: {model_save_path}")
    print(f"Final epoch-{epochs} checkpoint: {final_model_save_path}")
    print(f"Training history: {history_csv_path}")
    print(f"Training log: {log_path}")
    append_training_log(
        log_path,
        f"Training finished. Best Val F1={best_valid_f1:.6f}; final checkpoint={final_model_save_path}",
    )

    return history_df



def save_training_summary_latex(output_dir, history_df):
    """Save compact training-history LaTeX tables for the reviewer response."""
    path = os.path.join(output_dir, "training_history_summary.tex")
    cols = [
        "epoch", "train_loss", "train_classification_loss",
        "train_ot_loss", "train_hc_cl_loss", "val_loss",
        "val_accuracy", "val_f1", "valid_ot_rate_percent",
        "zero_visual_node_rate_percent", "mean_valid_ot_cost",
        "mean_valid_tau", "batch_size",
        "nominal_negatives_per_anchor", "optimizer_updates_per_epoch",
    ]
    x = history_df[[c for c in cols if c in history_df.columns]].copy()
    rename = {
        "epoch": "Epoch", "train_loss": "Train Loss",
        "train_classification_loss": "Cls. Loss", "train_ot_loss": "OT Loss",
        "train_hc_cl_loss": "HC-CL Loss", "val_loss": "Val. Loss",
        "val_accuracy": "Val. Acc.", "val_f1": "Val. F1",
        "valid_ot_rate_percent": r"Valid OT (\%)",
        "zero_visual_node_rate_percent": r"Zero Visual Nodes (\%)",
        "mean_valid_ot_cost": "Mean OT", "mean_valid_tau": r"Mean $\tau$",
        "batch_size": "Batch", "nominal_negatives_per_anchor": "Neg./Anchor",
        "optimizer_updates_per_epoch": "Updates/Epoch",
    }
    x.rename(columns=rename, inplace=True)
    x.to_latex(path, index=False, escape=False, float_format="%.4f")
    return path


def save_batch_size_accounting(output_dir, train_loader):
    """Record the exact negative count/update count for the actual run."""
    b = int(train_loader.batch_size)
    n = len(train_loader.dataset)
    updates = len(train_loader)
    rows = [{
        "batch_size": b,
        "nominal_negatives_per_anchor": max(0, b - 1),
        "samples_per_epoch": n,
        "optimizer_updates_per_epoch": updates,
        "epochs": EPOCHS,
        "nominal_negative_exposures_per_epoch": n * max(0, b - 1),
    }]
    df = pd.DataFrame(rows)
    csv = os.path.join(output_dir, "batch_size_accounting.csv")
    tex = os.path.join(output_dir, "batch_size_accounting.tex")
    df.to_csv(csv, index=False)
    x = df.copy()
    x.columns = [
        "Batch Size", "Nominal Neg./Anchor", "Samples/Epoch",
        "Updates/Epoch", "Epochs", "Nominal Negative Exposures/Epoch"
    ]
    x.to_latex(tex, index=False, escape=False, float_format="%.0f")
    return csv, tex


def evaluate_comment5_all_splits(model, loaders, device, output_dir, label_map_inv, n_per_group=3):
    """Run Comment-5 OT/uncertainty analysis independently on train/val/test."""
    all_summaries = []
    all_diagnostics = []
    for split_name, loader in loaders.items():
        split_dir = os.path.join(output_dir, f"comment5_{split_name.lower()}")
        os.makedirs(split_dir, exist_ok=True)
        all_df, rep, summary = evaluate_comment5_uncertainty(
            model=model,
            loader=loader,
            device=device,
            output_dir=split_dir,
            label_map_inv=label_map_inv,
            n_per_group=n_per_group,
        )
        summary = summary.copy()
        summary.insert(0, "split", split_name)
        all_summaries.append(summary)
        diag_path = os.path.join(split_dir, "comment5_ot_node_diagnostic.csv")
        if os.path.exists(diag_path):
            d = pd.read_csv(diag_path)
            d.insert(0, "split", split_name)
            all_diagnostics.append(d)
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "comment5_uncertainty_summary_all_splits.csv"), index=False)
        compact = combined.copy()
        compact.columns = [str(c) for c in compact.columns]
        compact.to_latex(
            os.path.join(output_dir, "comment5_uncertainty_summary_all_splits.tex"),
            index=False, escape=False, float_format="%.4f"
        )
    if all_diagnostics:
        pd.concat(all_diagnostics, ignore_index=True).to_csv(
            os.path.join(output_dir, "comment5_ot_node_diagnostic_all_splits.csv"), index=False
        )





# ============================================================
# FULL IMAGE PATH SANITY CHECK
# ============================================================

def full_image_path_sanity_check(datasets):
    """
    Verify every image path in train/validation/test BEFORE OT sanity
    checking or training. No missing image is converted to a black image.
    """
    print("\n" + "=" * 90)
    print("FULL DATASET IMAGE PATH SANITY CHECK")
    print("=" * 90)

    total_missing = 0

    for split_name, dataset in datasets.items():
        missing = []

        for idx in range(len(dataset.df)):
            raw = str(dataset.df.iloc[idx]["image"]).strip()

            if os.path.isabs(raw):
                resolved = os.path.normpath(raw)
            else:
                clean = raw.replace("\\", "/").lstrip("/")

                if clean.startswith("data_image/"):
                    clean = clean[len("data_image/"):]

                resolved = os.path.normpath(
                    os.path.join(
                        dataset.image_base_path,
                        clean,
                    )
                )

            if not os.path.isfile(resolved):
                missing.append(
                    (idx, raw, resolved)
                )

        print(
            f"{split_name:12s}: "
            f"{len(dataset.df):5d} samples | "
            f"missing: {len(missing):5d}"
        )

        for idx, raw, resolved in missing[:5]:
            print(f"  index    : {idx}")
            print(f"  TSV      : {raw}")
            print(f"  resolved : {resolved}")

        total_missing += len(missing)

    print("=" * 90)

    if total_missing:
        raise FileNotFoundError(
            f"IMAGE PATH SANITY CHECK FAILED: "
            f"{total_missing} missing image(s). "
            f"Training stopped before OT sanity checking."
        )

    print("IMAGE PATH SANITY CHECK PASSED.")
    print("All train/validation/test image files exist.")
    print("=" * 90)


def run_ot_gradient_sanity_check(model, loader, device, max_batches=100):
    """Validate image paths, node extraction, OT availability, and OT gradients.

    We deliberately inspect more than eight batches because the first batch(es)
    can be unrepresentative. If zero visual nodes persist, print the actual image
    paths and node counts so the run fails for a diagnosable reason rather than
    silently training a model with no OT supervision.
    """
    model.train()
    found_valid = False
    checked = 0
    total_samples = 0
    total_visual_zero = 0
    total_text_zero = 0
    total_valid_ot = 0

    for batch in loader:
        checked += 1
        model.zero_grad(set_to_none=True)
        _, _, losses, diag = model(batch, lambda_hc_cl=0.0, lambda_ot=1.0)

        vc = diag["visual_node_counts"].detach().cpu()
        tc = diag["text_node_counts"].detach().cpu()
        vm = diag["valid_ot_mask"].detach().cpu()
        total_samples += int(vc.numel())
        total_visual_zero += int((vc == 0).sum())
        total_text_zero += int((tc == 0).sum())
        total_valid_ot += int(vm.sum())

        print(
            f"OT sanity batch {checked:03d}: "
            f"visual_nodes={vc.tolist()} | text_nodes={tc.tolist()} | "
            f"valid_OT={int(vm.sum())}/{len(vm)}"
        )

        if int(vm.sum()) == 0:
            if checked >= max_batches:
                break
            continue

        found_valid = True
        ot_value = float(losses["ot"].detach().cpu())
        if not np.isfinite(ot_value):
            raise RuntimeError("OT gradient sanity check failed: OT loss is non-finite.")

        losses["ot"].backward()
        grad_sum = 0.0
        grad_tensors = 0
        ot_related_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = float(param.grad.detach().abs().sum().cpu())
                grad_sum += g
                grad_tensors += 1
                if any(k in name for k in ["hgat_visual", "visual_refiner", "classifier", "uncert_mlp"]):
                    ot_related_grad += g

        print("\n" + "=" * 90)
        print("OT GRADIENT SANITY CHECK")
        print("=" * 90)
        print(f"Batches inspected          : {checked}")
        print(f"Samples inspected          : {total_samples}")
        print(f"Valid OT samples           : {total_valid_ot}")
        print(f"Valid OT rate              : {100.0 * total_valid_ot / max(1,total_samples):.2f}%")
        print(f"Zero visual nodes          : {total_visual_zero}")
        print(f"Zero text nodes            : {total_text_zero}")
        print(f"Current differentiable OT  : {ot_value:.8f}")
        print(f"Parameters with grad       : {grad_tensors}")
        print(f"Total |OT gradient|        : {grad_sum:.8e}")
        print(f"OT-related gradient mass   : {ot_related_grad:.8e}")
        print("=" * 90)

        if grad_tensors == 0 or grad_sum <= 0.0:
            raise RuntimeError(
                "OT gradient sanity check FAILED: OT has no live gradient path."
            )

        model.zero_grad(set_to_none=True)
        print("OT gradient sanity check PASSED.\n")
        return True

    model.zero_grad(set_to_none=True)
    rate = 100.0 * total_valid_ot / max(1, total_samples)
    print("\n" + "=" * 90)
    print("OT NODE EXTRACTION DIAGNOSTIC FAILED")
    print("=" * 90)
    print(f"Batches inspected : {checked}")
    print(f"Samples inspected : {total_samples}")
    print(f"Valid OT samples  : {total_valid_ot} ({rate:.2f}%)")
    print(f"Zero visual nodes : {total_visual_zero} ({100*total_visual_zero/max(1,total_samples):.2f}%)")
    print(f"Zero text nodes   : {total_text_zero} ({100*total_text_zero/max(1,total_samples):.2f}%)")
    print("=" * 90)
    raise RuntimeError(
        f"No valid OT sample found in the first {checked} batches. "
        "The run was stopped before 50 epochs. This normally indicates an "
        "image-path/detector/node-extraction problem, not an OT-gradient problem. "
        "The corrected dataset loader now refuses black-image substitution."
    )


def evaluate_fallback_frequency(model, loader, device, output_dir, split_name="Test"):
    """Measure true all-node visual fallback usage separately from 0/1-node cases."""
    model.eval()
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Fallback analysis ({split_name})"):
            _, _, _, info = model(batch)
            vc = info["visual_node_counts"].cpu().numpy()
            fc = info["fallback_used"].cpu().numpy().astype(bool)
            ec = info["visual_edge_counts"].cpu().numpy()
            for j in range(len(vc)):
                rows.append({
                    "split": split_name,
                    "sample_index": offset + j,
                    "image_path": batch["image_paths"][j],
                    "visual_nodes": int(vc[j]),
                    "visual_edges": int(ec[j]),
                    "true_fallback_used": int(fc[j]),
                    "zero_visual_nodes": int(vc[j] == 0),
                    "single_visual_node": int(vc[j] == 1),
                })
            offset += len(vc)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"fallback_per_sample_{split_name.lower()}.csv"), index=False)
    n = max(1, len(df))
    fallback_count = int(df.true_fallback_used.sum())
    summary = pd.DataFrame([{
        "split": split_name,
        "samples": len(df),
        "true_fallback_samples": fallback_count,
        "fallback_rate_percent": 100.0 * fallback_count / n,
        "zero_visual_node_samples": int(df.zero_visual_nodes.sum()),
        "zero_visual_node_rate_percent": 100.0 * df.zero_visual_nodes.mean(),
        "single_visual_node_samples": int(df.single_visual_node.sum()),
        "single_visual_node_rate_percent": 100.0 * df.single_visual_node.mean(),
        "mean_visual_edges": float(df.visual_edges.mean()) if len(df) else 0.0,
    }])
    summary.to_csv(os.path.join(output_dir, f"fallback_summary_{split_name.lower()}.csv"), index=False)
    return df, summary


def evaluate_fallback_all_splits(model, loaders, device, output_dir):
    summaries = []
    for split, loader in loaders.items():
        _, summary = evaluate_fallback_frequency(model, loader, device, output_dir, split)
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "fallback_summary_all_splits.csv"), index=False)
    tex = combined.copy()
    tex.columns = [
        "Split", "Samples", "True Fallback", r"Fallback Rate (\%)",
        "Zero Nodes", r"Zero Node Rate (\%)", "Single Node",
        r"Single Node Rate (\%)", "Mean Visual Edges"
    ]
    tex.to_latex(
        os.path.join(output_dir, "fallback_summary_all_splits.tex"),
        index=False, escape=False, float_format="%.2f"
    )
    return combined


def save_test_predictions(model, loader, device, output_dir, label_map_inv):
    """Saves test set predictions to a CSV file."""
    print("Generating and saving test predictions...")
    _, preds, image_paths = evaluate_loop(model, loader, device, return_preds=True)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_label_id': preds
    })
    results_df['predicted_label'] = results_df['predicted_label_id'].map(label_map_inv)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to: {output_path}")

def calculate_parameters(model):
    """Calculates the number of trainable parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    return num_params

def calculate_flops(model, loader, device):
    """Calculates the FLOPs of the model for a sample batch."""
    model.eval()
    sample_batch = next(iter(loader))
    
    # We need to trace the forward pass. `thop.profile` is a good tool for this.
    # We need to provide dummy arguments to the forward pass
    
    # The forward pass of SARGCL takes the whole batch dictionary.
    # We create a wrapper or adapt the profiling call.
    def forward_wrapper(batch):
        return model(batch)

    # Note: `thop` might have trouble with complex data structures like lists of tensors.
    # A simplified FLOPs calculation might be necessary if `thop` fails.
    try:
        macs, params = profile(model, inputs=(sample_batch,), verbose=False)
        # FLOPs is approx 2 * MACs
        flops = macs * 2
        print(f"FLOPs (G): {flops / 1e9:.2f} GFLOPs")
        return flops
    except Exception as e:
        print(f"Could not calculate FLOPs automatically with `thop`: {e}")
        print("This can happen with complex models. Consider a manual FLOPs estimation if needed.")
        return -1


def calculate_inference_time(model, loader, device, num_batches=50):
    """Measures the average inference time per batch."""
    print(f"Measuring inference time over {num_batches} batches...")
    model.eval()
    total_time = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=num_batches, desc="Inference Time Test")):
            if i >= num_batches:
                break
            start_time = time.time()
            _ = model(batch)
            torch.cuda.synchronize() # Wait for GPU operations to complete
            end_time = time.time()
            total_time += (end_time - start_time)
    
    avg_time_per_batch = total_time / num_batches
    print(f"Average inference time per batch: {avg_time_per_batch * 1000:.2f} ms")
    return avg_time_per_batch


# In[14]:


# ============== Notebook Configuration / Training Run ==============

# Set these paths for your environment.
TRAIN_CSV = "/home/nagendra/TMM/task_informative_text_img_agreed_lab_train.tsv"
VALID_CSV = "/home/nagendra/TMM/task_informative_text_img_agreed_lab_dev.tsv"
TEST_CSV = "/home/nagendra/TMM/task_informative_text_img_agreed_lab_test.tsv"

OUTPUT_DIR = "./sargcl_50epoch_results"
BATCH_SIZE = 8
EPOCHS = 50
LR = 2e-5
VISUAL_CONFIDENCE = 0.10
MAX_VISUAL_NODES = 12

# Direct checkpoint evaluation.
# Set True only if you intentionally want to retrain.
RUN_TRAINING = True
LAMBDA_HC_CL = 0.1
LAMBDA_OT = 0.1
SEED = 42
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_sargcl_clip_model_new.pth")
FINAL_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "sargcl_clip_model_epoch50_final.pth")
TRAINING_HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")
TRAINING_LOG_PATH = os.path.join(OUTPUT_DIR, "training.log")

CLIP_MODEL_NAME = "ViT-B/32"
LABEL_MAP = {"not_informative": 0, "informative": 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP preprocessing.
_, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)

# Load datasets.
print("Loading data...")
try:
    df_train = pd.read_csv(TRAIN_CSV, sep="\t")
    df_valid = pd.read_csv(VALID_CSV, sep="\t")
    df_test = pd.read_csv(TEST_CSV, sep="\t")
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Could not load one of the TSV files: {e}. "
        "Please check TRAIN_CSV, VALID_CSV, and TEST_CSV."
    )

for df in (df_train, df_valid, df_test):
    df["label"] = df["label"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)

train_dataset = DisasterDataset(df_train, clip_preprocess=clip_preprocess)
valid_dataset = DisasterDataset(df_valid, clip_preprocess=clip_preprocess)
test_dataset = DisasterDataset(df_test, clip_preprocess=clip_preprocess)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

# -------------------------------------------------------------------------
# RAW DATALOADER SANITY CHECK
# -------------------------------------------------------------------------
# This is deliberately performed before the expensive cache stage.
print("\n" + "=" * 90)
print("RAW DATALOADER SANITY CHECK")
print("=" * 90)

_debug_batch = next(iter(train_loader))

print(
    "pixel_values :",
    tuple(_debug_batch["pixel_values"].shape),
    _debug_batch["pixel_values"].dtype,
)
print(
    "input_ids    :",
    tuple(_debug_batch["input_ids"].shape),
    _debug_batch["input_ids"].dtype,
)
print(
    "labels       :",
    tuple(_debug_batch["label"].shape),
    _debug_batch["label"].dtype,
)
print(
    "raw_images   :",
    len(_debug_batch["raw_images"]),
)
print(
    "raw_texts    :",
    len(_debug_batch["raw_texts"]),
)

if _debug_batch["pixel_values"].ndim != 4:
    raise RuntimeError(
        "Dataloader sanity check failed: pixel_values must be [B,3,H,W]."
    )

if _debug_batch["pixel_values"].shape[1] != 3:
    raise RuntimeError(
        "Dataloader sanity check failed: pixel_values must have 3 channels."
    )

if _debug_batch["input_ids"].ndim != 2:
    raise RuntimeError(
        "Dataloader sanity check failed: input_ids must be [B,77]."
    )

print(
    "First image path:",
    _debug_batch["image_paths"][0],
)
print("=" * 90)
print("RAW DATALOADER SANITY CHECK PASSED.")
print("=" * 90)

# Initialize model and optimizer.

# =============================================================================
# FROZEN FEATURE CACHE + FAST 50-EPOCH TRAINING
# =============================================================================
# Faster R-CNN, BLIP, spaCy, and frozen CLIP are preprocessing components.
# They are executed ONCE per split and their outputs are cached. The repeated
# training loop contains the complete trainable SARGCL pathway:
# visual hypergraph refinement -> HGAT -> OT/Sinkhorn ->
# difficulty-regulated contrastive learning -> classifier.
# =============================================================================

CACHE_DIR = "./sargcl_feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class CachedSARGCLDataset(Dataset):
    def __init__(self, cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        self.samples = obj["samples"]
        self.split = obj["split"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cached_collate_fn(batch):
    return {
        "gV_clip": torch.stack([x["gV_clip"] for x in batch]),
        "gT_clip": torch.stack([x["gT_clip"] for x in batch]),
        "visual_nodes": [x["visual_nodes"] for x in batch],
        "text_nodes": [x["text_nodes"] for x in batch],
        "text_hyperedges": [x["text_hyperedges"] for x in batch],
        "label": torch.stack([x["label"] for x in batch]),
        "image_paths": [x["image_path"] for x in batch],
        "raw_texts": [x["raw_text"] for x in batch],
        "captions": [x["captions"] for x in batch],
        "visual_boxes": [x["visual_boxes"] for x in batch],
    }


@torch.no_grad()
def build_frozen_cache(model, loader, split_name, cache_path, device):
    """Run expensive frozen preprocessing exactly once."""
    if os.path.exists(cache_path):
        try:
            old = torch.load(cache_path, map_location="cpu")
            if len(old.get("samples", [])) == len(loader.dataset):
                print(f"Using existing {split_name} cache: {cache_path}")
                return
        except Exception:
            pass

    model.eval()
    samples = []

    print("\n" + "=" * 90)
    print(f"BUILDING FROZEN CACHE: {split_name.upper()}")
    print("=" * 90)
    print("Faster R-CNN + BLIP + spaCy + frozen CLIP are executed ONCE.")
    print("They will NOT run again during the 50 training epochs.")
    print("=" * 90)

    for batch in tqdm(loader, desc=f"Caching {split_name}"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        # Frozen global CLIP features.
        gV = model.enc.encode_image_global(pixel_values).cpu()
        gT = model.enc.encode_text_global(input_ids).cpu()

        # Frozen detector + BLIP caption generation.
        _, captions_list = model.visual_nodes(batch["raw_images"])

        # Frozen CLIP encoding of visual captions.
        visual_nodes = [
            model.enc.encode_text_for_nodes(caps, device).cpu()
            for caps in captions_list
        ]

        # Frozen spaCy text nodes + frozen CLIP encoding.
        text_nodes_list, text_hyperedges = (
            model.text_nodes.extract_nodes_and_hyperedges(
                batch["raw_texts"]
            )
        )

        text_nodes = [
            model.enc.encode_text_for_nodes(nodes, device).cpu()
            for nodes in text_nodes_list
        ]

        for i in range(len(batch["raw_texts"])):
            samples.append({
                "gV_clip": gV[i].clone(),
                "gT_clip": gT[i].clone(),
                "visual_nodes": visual_nodes[i].clone(),
                "text_nodes": text_nodes[i].clone(),
                "text_hyperedges": text_hyperedges[i],
                "label": batch["label"][i].cpu().clone(),
                "image_path": batch["image_paths"][i],
                "raw_text": batch["raw_texts"][i],
                "captions": captions_list[i],
                "visual_boxes": [],
            })

    torch.save(
        {
            "split": split_name,
            "num_samples": len(samples),
            "clip_dim": int(model.clip_dim),
            "visual_conf": float(model.visual_nodes.min_conf),
            "max_visual_nodes": int(model.visual_nodes.max_nodes),
            "samples": samples,
        },
        cache_path,
    )

    print(
        f"Saved {split_name} cache: {cache_path} "
        f"({len(samples)} samples)"
    )


class CachedSARGCL(nn.Module):
    """Trainable SARGCL core using frozen cached representations."""

    def __init__(self, base):
        super().__init__()

        self.device = base.device
        self.clip_dim = base.clip_dim

        # Only trainable SARGCL modules are registered here.
        self.visual_refiner = base.visual_refiner
        self.hgat_visual = base.hgat_visual
        self.hgat_text = base.hgat_text
        self.uncert_mlp = base.uncert_mlp
        self.classifier = base.classifier
        self.temp_mlp = base.temp_mlp

        self.tau_min = base.tau_min
        self.tau_max = base.tau_max
        self.tau_base = base.tau_base

    def build_incidence(self, num_nodes, hyperedges):
        if num_nodes == 0:
            return torch.zeros(0, 0, device=self.device)

        E = len(hyperedges)
        H = torch.zeros(
            num_nodes,
            max(E, 1),
            device=self.device,
        )

        if E == 0:
            H[:, 0] = 1.0
            return H

        for e_idx, nodes in enumerate(hyperedges):
            for n in nodes:
                if 0 <= int(n) < num_nodes:
                    H[int(n), e_idx] = 1.0

        return H

    def compute_dynamic_temperature(self, ot_costs):
        with torch.no_grad():
            x = ot_costs.unsqueeze(-1)

        factor = self.temp_mlp(x)

        return (
            self.tau_min
            + (self.tau_max - self.tau_min)
            * factor.squeeze(-1)
        )

    def lhc_cl_loss(self, zV, zT, ot_costs):
        B = zV.size(0)

        if B == 0:
            return torch.zeros(
                (),
                device=self.device,
            )

        tau_i = self.compute_dynamic_temperature(ot_costs)

        zV_norm = F.normalize(zV, dim=-1)
        zT_norm = F.normalize(zT, dim=-1)

        sim_matrix = zV_norm @ zT_norm.t()

        sim_positive = torch.diag(sim_matrix) / tau_i

        sim_negatives = sim_matrix / self.tau_base

        mask = torch.eye(
            B,
            device=self.device,
            dtype=torch.bool,
        )

        sim_negatives = sim_negatives.masked_fill(
            mask,
            -1e9,
        )

        denominator = (
            torch.exp(sim_positive)
            + torch.sum(torch.exp(sim_negatives), dim=1)
        )

        numerator = torch.exp(sim_positive)

        return -torch.log(
            numerator / (denominator + 1e-8)
        ).mean()

    def forward(self, batch, lambda_hc_cl=0.1, lambda_ot=0.1):
        gV = batch["gV_clip"].to(
            self.device,
            non_blocking=True,
        )
        gT = batch["gT_clip"].to(
            self.device,
            non_blocking=True,
        )

        B = gV.size(0)

        visual_nodes = [
            x.to(self.device, non_blocking=True)
            for x in batch["visual_nodes"]
        ]

        text_nodes = [
            x.to(self.device, non_blocking=True)
            for x in batch["text_nodes"]
        ]

        # SAME visual hypergraph refinement as the original model.
        refined_visual = self.visual_refiner(
            visual_nodes,
            gT,
        )

        zV_nodes = []
        zT_nodes = []
        zV_summaries = []
        zT_summaries = []

        for i in range(B):
            V_i, edges_i, _fallback_i = refined_visual[i]

            if V_i.size(0) > 0:
                H_V = self.build_incidence(
                    V_i.size(0),
                    edges_i,
                )
                zV_i = self.hgat_visual(
                    V_i,
                    H_V,
                )
            else:
                zV_i = torch.zeros(
                    0,
                    self.clip_dim,
                    device=self.device,
                )

            zV_nodes.append(zV_i)

            if zV_i.size(0) > 0:
                zV_summaries.append(zV_i.mean(dim=0))
            else:
                zV_summaries.append(
                    torch.zeros(
                        self.clip_dim,
                        device=self.device,
                    )
                )

            T_i = text_nodes[i]

            if T_i.size(0) > 0:
                H_T = self.build_incidence(
                    T_i.size(0),
                    batch["text_hyperedges"][i],
                )
                zT_i = self.hgat_text(
                    T_i,
                    H_T,
                )
            else:
                zT_i = torch.zeros(
                    0,
                    self.clip_dim,
                    device=self.device,
                )

            zT_nodes.append(zT_i)

            if zT_i.size(0) > 0:
                zT_summaries.append(zT_i.mean(dim=0))
            else:
                zT_summaries.append(
                    torch.zeros(
                        self.clip_dim,
                        device=self.device,
                    )
                )

        zV = torch.stack(zV_summaries)
        zT = torch.stack(zT_summaries)

        zcontext = torch.cat(
            [zV, zT],
            dim=-1,
        )

        alphas = self.uncert_mlp(zcontext) + 1e-3
        weights = alphas / alphas.sum(
            dim=-1,
            keepdim=True,
        )

        zhybrid = (
            weights[:, 0:1] * zV
            + weights[:, 1:2] * zT
            + weights[:, 2:3] * gV
            + weights[:, 3:4] * gT
        )

        final_representation = torch.cat(
            [zhybrid, gV, gT],
            dim=-1,
        )

        logits = self.classifier(
            final_representation
        )

        losses = {
            "classification": F.cross_entropy(
                logits,
                batch["label"].to(self.device),
            )
        }

        valid_indices = [
            i
            for i in range(B)
            if zV_nodes[i].size(0) > 0
            and zT_nodes[i].size(0) > 0
        ]

        ot_costs = torch.zeros(
            B,
            device=self.device,
        )

        if valid_indices:
            vals = torch.stack([
                ot_loss(
                    zV_nodes[i],
                    zT_nodes[i],
                )
                for i in valid_indices
            ])

            ot_costs[valid_indices] = vals
            losses["ot"] = ot_costs.mean()
        else:
            losses["ot"] = (
                zV.sum() * 0.0
                + zT.sum() * 0.0
            )

        # OT cost remains DETACHED for difficulty regulation, exactly as in
        # the original SARGCL formulation.
        losses["hc_cl"] = self.lhc_cl_loss(
            zV,
            zT,
            ot_costs.detach(),
        )

        losses["total"] = (
            losses["classification"]
            + lambda_hc_cl * losses["hc_cl"]
            + lambda_ot * losses["ot"]
        )

        diagnostics = {
            "weights": weights,
            "alphas": alphas,
            "visual_nodes": [
                x.size(0)
                for x in zV_nodes
            ],
            "text_nodes": [
                x.size(0)
                for x in zT_nodes
            ],
            "valid_ot": len(valid_indices),
            "ot_costs": ot_costs.detach(),
        }

        return (
            logits,
            final_representation,
            losses,
            diagnostics,
        )


def cached_ot_gradient_sanity_check(
    model,
    loader,
    device,
):
    model.train()
    model.zero_grad(set_to_none=True)

    checked = 0
    valid = 0
    zero_v = 0
    zero_t = 0
    target_ot = None

    for batch in loader:
        _, _, losses, diag = model(batch)

        v_counts = diag["visual_nodes"]
        t_counts = diag["text_nodes"]

        checked += len(v_counts)
        valid += diag["valid_ot"]
        zero_v += sum(
            x == 0 for x in v_counts
        )
        zero_t += sum(
            x == 0 for x in t_counts
        )

        print(
            "Cached OT sanity: "
            f"visual_nodes={v_counts} | "
            f"text_nodes={t_counts} | "
            f"valid_OT={diag['valid_ot']}/{len(v_counts)}"
        )

        if diag["valid_ot"] > 0:
            target_ot = losses["ot"]
            break

    if target_ot is None:
        raise RuntimeError(
            "Cached OT sanity check FAILED: "
            "no valid OT sample was found."
        )

    target_ot.backward()

    grad_count = 0
    grad_mass = 0.0

    for p in model.parameters():
        if p.grad is not None:
            grad_count += 1
            grad_mass += float(
                p.grad.detach().abs().sum().item()
            )

    print("\n" + "=" * 90)
    print("CACHED OT GRADIENT SANITY CHECK")
    print("=" * 90)
    print(f"Samples inspected       : {checked}")
    print(f"Valid OT samples        : {valid}")
    print(
        f"Valid OT rate           : "
        f"{100.0 * valid / max(1, checked):.2f}%"
    )
    print(f"Zero visual nodes       : {zero_v}")
    print(f"Zero text nodes         : {zero_t}")
    print(
        f"Current differentiable OT: "
        f"{target_ot.item():.8f}"
    )
    print(f"Parameters with grad    : {grad_count}")
    print(
        f"Total |OT gradient|     : "
        f"{grad_mass:.8e}"
    )
    print("=" * 90)

    if grad_count == 0 or grad_mass <= 0:
        raise RuntimeError(
            "Cached OT gradient sanity check FAILED: "
            "OT produced no trainable gradient."
        )

    model.zero_grad(set_to_none=True)
    print("Cached OT gradient sanity check PASSED.")


def cached_evaluate(model, loader, device):
    model.eval()

    labels = []
    preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Cached Evaluation",
            leave=False,
        ):
            logits, _, losses, _ = model(batch)

            total_loss += float(
                losses["classification"].item()
            )

            labels.extend(
                batch["label"].numpy().tolist()
            )

            preds.extend(
                torch.argmax(
                    logits,
                    dim=1,
                ).cpu().numpy().tolist()
            )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "loss": total_loss / max(
            1,
            len(loader),
        ),
        "accuracy": accuracy_score(
            labels,
            preds,
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# Explicit cached-training hyperparameters
EPOCHS = 50
LAMBDA_HC_CL = 0.1
LAMBDA_OT = 0.1
GRAD_CLIP = 1.0

def train_cached_50_epochs(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
):
    best_f1 = -1.0
    history = []

    best_path = os.path.join(
        OUTPUT_DIR,
        "best_sargcl_clip_model_cached.pth",
    )

    final_path = os.path.join(
        OUTPUT_DIR,
        "sargcl_clip_model_cached_epoch50_final.pth",
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        "cached_training_history.csv",
    )

    log_path = os.path.join(
        OUTPUT_DIR,
        "cached_training.log",
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "SARGCL Cached 50-Epoch Training Log\n"
        )

    print("\n" + "=" * 90)
    print("STARTING CACHED 50-EPOCH SARGCL TRAINING")
    print("=" * 90)
    print(f"Best checkpoint : {best_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"History CSV     : {history_path}")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        totals = {
            "total": 0.0,
            "classification": 0.0,
            "ot": 0.0,
            "hc_cl": 0.0,
        }

        diag_total = 0
        diag_valid = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{EPOCHS} "
                "[Cached Training]"
            ),
        )

        for step, batch in enumerate(
            pbar,
            start=1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            _, _, losses, diag = model(
                batch,
                lambda_hc_cl=LAMBDA_HC_CL,
                lambda_ot=LAMBDA_OT,
            )

            losses["total"].backward()

            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

            optimizer.step()

            for key in totals:
                totals[key] += float(
                    losses[key].item()
                )

            diag_total += len(
                diag["visual_nodes"]
            )
            diag_valid += diag["valid_ot"]

            pbar.set_postfix(
                loss=(
                    f"{totals['total']/step:.4f}"
                ),
                cls=(
                    f"{totals['classification']/step:.4f}"
                ),
                ot=(
                    f"{totals['ot']/step:.4f}"
                ),
                hc=(
                    f"{totals['hc_cl']/step:.4f}"
                ),
            )

        n_batches = max(
            1,
            len(train_loader),
        )

        val_metrics = cached_evaluate(
            model,
            valid_loader,
            device,
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch": epoch,
            "train_loss": totals["total"] / n_batches,
            "train_classification_loss": (
                totals["classification"]
                / n_batches
            ),
            "train_ot_loss": (
                totals["ot"]
                / n_batches
            ),
            "train_hc_cl_loss": (
                totals["hc_cl"]
                / n_batches
            ),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1_score"],
            "epoch_time_sec": epoch_time,
            "batch_size": train_loader.batch_size,
            "nominal_negatives_per_anchor": (
                train_loader.batch_size - 1
            ),
            "optimizer_updates_per_epoch": (
                len(train_loader)
            ),
            "samples_per_epoch": (
                len(train_loader.dataset)
            ),
            "valid_ot_samples_seen": diag_valid,
            "valid_ot_rate_percent": (
                100.0
                * diag_valid
                / max(1, diag_total)
            ),
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"Epoch {epoch:02d}: "
                f"train={row['train_loss']:.6f}, "
                f"val_f1={row['val_f1']:.6f}, "
                f"time_sec={epoch_time:.2f}\n"
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train={row['train_loss']:.4f} | "
            f"Cls={row['train_classification_loss']:.4f} | "
            f"OT={row['train_ot_loss']:.4f} | "
            f"HC={row['train_hc_cl_loss']:.4f} | "
            f"Val F1={row['val_f1']:.4f} | "
            f"Time={epoch_time/60:.2f} min"
        )

        if row["val_f1"] > best_f1:
            best_f1 = row["val_f1"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_f1": best_f1,
                },
                best_path,
            )

            print(
                f"-> New best cached model saved: "
                f"{best_path} "
                f"(Val F1={best_f1:.4f})"
            )

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "val_f1": best_f1,
        },
        final_path,
    )

    print("\n" + "=" * 90)
    print("CACHED TRAINING COMPLETE")
    print("=" * 90)
    print(f"Best checkpoint : {best_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"History CSV     : {history_path}")
    print(f"Training log    : {log_path}")
    print(f"Best Val F1     : {best_f1:.4f}")
    print("=" * 90)




# =============================================================================
# N_MAX SCALABILITY / DENSITY SENSITIVITY EVALUATION
# =============================================================================
# This script DOES NOT retrain SARGCL. It evaluates the already trained
# cached checkpoint under different visual-node budgets. Existing N_max=12
# cache is reused for 4/8/12. If N_max=16 is requested, a TEST-ONLY expanded
# cache is built once with Faster R-CNN/BLIP/CLIP and then reused.
# =============================================================================

SCALABILITY_CACHE_DIR = "./sargcl_feature_cache"
SCALABILITY_OUTPUT_DIR = "./sargcl_nmax_scalability_results_v5"
SCALABILITY_CHECKPOINT = "./sargcl_results_v5/best_sargcl_clip_model_cached.pth"  # v5 model
NMAX_VALUES = [4, 8, 12, 16]
SCALABILITY_BATCH_SIZE = 32
TEST_ONLY_EXPANDED_CACHE = os.path.join(
    SCALABILITY_CACHE_DIR, "test_cache_nmax16.pt"
)

os.makedirs(SCALABILITY_OUTPUT_DIR, exist_ok=True)


def _safe_torch_load(path):
    """Load the legacy cache/checkpoint format used by this project."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("Unsupported checkpoint format.")


def _load_cached_model(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n"
            "Set SCALABILITY_CHECKPOINT to the trained cached checkpoint."
        )

    meta = _safe_torch_load(
        os.path.join(SCALABILITY_CACHE_DIR, "train_cache.pt")
    )
    clip_dim = int(meta["clip_dim"])

    # Instantiate the same architecture used for cached training.
    base = SARGCL(
        device=device,
        num_classes=NUM_CLASSES,
        clip_model_name=CLIP_MODEL_NAME,
        max_visual_nodes=12,
        visual_conf=VISUAL_CONFIDENCE,
    ).to(device)

    model = CachedSARGCL(base).to(device)
    ckpt = _safe_torch_load(checkpoint_path)
    state = _checkpoint_state_dict(ckpt)

    # Cached checkpoints can contain DataParallel prefixes.
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # v5 checkpoints have no temp_mlp (temperature is parameter-free). The
    # temperature only affects the training loss, not predictions or OT costs.
    missing = [k for k in missing if not k.startswith("temp_mlp")]
    if missing:
        raise RuntimeError(
            "Checkpoint is missing trainable SARGCL parameters:\n"
            + "\n".join(missing[:20])
        )
    if unexpected:
        print(
            "WARNING: ignoring unexpected checkpoint keys:",
            unexpected[:10],
        )

    model.eval()

    # Release expensive frozen components. Cached evaluation never uses them.
    try:
        del base.visual_nodes
        del base.text_nodes
        del base.enc
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loaded trained cached checkpoint successfully.")
    print("Cached CLIP dimension:", clip_dim)
    return model


def _load_cache(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    obj = _safe_torch_load(path)
    if "samples" not in obj:
        raise RuntimeError(f"Invalid SARGCL cache: {path}")
    if not obj["samples"]:
        raise RuntimeError(f"Empty SARGCL cache: {path}")
    return obj


def _validate_cache(obj, name, required_nmax=None):
    samples = obj["samples"]
    counts = []
    text_counts = []
    bad = []

    for i, s in enumerate(samples):
        if "visual_nodes" not in s or "text_nodes" not in s:
            bad.append(i)
            continue
        v = s["visual_nodes"]
        t = s["text_nodes"]
        if not torch.is_tensor(v) or v.ndim != 2:
            bad.append(i)
            continue
        if not torch.is_tensor(t) or t.ndim != 2:
            bad.append(i)
            continue
        if v.shape[-1] != int(obj["clip_dim"]):
            raise RuntimeError(
                f"{name}: sample {i} visual feature dimension {v.shape[-1]} "
                f"!= cached CLIP dimension {obj['clip_dim']}"
            )
        if t.shape[-1] != int(obj["clip_dim"]):
            raise RuntimeError(
                f"{name}: sample {i} text feature dimension {t.shape[-1]} "
                f"!= cached CLIP dimension {obj['clip_dim']}"
            )
        counts.append(int(v.shape[0]))
        text_counts.append(int(t.shape[0]))

    if bad:
        raise RuntimeError(
            f"{name}: malformed visual/text node tensors in samples "
            f"{bad[:10]}"
        )

    max_available = max(counts) if counts else 0
    meta_max = int(obj.get("max_visual_nodes", max_available))
    print(
        f"{name}: samples={len(samples)}, meta_max_visual_nodes={meta_max}, "
        f"actual_max_cached={max_available}, mean_visual={np.mean(counts):.3f}, "
        f"mean_text={np.mean(text_counts):.3f}"
    )

    if required_nmax is not None and max_available < required_nmax:
        raise RuntimeError(
            f"{name}: requested N_max={required_nmax}, but cache contains at "
            f"most {max_available} visual nodes. An expanded cache is required."
        )

    return np.asarray(counts), np.asarray(text_counts)


def _slice_cache(obj, nmax):
    """Create an in-memory capped view without changing the cache on disk."""
    samples = []
    for s in obj["samples"]:
        ns = dict(s)
        v = s["visual_nodes"]
        if v.size(0) > nmax:
            ns["visual_nodes"] = v[:nmax].clone()
        else:
            ns["visual_nodes"] = v.clone()
        ns["text_nodes"] = s["text_nodes"].clone()
        samples.append(ns)
    return {"samples": samples, "split": obj.get("split", "Test"), "clip_dim": obj["clip_dim"]}


class _MemoryCachedDataset(Dataset):
    def __init__(self, obj):
        self.samples = obj["samples"]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def _memory_collate(batch):
    return {
        "gV_clip": torch.stack([x["gV_clip"] for x in batch]),
        "gT_clip": torch.stack([x["gT_clip"] for x in batch]),
        "visual_nodes": [x["visual_nodes"] for x in batch],
        "text_nodes": [x["text_nodes"] for x in batch],
        "text_hyperedges": [x["text_hyperedges"] for x in batch],
        "label": torch.stack([x["label"] for x in batch]),
        "image_paths": [x.get("image_path", "") for x in batch],
        "raw_texts": [x.get("raw_text", "") for x in batch],
        "captions": [x.get("captions", []) for x in batch],
        "visual_boxes": [x.get("visual_boxes", []) for x in batch],
    }


def _make_expanded_test_cache(path):
    """Build only a test cache with N_max=16; no training is performed."""
    if os.path.isfile(path):
        obj = _load_cache(path)
        max_available = max(
            int(s["visual_nodes"].shape[0]) for s in obj["samples"]
        )
        if max_available >= 16:
            print("Existing expanded test cache found:", path)
            return obj
        print("Existing expanded cache is insufficient; rebuilding it.")

    print("\n" + "=" * 100)
    print("BUILDING TEST-ONLY EXPANDED CACHE: N_max=16")
    print("This is preprocessing only. NO training and NO checkpoint update.")
    print("Faster R-CNN + BLIP + frozen CLIP will run once on TEST only.")
    print("=" * 100)

    expanded_model = SARGCL(
        device=device,
        num_classes=NUM_CLASSES,
        clip_model_name=CLIP_MODEL_NAME,
        max_visual_nodes=16,
        visual_conf=VISUAL_CONFIDENCE,
    ).to(device)
    expanded_model.eval()

    test_loader_local = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    build_frozen_cache(
        expanded_model,
        test_loader_local,
        "test_nmax16",
        path,
        device,
    )

    del expanded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _load_cache(path)


def _evaluate_one_nmax(model, base_obj, nmax):
    capped = _slice_cache(base_obj, nmax)
    ds = _MemoryCachedDataset(capped)
    loader = DataLoader(
        ds,
        batch_size=SCALABILITY_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_memory_collate,
    )

    labels, preds = [], []
    visual_counts = []
    text_counts = []
    ot_values = []
    valid = 0
    total = 0
    cap_hit = 0
    total_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating N_max={nmax}"):
            bsz = len(batch["label"])
            start = time.perf_counter()
            logits, _, losses, diag = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - start

            p = torch.argmax(logits, dim=1).cpu().numpy()
            y = batch["label"].cpu().numpy()
            labels.extend(y.tolist())
            preds.extend(p.tolist())

            vc = np.asarray(diag["visual_nodes"], dtype=np.int64)
            tc = np.asarray(diag["text_nodes"], dtype=np.int64)
            visual_counts.extend(vc.tolist())
            text_counts.extend(tc.tolist())
            valid += int(diag["valid_ot"])
            total += bsz
            cap_hit += int(np.sum(vc >= nmax))
            ot_values.extend(diag["ot_costs"].detach().cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    elapsed_ms = 1000.0 * total_seconds / max(total, 1)
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )

    valid_ot_mask = np.asarray(ot_values) > 0.0
    valid_ot_cost = (
        float(np.mean(np.asarray(ot_values)[valid_ot_mask]))
        if valid_ot_mask.any() else np.nan
    )

    row = {
        "split": "Test",
        "N_max": int(nmax),
        "samples": int(total),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "mean_visual_nodes": float(np.mean(visual_counts)),
        "median_visual_nodes": float(np.median(visual_counts)),
        "std_visual_nodes": float(np.std(visual_counts)),
        "zero_visual_node_rate_percent": float(100.0 * np.mean(np.asarray(visual_counts) == 0)),
        "cap_hit_rate_percent": float(100.0 * cap_hit / max(total, 1)),
        "mean_text_nodes": float(np.mean(text_counts)),
        "valid_ot_rate_percent": float(100.0 * valid / max(total, 1)),
        "mean_valid_ot_cost": valid_ot_cost,
        "mean_ot_matrix_entries": float(np.mean(np.asarray(visual_counts) * np.asarray(text_counts))),
        "estimated_ot_entries_at_cap": float(nmax * np.mean(text_counts)),
        "inference_ms_per_sample": float(elapsed_ms),
        "peak_gpu_memory_gb": float(peak_gb),
    }
    return row


def _write_results(rows):
    df = pd.DataFrame(rows)
    csv_path = os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_scalability_test.csv")
    tex_path = os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_scalability_test.tex")
    df.to_csv(csv_path, index=False)

    table = df[[
        "N_max", "samples", "accuracy", "precision", "recall", "f1",
        "macro_f1", "mean_visual_nodes", "cap_hit_rate_percent",
        "valid_ot_rate_percent", "mean_valid_ot_cost",
        "inference_ms_per_sample", "peak_gpu_memory_gb"
    ]].copy()
    table.columns = [
        r"$N_{\max}$", "Samples", "Acc.", "Prec.", "Recall", "F1",
        "Macro-F1", "Mean Nodes", r"Cap Hit (\%)", r"Valid OT (\%)",
        "Mean OT", "Time (ms/sample)", "Peak GPU (GB)"
    ]
    for c in ["Acc.", "Prec.", "Recall", "F1", "Macro-F1"]:
        table[c] = table[c].map(lambda x: f"{x:.4f}")
    for c in ["Mean Nodes", "Mean OT", "Time (ms/sample)", "Peak GPU (GB)"]:
        table[c] = table[c].map(lambda x: f"{x:.3f}")
    for c in [r"Cap Hit (\%)", r"Valid OT (\%)"]:
        table[c] = table[c].map(lambda x: f"{x:.2f}")

    table.to_latex(
        tex_path,
        index=False,
        escape=False,
        column_format="rrrrrrrrrrrrr",
    )

    # Full diagnostic table.
    df.to_csv(
        os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_scalability_full.csv"),
        index=False,
    )

    # Node-distribution table for every N_max.
    dist_rows = []
    test_obj = _load_cache(os.path.join(SCALABILITY_CACHE_DIR, "test_cache.pt"))
    for nmax in NMAX_VALUES:
        source = test_obj
        if nmax == 16 and os.path.isfile(TEST_ONLY_EXPANDED_CACHE):
            source = _load_cache(TEST_ONLY_EXPANDED_CACHE)
        counts = np.asarray([int(s["visual_nodes"].shape[0]) for s in source["samples"]])
        for n in range(0, min(nmax, int(counts.max())) + 1):
            dist_rows.append({
                "N_max": nmax,
                "node_count": n,
                "samples": int(np.sum(np.minimum(counts, nmax) == n)),
                "percentage": float(100 * np.mean(np.minimum(counts, nmax) == n)),
            })
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_node_distribution.csv"),
        index=False,
    )

    print("\n" + "=" * 110)
    print("N_MAX SCALABILITY RESULTS")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
    print("Generated:")
    print("  ", csv_path)
    print("  ", tex_path)
    print("  ", os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_scalability_full.csv"))
    print("  ", os.path.join(SCALABILITY_OUTPUT_DIR, "nmax_node_distribution.csv"))
    print("=" * 110)


def main_nmax_scalability():
    print("\n" + "=" * 110)
    print("SARGCL N_MAX SCALABILITY / COMPLEX-IMAGE ANALYSIS")
    print("=" * 110)
    print("NO TRAINING WILL BE PERFORMED.")
    print("Checkpoint:", SCALABILITY_CHECKPOINT)
    print("N_max values:", NMAX_VALUES)
    print("Existing cache is reused whenever it contains enough nodes.")
    print("=" * 110)

    test_cache_path = os.path.join(SCALABILITY_CACHE_DIR, "test_cache.pt")
    base_test = _load_cache(test_cache_path)
    _validate_cache(base_test, "Test cache")

    max_cached = max(int(s["visual_nodes"].shape[0]) for s in base_test["samples"])
    print(f"Current test cache maximum visual nodes: {max_cached}")

    expanded = None
    if 16 in NMAX_VALUES and max_cached < 16:
        expanded = _make_expanded_test_cache(TEST_ONLY_EXPANDED_CACHE)
        _validate_cache(expanded, "Expanded N_max=16 test cache", required_nmax=16)

    model = _load_cached_model(SCALABILITY_CHECKPOINT)

    rows = []
    for nmax in NMAX_VALUES:
        if nmax <= max_cached:
            source = base_test
        elif nmax == 16 and expanded is not None:
            source = expanded
        else:
            raise RuntimeError(
                f"Cannot evaluate N_max={nmax}: no cache contains that many nodes."
            )
        rows.append(_evaluate_one_nmax(model, source, nmax))

    _write_results(rows)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main_nmax_scalability()
