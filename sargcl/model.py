"""SARGCL — Full model combining all components.

State-Aware Hypergraph Representation Learning with
Difficulty-Regulated Contrastive Alignment.
"""

from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CLIPEncoders
from .visual_extractor import VisualNodeExtractor
from .text_extractor import TextNodeExtractor
from .hypergraph import VisualHypergraphRefiner, HGATEncoder
from .ot import ot_loss


class SARGCL(nn.Module):
    """State-Aware Relational Graph Contrastive Learning model.

    Integrates:
    1. CLIP-based global encoders
    2. State-aware visual node extraction (Faster R-CNN + BLIP)
    3. Syntactic textual hypergraph construction (spaCy)
    4. Cross-modal visual hypergraph refinement
    5. Hypergraph Attention Networks (HGAT) for both modalities
    6. Sinkhorn OT-based node alignment
    7. Difficulty-regulated hierarchical contrastive loss (LHC-CL)
    8. Dirichlet uncertainty-weighted fusion

    Args:
        device: Torch device.
        num_classes: Number of output classes.
        clip_model_name: CLIP architecture variant.
        blip_model: HuggingFace BLIP model identifier.
        max_visual_nodes: Maximum visual entity nodes per image.
        visual_conf: Minimum detection confidence.
        top_k_visual_edges: Maximum visual hyperedges retained.
        hgat_layers: Number of HGAT encoder layers.
        tau_min: Minimum contrastive temperature.
        tau_max: Maximum contrastive temperature.
        tau_base: Base temperature for negative pairs.
    """

    def __init__(
        self,
        device,
        num_classes: int,
        clip_model_name: str = "ViT-B/32",
        blip_model: str = "Salesforce/blip-image-captioning-base",
        max_visual_nodes: int = 12,
        visual_conf: float = 0.5,
        top_k_visual_edges: int = 8,
        hgat_layers: int = 2,
        tau_min: float = 0.05,
        tau_max: float = 0.1,
        tau_base: float = 0.07,
    ):
        super().__init__()
        self.device = device

        # Foundational encoders
        self.enc = CLIPEncoders(clip_model_name=clip_model_name, device=device)
        self.clip_dim = self.enc.text_hidden  # 512 for ViT-B/32

        # Node extractors
        self.visual_nodes = VisualNodeExtractor(
            device=device,
            blip_model=blip_model,
            max_nodes=max_visual_nodes,
            min_conf=visual_conf,
        )
        self.text_nodes = TextNodeExtractor()

        # Cross-modal refinement
        self.visual_refiner = VisualHypergraphRefiner(
            clip_hidden=self.clip_dim, top_k_edges=top_k_visual_edges
        )

        # Hypergraph attention encoders
        self.hgat_visual = HGATEncoder(
            in_dim=self.clip_dim,
            hid_dim=self.clip_dim,
            out_dim=self.clip_dim,
            layers=hgat_layers,
        )
        self.hgat_text = HGATEncoder(
            in_dim=self.clip_dim,
            hid_dim=self.clip_dim,
            out_dim=self.clip_dim,
            layers=hgat_layers,
        )

        # Dirichlet uncertainty MLP
        self.uncert_mlp = nn.Sequential(
            nn.Linear(self.clip_dim * 2, self.clip_dim),
            nn.ReLU(),
            nn.Linear(self.clip_dim, 4),
            nn.Softplus(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim * 3, self.clip_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.clip_dim, num_classes),
        )

        # Difficulty-regulated contrastive temperature parameters
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_base = tau_base
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    # ------------------------------------------------------------------
    # Helper: build incidence matrix
    # ------------------------------------------------------------------

    def build_incidence(
        self, num_nodes: int, hyperedges: List[List[int]]
    ) -> torch.Tensor:
        """Build a binary incidence matrix ``H`` of shape ``(N, E)``."""
        if num_nodes == 0:
            return torch.zeros(0, 0, device=self.device)
        E = len(hyperedges)
        H = torch.zeros(num_nodes, max(E, 1), device=self.device)
        if E == 0:
            H[:, 0] = 1.0
            return H
        for e_idx, nodes in enumerate(hyperedges):
            for n in nodes:
                if n < num_nodes:
                    H[n, e_idx] = 1.0
        return H

    # ------------------------------------------------------------------
    # LHC-CL: Hierarchical Consistency Contrastive Loss
    # ------------------------------------------------------------------

    def compute_dynamic_temperature(self, ot_costs: torch.Tensor) -> torch.Tensor:
        """Compute sample-specific temperature from OT difficulty signal."""
        with torch.no_grad():
            ot_costs_unsqueezed = ot_costs.unsqueeze(-1)
        temp_factor = self.temp_mlp(ot_costs_unsqueezed)
        return self.tau_min + (self.tau_max - self.tau_min) * temp_factor.squeeze(-1)

    def lhc_cl_loss(
        self,
        zV: torch.Tensor,
        zT: torch.Tensor,
        ot_costs: torch.Tensor,
    ) -> torch.Tensor:
        """Difficulty-regulated contrastive loss.

        Uses OT-derived per-sample temperatures for positive pairs and
        a fixed base temperature for negative pairs.
        """
        B = zV.size(0)
        if B == 0:
            return torch.tensor(0.0, device=self.device)

        tau_i = self.compute_dynamic_temperature(ot_costs)

        zV_norm = F.normalize(zV, dim=-1)
        zT_norm = F.normalize(zT, dim=-1)

        sim_matrix = zV_norm @ zT_norm.t()  # (B, B)

        # Positive pairs (diagonal)
        sim_positive = torch.diag(sim_matrix) / tau_i

        # Negative pairs (off-diagonal) with base temperature
        sim_negatives = sim_matrix / self.tau_base
        mask = torch.eye(B, device=self.device, dtype=torch.bool)
        sim_negatives.masked_fill_(mask, -1e9)

        denominator = torch.exp(sim_positive) + torch.sum(
            torch.exp(sim_negatives), dim=1
        )
        numerator = torch.exp(sim_positive)

        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],
        lambda_hc_cl: float = 0.1,
        lambda_ot: float = 0.1,
    ):
        """Full forward pass.

        Returns:
            logits: Classification logits ``(B, C)``.
            final_representation: Fused multimodal features ``(B, 3D)``.
            losses: Dictionary of loss components.
            diagnostics: Dictionary of per-sample diagnostic tensors.
        """
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        B = pixel_values.size(0)

        # Global CLIP features
        gV_clip = self.enc.encode_image_global(pixel_values)
        gT_clip = self.enc.encode_text_global(input_ids)

        # Visual node extraction & encoding
        _, captions_list = self.visual_nodes(batch["raw_images"])
        vis_node_feats = [
            self.enc.encode_text_for_nodes(caps, self.device)
            for caps in captions_list
        ]

        # Textual node extraction & encoding
        text_nodes_list, text_hyperedges = (
            self.text_nodes.extract_nodes_and_hyperedges(batch["raw_texts"])
        )
        txt_node_feats = [
            self.enc.encode_text_for_nodes(nodes, self.device)
            for nodes in text_nodes_list
        ]

        # Cross-modal visual hypergraph refinement
        refined_visual = self.visual_refiner(vis_node_feats, gT_clip)

        zV_nodes, zT_nodes, zV_summ, zT_summ = [], [], [], []

        for i in range(B):
            # Visual HGAT
            V_i, edges_i = refined_visual[i]
            if V_i.size(0) == 0:
                zVi_nodes = torch.zeros(0, self.clip_dim, device=self.device)
            else:
                zVi_nodes = self.hgat_visual(
                    V_i, self.build_incidence(V_i.size(0), edges_i)
                )
            zV_nodes.append(zVi_nodes)
            zV_summ.append(
                zVi_nodes.mean(dim=0)
                if zVi_nodes.size(0) > 0
                else torch.zeros(self.clip_dim, device=self.device)
            )

            # Textual HGAT
            T_i = txt_node_feats[i]
            HTi = self.build_incidence(T_i.size(0), text_hyperedges[i])
            if T_i.size(0) == 0:
                zTi_nodes = torch.zeros(0, self.clip_dim, device=self.device)
            else:
                zTi_nodes = self.hgat_text(T_i, HTi)
            zT_nodes.append(zTi_nodes)
            zT_summ.append(
                zTi_nodes.mean(dim=0)
                if zTi_nodes.size(0) > 0
                else torch.zeros(self.clip_dim, device=self.device)
            )

        zV, zT = torch.stack(zV_summ), torch.stack(zT_summ)

        # Dirichlet uncertainty-weighted fusion
        zcontext = torch.cat([zV, zT], dim=-1)
        alphas = self.uncert_mlp(zcontext) + 1e-3
        weights = alphas / alphas.sum(dim=-1, keepdim=True)

        zhybrid = (
            weights[:, 0:1] * zV
            + weights[:, 1:2] * zT
            + weights[:, 2:3] * gV_clip
            + weights[:, 3:4] * gT_clip
        )

        final_representation = torch.cat([zhybrid, gV_clip, gT_clip], dim=-1)
        logits = self.classifier(final_representation)

        # ------ Losses ------
        losses = {}
        if "label" in batch:
            losses["classification"] = F.cross_entropy(
                logits, batch["label"].to(self.device)
            )

        # Node-level OT alignment
        ot_vals = [
            ot_loss(zV_nodes[i], zT_nodes[i])
            for i in range(B)
            if zV_nodes[i].size(0) > 0 and zT_nodes[i].size(0) > 0
        ]

        ot_costs_tensor = torch.zeros(B, device=self.device)
        valid_ot_indices = [
            i
            for i, (v, t) in enumerate(zip(zV_nodes, zT_nodes))
            if v.size(0) > 0 and t.size(0) > 0
        ]
        if ot_vals:
            ot_costs_tensor[valid_ot_indices] = torch.stack(ot_vals)
            losses["ot"] = ot_costs_tensor.mean()
        else:
            losses["ot"] = torch.tensor(0.0, device=self.device)

        # Difficulty-regulated contrastive loss
        losses["hc_cl"] = self.lhc_cl_loss(zV, zT, ot_costs_tensor.detach())

        # ------ Diagnostics ------
        tau_i = self.compute_dynamic_temperature(ot_costs_tensor.detach())

        with torch.no_grad():
            zV_norm = F.normalize(zV, dim=-1)
            zT_norm = F.normalize(zT, dim=-1)
            sim_matrix = zV_norm @ zT_norm.t()
            positive_similarity = torch.diag(sim_matrix)

            batch_sim = sim_matrix.size(0)
            if batch_sim > 1:
                negative_matrix = sim_matrix.clone()
                negative_mask = torch.eye(
                    batch_sim, device=sim_matrix.device, dtype=torch.bool
                )
                negative_matrix.masked_fill_(negative_mask, -1e9)
                hardest_negative_similarity = negative_matrix.max(dim=1).values
            else:
                hardest_negative_similarity = torch.zeros(
                    batch_sim, device=sim_matrix.device
                )

            contrastive_margin = positive_similarity - hardest_negative_similarity

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
        }

        return logits, final_representation, losses, diagnostics
