"""Hypergraph neural network components for SARGCL.

Includes:
- Dynamic Cross-Modal Visual Hypergraph Refinement
- Hypergraph Attention Network (HGAT)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .utils import cosine_sim


# ============== Dynamic Cross-Modal Visual Hypergraph Refinement ==============


class VisualHypergraphRefiner(nn.Module):
    """Refine visual hypergraph structure using textual context.

    Prunes semantically irrelevant visual relations by scoring candidate
    hyperedges against the global textual representation.

    Args:
        clip_hidden: Dimensionality of CLIP embeddings.
        top_k_edges: Maximum number of hyperedges to retain per image.
    """

    def __init__(self, clip_hidden: int, top_k_edges: int = 8):
        super().__init__()
        self.top_k_edges = top_k_edges
        self.scorer = nn.Sequential(
            nn.Linear(clip_hidden * 2, clip_hidden),
            nn.ReLU(),
            nn.Linear(clip_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_node_feats: List[torch.Tensor],
        gT: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, List[List[int]]]]:
        """Score and select top-k visual hyperedges using textual gating.

        Args:
            visual_node_feats: Per-image node feature tensors.
            gT: Global textual embeddings ``(B, D)``.

        Returns:
            List of ``(node_features, hyperedge_list)`` per image.
        """
        B = len(visual_node_feats)
        refined = []

        for i in range(B):
            V = visual_node_feats[i]
            n = V.size(0)
            if n <= 1:
                refined.append((V, [list(range(n))]))
                continue

            with torch.no_grad():
                sims = cosine_sim(V, V)
                knn = 3 if n > 3 else max(1, n - 1)
                neighbors = torch.topk(sims, k=knn + 1, dim=1).indices[:, 1:]

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
            top = [idxs for _, idxs in ees[: self.top_k_edges]]
            if not top:
                top = [list(range(n))]
            refined.append((V, top))

        return refined


# ============== Hypergraph Attention Network (HGAT) ==============


class HGATLayer(nn.Module):
    """Single Hypergraph Attention layer.

    Implements multi-head attention on hypergraph-structured data,
    aggregating information through hyperedges.

    Args:
        in_dim: Input feature dimensionality.
        out_dim: Output dimensionality per head.
        dropout: Dropout probability.
        heads: Number of attention heads.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, heads: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.lin_node = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_edge = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HGAT layer.

        Args:
            X: Node feature matrix ``(N, in_dim)``.
            H: Incidence matrix ``(N, E)`` mapping nodes to hyperedges.

        Returns:
            Updated node features ``(N, out_dim * heads)``.
        """
        if X.size(0) != H.size(0):
            raise RuntimeError(
                f"Node feature matrix X and incidence matrix H have inconsistent "
                f"number of nodes: {X.size(0)} vs {H.size(0)}"
            )

        N, E = H.shape
        deg_e = H.sum(dim=0).clamp(min=1.0)

        E_feat = torch.matmul(H.t(), X) / deg_e.unsqueeze(-1)
        X_h = self.lin_node(X).view(N, self.heads, self.out_dim)
        E_h = self.lin_edge(E_feat).view(E, self.heads, self.out_dim)

        HX = torch.einsum("en,nhd->ehd", H.t(), X_h) / deg_e.view(E, 1, 1)

        M = HX + E_h
        alpha = torch.softmax(torch.einsum("ehd,hd->eh", M, self.attn), dim=0)
        alpha = self.dropout(alpha)

        out = torch.einsum("ne,eh,ehd->nhd", H, alpha, M).reshape(
            N, self.heads * self.out_dim
        )
        return self.act(out)


class HGATEncoder(nn.Module):
    """Multi-layer Hypergraph Attention encoder.

    Args:
        in_dim: Input feature dimensionality.
        hid_dim: Hidden layer dimensionality.
        out_dim: Output dimensionality (must be divisible by ``heads``).
        layers: Number of HGAT layers.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 256,
        out_dim: int = 256,
        layers: int = 2,
    ):
        super().__init__()
        mods = []
        heads = 2

        if layers == 1:
            if out_dim % heads != 0:
                raise ValueError(
                    f"out_dim ({out_dim}) must be divisible by heads ({heads})"
                )
            mods.append(HGATLayer(in_dim, out_dim // heads, heads=heads))
        else:
            mods.append(HGATLayer(in_dim, hid_dim, heads=heads))
            for _ in range(layers - 2):
                mods.append(HGATLayer(hid_dim * heads, hid_dim, heads=heads))
            if out_dim % heads != 0:
                raise ValueError(
                    f"out_dim ({out_dim}) must be divisible by heads ({heads})"
                )
            mods.append(HGATLayer(hid_dim * heads, out_dim // heads, heads=heads))

        self.layers = nn.ModuleList(mods)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Encode node features through stacked HGAT layers."""
        for layer in self.layers:
            X = layer(X, H)
        return X
