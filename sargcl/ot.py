"""Sinkhorn Optimal Transport loss for cross-modal node alignment."""

import torch

from .utils import cosine_sim


def sinkhorn_knopp(
    C: torch.Tensor,
    epsilon: float = 0.07,
    n_iters: int = 50,
) -> torch.Tensor:
    """Compute the entropic optimal transport plan via the Sinkhorn–Knopp algorithm.

    Args:
        C: Cost matrix ``(n, m)``.
        epsilon: Entropic regularisation strength.
        n_iters: Number of Sinkhorn iterations.

    Returns:
        Transport plan ``P`` of shape ``(n, m)``.
    """
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


def ot_loss(
    V_nodes: torch.Tensor,
    T_nodes: torch.Tensor,
    epsilon: float = 0.07,
    n_iters: int = 50,
) -> torch.Tensor:
    """Compute the OT-based cross-modal node alignment cost.

    Uses cosine distance as the ground cost and Sinkhorn to solve
    the regularised OT problem.

    Args:
        V_nodes: Visual node features ``(n, D)``.
        T_nodes: Textual node features ``(m, D)``.
        epsilon: Sinkhorn regularisation.
        n_iters: Number of Sinkhorn iterations.

    Returns:
        Scalar OT cost.
    """
    S = cosine_sim(V_nodes, T_nodes)
    C = 1.0 - S.clamp(-1, 1)
    P = sinkhorn_knopp(C, epsilon=epsilon, n_iters=n_iters)
    return (P * C).sum()
