"""Training objective of the paper: a cosine proxy loss with an additive margin on the l2-normalized descriptor,
plus mild regularizers of the geometry and the thresholds (each active only when its parameters are learned):

    L = L_proxy + lambda_geom * L_geom + lambda_omega * L_omega,   lambda_geom = 0.01, lambda_omega = 0.001,
    L_geom = 1.0 * radius prior + 0.5 * angular-gap penalty (gaps below 0.3 rad) + 0.3 * pull toward standard angles.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyCosineLoss(nn.Module):
    """Cross-entropy over s * cos(z, p_c), with the margin xi subtracted from the true class (paper: s=20, xi=0.1).
    The class proxies p_c are learned and discarded at inference."""

    def __init__(self, num_classes: int, dim: int, scale: float = 20.0, margin: float = 0.1):
        super().__init__()
        self.scale, self.margin = float(scale), float(margin)
        self.proxies = nn.Parameter(F.normalize(torch.randn(num_classes, dim), p=2, dim=1))

    def forward(self, descriptors, labels):
        cos = F.normalize(descriptors, p=2, dim=1) @ F.normalize(self.proxies, p=2, dim=1).t()
        logits = (cos - F.one_hot(labels, cos.shape[1]).to(cos.dtype) * self.margin) * self.scale
        return F.cross_entropy(logits, labels)


def geometry_regularizer(model, weights=(1.0, 0.5, 0.3), min_gap: float = 0.3):
    """Radius prior (mean squared change of the basis radii), angular-gap penalty and pull toward the standard angles,
    summed over the branches."""
    total = 0.0
    for br in model.branches.values():
        r_prior = (br.radii() - br.init_radii).pow(2).mean()
        ang = br.angles(); srt, _ = torch.sort(ang)
        gaps = torch.cat([torch.diff(srt), (2 * math.pi - srt[-1] + srt[0]).unsqueeze(0)])
        spacing = torch.relu(min_gap - gaps).pow(2).sum()
        d = (ang.unsqueeze(1) - br.standard_angles.unsqueeze(0)).abs()
        prior = torch.min(d, 2 * math.pi - d).min(dim=1).values.pow(2).sum()
        total = total + weights[0] * r_prior + weights[1] * spacing + weights[2] * prior
    return total


def threshold_regularizer(model):
    """Squared l2 norm of all per-direction thresholds."""
    return sum(br.threshold.pow(2).sum() for br in model.branches.values())


def flex_loss(model, proxy_loss, descriptors, labels, lambda_geom: float = 0.01, lambda_omega: float = 0.001):
    """The full objective; the regularizers only count when their parameters are learned."""
    loss = proxy_loss(descriptors, labels)
    groups = set(getattr(model, "learned_groups", ()))
    if groups & {"radius", "angles"}:
        loss = loss + lambda_geom * geometry_regularizer(model)
    if "threshold" in groups:
        loss = loss + lambda_omega * threshold_regularizer(model)
    return loss
