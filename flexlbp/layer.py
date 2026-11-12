"""FLEX-LBP: learnable local binary patterns with exact binary inference.

Every forward pass computes hard binary comparisons and exact histogram counts; straight-through estimators (STE)
provide surrogate derivatives in the backward pass only, so the descriptor used during training is the descriptor
used at test time.

The layer has three branches (fine, mid, coarse). Each branch applies its own 3x3 color transform, quantizes the
result to a learnable number of gray levels (rounding STE; frozen at 256 levels unless learned) and applies an 8-neighbor
sign-plus-magnitude LBP operator with learnable radii (four basis radii, smooth directional field), angles (softmax
increments) and per-direction thresholds, on all nine channel pairs (center channel i, neighbor channel j). Like
opponent color LBP, it compares channels with each other, but it keeps both orderings of each pair. The output is the concatenation of the l1-normalized branch histograms, l2-normalized once:
3 branches x 9 pairs x 2 codes x 256 bins = 13,824 dimensions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
BRANCHES = ("fine", "mid", "coarse")

# radius templates: (initial radius, lower bound, upper bound) per branch; the paper uses compact on NewBarkTex,
# compact_parquet on Parquet and wide on every other benchmark
TEMPLATES = {
    "wide": {"fine": (2.0, 1.0, 3.0), "mid": (6.0, 4.0, 8.0), "coarse": (10.0, 7.0, 13.0)},
    "compact": {"fine": (2.0, 1.5, 4.0), "mid": (5.0, 3.5, 7.0), "coarse": (8.0, 6.5, 10.5)},
    "compact_parquet": {"fine": (2.0, 1.0, 3.0), "mid": (5.0, 3.5, 7.0), "coarse": (8.0, 6.0, 10.0)},
}

# which parameter groups each configuration of the paper learns
PRESETS = {
    "fixed": (),
    "color": ("color",),
    "radius": ("radius",),
    "threshold": ("threshold",),
    "geometry": ("radius", "angles", "threshold"),
    "mixer_q": ("color", "quantization"),
    "all_joint": ("color", "radius", "angles", "threshold"),
}


class _BitFlipHistogram(torch.autograd.Function):
    """Exact normalized histogram of the codes packed from hard bits; backward: each bit receives the loss change of
    flipping that bit alone, (g[v with bit k set] - g[v with bit k cleared]) / N (the paper's bit-aware ablation)."""

    @staticmethod
    def forward(ctx, bits, n_bins):
        B, C, K, H, W = bits.shape
        powers = (2 ** torch.arange(K, device=bits.device)).view(1, 1, K, 1, 1)
        codes = ((bits > 0.5).long() * powers).sum(dim=2).view(B, C, -1)
        hist = torch.zeros(B, C, n_bins, device=bits.device, dtype=torch.float32)
        hist.scatter_add_(2, codes, torch.ones_like(codes, dtype=torch.float32))
        denom = hist.sum(dim=-1, keepdim=True) + 1e-7
        ctx.save_for_backward(codes, denom); ctx.shape = (B, C, K, H, W)
        return hist / denom

    @staticmethod
    def backward(ctx, grad_hist):
        codes, denom = ctx.saved_tensors
        B, C, K, H, W = ctx.shape
        g = grad_hist.to(torch.float32)
        grads = [(g.gather(2, codes | (1 << k)) - g.gather(2, codes & ~(1 << k))) / denom for k in range(K)]
        return torch.stack(grads, dim=2).view(B, C, K, H, W), None


def _ste(hard, soft):
    """Straight-through estimator: the forward value is `hard`, the gradient is that of `soft`."""
    return hard + (soft - soft.detach())


class ColorTransform(nn.Module):
    """Learnable 3x3 color transform M (output channel i = sum_j M[i, j] * input channel j), initialized at I + noise."""

    def __init__(self, init_noise: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3) + torch.randn(3, 3) * init_noise)

    def forward(self, x):
        B, C, H, W = x.shape
        return torch.einsum("ij,bjk->bik", self.weight, x.reshape(B, 3, -1)).reshape(B, 3, H, W)


class FLEXLBPBranch(nn.Module):
    """One sign-plus-magnitude LBP branch with learnable geometry and thresholds (see the module docstring)."""

    def __init__(self, init_radius: float, radius_min: float, radius_max: float, K: int = 8,
                 radius_directions: int = 4, axis_temperature: float = 8.0, tau: float = 10.0,
                 input_quantization: bool = True, quantization_init: float = 256.0,
                 quantization_min: float = 8.0, quantization_max: float = 256.0, padding: str = "reflect",
                 support_size: int = 1, histogram_estimator: str = "adjacent"):
        super().__init__()
        if radius_directions not in (1, 2, 4):
            raise ValueError("radius_directions must be 1, 2 or 4")
        if support_size < 1 or support_size % 2 == 0:
            raise ValueError("support_size must be a positive odd integer")
        if histogram_estimator not in ("adjacent", "bitflip"):
            raise ValueError("histogram_estimator must be 'adjacent' or 'bitflip'")
        self.K, self.tau, self.padding = int(K), float(tau), padding
        self.radius_min, self.radius_max = float(radius_min), float(radius_max)
        self.axis_temperature = float(axis_temperature)
        self.support_size, self.histogram_estimator = int(support_size), histogram_estimator
        self.n_bins = 2 ** self.K
        init = torch.full((radius_directions,), float(init_radius))
        u = ((init - radius_min) / (radius_max - radius_min)).clamp(1e-6, 1 - 1e-6)
        self.radius_param = nn.Parameter(torch.log(u / (1 - u)))       # sigmoid-bounded basis radii
        self.angle_logits = nn.Parameter(torch.zeros(self.K))          # softmax increments, uniform at init
        self.threshold = nn.Parameter(torch.zeros(self.K))             # per-direction comparison thresholds
        self.input_quantization = bool(input_quantization)
        self.quantization_min, self.quantization_max = float(quantization_min), float(quantization_max)
        if self.input_quantization:                                    # sigmoid-bounded number of gray levels
            q = min(max((quantization_init - quantization_min) / (quantization_max - quantization_min), 1e-6), 1 - 1e-6)
            self.quantization_param = nn.Parameter(torch.tensor(math.log(q / (1 - q)), dtype=torch.float32))
        basis = torch.zeros(1) if radius_directions == 1 else torch.linspace(0, math.pi, radius_directions + 1)[:-1]
        self.register_buffer("basis_angles", basis)
        self.register_buffer("init_radii", init.clone())
        self.register_buffer("standard_angles", torch.linspace(0, 2 * math.pi, self.K + 1)[:-1])

    # ------------------------------------------------------------------ parameters
    def radii(self):
        """Basis radii in [radius_min, radius_max]."""
        return self.radius_min + (self.radius_max - self.radius_min) * torch.sigmoid(self.radius_param)

    def angles(self):
        """Ordered sampling angles covering the full circle, theta_0 = 0."""
        a = torch.cumsum(F.softmax(self.angle_logits, dim=0) * 2 * math.pi, dim=0)
        return torch.cat([torch.zeros(1, device=a.device), a[:-1]])

    def effective_radii(self, angles=None):
        """Per-direction radii: soft average of the basis radii by angular proximity to their axes."""
        angles = self.angles() if angles is None else angles
        r = self.radii()
        if r.numel() == 1:
            return r[0].expand_as(angles)
        w = F.softmax(self.axis_temperature * torch.cos(2.0 * (angles.unsqueeze(1) - self.basis_angles.view(1, -1))), dim=1)
        return (w * r.view(1, -1)).sum(dim=1)

    def quantization_levels(self):
        if not self.input_quantization:
            return None
        return self.quantization_min + (self.quantization_max - self.quantization_min) * torch.sigmoid(self.quantization_param)

    # ------------------------------------------------------------------ sampling
    def _quantize(self, x):
        if not self.input_quantization:
            return x
        steps = torch.clamp(self.quantization_levels() - 1.0, min=1.0)
        xc = x.clamp(0.0, 1.0)
        return torch.round(xc * steps) / steps + (xc - xc.detach())     # rounding STE (identity backward)

    def _support_offsets(self, device):
        if self.support_size == 1:
            z = torch.zeros(1, device=device)
            return z, z
        c = torch.arange(-(self.support_size // 2), self.support_size // 2 + 1, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        return gx.reshape(-1), gy.reshape(-1)

    def _sample(self, x, dx, dy):
        """Bilinear samples of x at (col + dx[m], row + dy[m]) on the reflect-padded image -> (B, C, M, H, W)."""
        B, C, H, W = x.shape
        pad = int(math.ceil(self.radius_max + self.support_size // 2)) + 2
        if self.padding != "reflect" or pad <= min(H - 1, W - 1):
            src, grid_padding = F.pad(x, (pad, pad, pad, pad), mode=self.padding), "zeros"
        else:                                                           # image too small for explicit reflection
            src, pad = x, 0
            grid_padding = {"reflect": "reflection", "replicate": "border"}.get(self.padding, "zeros")
        Hs, Ws = src.shape[2], src.shape[3]
        ys = torch.arange(H, device=x.device, dtype=torch.float32).view(H, 1).expand(H, W)
        xs = torch.arange(W, device=x.device, dtype=torch.float32).view(1, W).expand(H, W)
        out = []
        for m in range(dx.numel()):
            gy = 2.0 * (ys + dy[m] + pad) / (Hs - 1) - 1.0
            gx = 2.0 * (xs + dx[m] + pad) / (Ws - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            out.append(F.grid_sample(src, grid, mode="bilinear", padding_mode=grid_padding, align_corners=True))
        return torch.stack(out, dim=2)

    def _reduce(self, samples):
        return samples[:, :, 0] if samples.shape[2] == 1 else samples.median(dim=2).values

    def sample(self, x):
        """Centers (B, 3, H, W) and neighbors (B, 3, K, H, W) at the learned radii and angles."""
        ang = self.angles(); rad = self.effective_radii(ang)
        ox, oy = rad * torch.cos(ang), -rad * torch.sin(ang)            # image coordinates: y grows downward
        sx, sy = self._support_offsets(x.device)
        centers = x if self.support_size == 1 else self._reduce(self._sample(x, sx, sy))
        neighbors = torch.stack([self._reduce(self._sample(x, ox[k] + sx, oy[k] + sy)) for k in range(self.K)], dim=2)
        return centers, neighbors

    # ------------------------------------------------------------------ codes and histograms
    def _histogram(self, bits):
        """Normalized 2^K-bin histogram of the codes packed from bits (B, 1, K, H, W), with the configured STE."""
        if self.histogram_estimator == "bitflip":
            return _BitFlipHistogram.apply(bits, self.n_bins).reshape(bits.shape[0], -1)
        B = bits.shape[0]
        powers = (2 ** torch.arange(self.K, device=bits.device, dtype=torch.float32)).view(1, 1, self.K, 1, 1)
        v = (bits * powers).sum(dim=2).reshape(B, -1)                   # exact integer codes in the forward pass
        codes = v.round().long().clamp(0, self.n_bins - 1)
        hard = torch.zeros(B, self.n_bins, device=bits.device).scatter_add_(1, codes, torch.ones_like(v))
        hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-7)
        vc = v.clamp(0, self.n_bins - 1); i0 = vc.floor().long(); i1 = (i0 + 1).clamp(max=self.n_bins - 1)
        w1 = vc - i0.float()                                            # adjacent-bin (linear interpolation) surrogate
        soft = torch.zeros(B, self.n_bins, device=bits.device).scatter_add_(1, i0, 1.0 - w1).scatter_add_(1, i1, w1)
        soft = soft / (soft.sum(dim=-1, keepdim=True) + 1e-7)
        return _ste(hard, soft)

    def codes(self, x):
        """Sign and magnitude bits of every channel pair: list of (sign_bits, magnitude_bits), each (B, 1, K, H, W)."""
        centers, neighbors = self.sample(self._quantize(x))
        thr = self.threshold.view(1, 1, self.K, 1, 1); out = []
        for i, j in PAIRS:
            d = neighbors[:, j:j + 1] - centers[:, i:i + 1].unsqueeze(2)
            sign = _ste((d > thr).float(), torch.sigmoid(self.tau * (d - thr)))
            mag = d.abs(); mu = mag.detach().mean(dim=(2, 3, 4), keepdim=True)   # image mean, gradient detached
            out.append((sign, _ste((mag > mu).float(), torch.sigmoid(self.tau * (mag - mu)))))
        return out

    def forward(self, x):
        """Branch descriptor (B, 9 * 2 * 2^K): sign and magnitude histograms of the nine channel pairs."""
        return torch.cat([h for s, m in self.codes(x) for h in (self._histogram(s), self._histogram(m))], dim=1)


class FLEXLBP(nn.Module):
    """Three-branch FLEX-LBP layer: (B, 3, H, W) RGB in [0, 1] -> (B, 13,824) descriptor.

    Args:
        template: 'wide', 'compact' or 'compact_parquet' radius template, or a dict {branch: (init, min, max)}.
        learn: one of PRESETS ('fixed', 'geometry', 'all_joint', ...) or an iterable of parameter groups
            ('color', 'radius', 'angles', 'threshold', 'quantization') to learn; the others are frozen.
        K, radius_directions, axis_temperature, tau: operator settings of the paper (8, 4, 8, 10).
        color_init_noise: the color transforms start at I + noise * N(0, 1) (paper: 0.01).
        support_size: 1 = point sampling (paper); 3 = 3x3 median support (photometric remedy of the paper).
        histogram_estimator: 'adjacent' (paper) or 'bitflip' (ablation).
        input_quantization: quantize each transformed image to a learnable number of gray levels before the
            comparisons, as in all configurations of the paper (frozen unless the 'quantization' group is learned).
        quantization_init, quantization_range: initial number of gray levels and its bounds; the default init is 128
            when quantization is learned (the paper's Mixer+Q) and 256 otherwise, in [8, 256].
    """

    def __init__(self, template="wide", learn="geometry", K: int = 8, radius_directions: int = 4,
                 axis_temperature: float = 8.0, tau: float = 10.0, color_init_noise: float = 0.01,
                 support_size: int = 1, histogram_estimator: str = "adjacent", padding: str = "reflect",
                 input_quantization: bool = True, quantization_init=None, quantization_range=(8.0, 256.0)):
        super().__init__()
        bounds = TEMPLATES[template] if isinstance(template, str) else template
        groups = PRESETS[learn] if isinstance(learn, str) else tuple(learn)
        if quantization_init is None:
            quantization_init = 128.0 if "quantization" in groups else 256.0
        self.branches = nn.ModuleDict({
            b: FLEXLBPBranch(*bounds[b], K=K, radius_directions=radius_directions, axis_temperature=axis_temperature,
                             tau=tau, input_quantization=input_quantization, quantization_init=quantization_init,
                             quantization_min=quantization_range[0], quantization_max=quantization_range[1], padding=padding,
                             support_size=support_size, histogram_estimator=histogram_estimator)
            for b in BRANCHES})
        self.color = nn.ModuleDict({b: ColorTransform(color_init_noise) for b in BRANCHES})
        self.set_learnable(groups)

    @property
    def dim(self):
        return sum(br.n_bins * 2 * len(PAIRS) for br in self.branches.values())

    def set_learnable(self, groups):
        """Freeze every descriptor parameter except the listed groups."""
        groups = set(groups)
        for b in BRANCHES:
            br = self.branches[b]
            self.color[b].weight.requires_grad_("color" in groups)
            br.radius_param.requires_grad_("radius" in groups)
            br.angle_logits.requires_grad_("angles" in groups)
            br.threshold.requires_grad_("threshold" in groups)
            if br.input_quantization:
                br.quantization_param.requires_grad_("quantization" in groups)
        self.learned_groups = tuple(sorted(groups))

    def forward(self, x):
        parts = []
        for b in BRANCHES:
            h = self.branches[b](self.color[b](x))
            parts.append(h / (h.sum(dim=1, keepdim=True) + 1e-7))          # l1 per branch
        return F.normalize(torch.cat(parts, dim=1), p=2, dim=1)          # l2 once

    @torch.no_grad()
    def describe(self, x, batch_size: int = 32):
        """Descriptors of a (N, 3, H, W) tensor in evaluation mode, in batches."""
        dev = next(self.parameters()).device
        return torch.cat([self(x[s:s + batch_size].to(dev)).cpu() for s in range(0, len(x), batch_size)])

    def parameters_summary(self):
        """The interpretable parameters: radii, angles (degrees), thresholds and color transforms per branch."""
        out = {}
        for b in BRANCHES:
            br = self.branches[b]
            with torch.no_grad():
                ang = br.angles()
                out[b] = {"basis_radii": br.radii().tolist(), "radii": br.effective_radii(ang).tolist(),
                          "angles_deg": torch.rad2deg(ang).tolist(), "thresholds": br.threshold.tolist(),
                          "color_transform": self.color[b].weight.tolist()}
                if br.input_quantization:
                    out[b]["quantization_levels"] = float(br.quantization_levels())
        return out
