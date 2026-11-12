"""Evaluation-time postprocessing and readouts of the paper.

- uniform_u2: projects every 256-bin code histogram onto the 58 uniform patterns (at most two circular 0/1
  transitions) plus one bin that pools all non-uniform codes (13,824 -> 3,186 dimensions for the full descriptor).
- power_normalize: sign(z) |z|^alpha, then l2 normalization (alpha = 0.7 in the paper).
- standardized 1-NN: z-score with training statistics, exact Euclidean nearest neighbor (the paper's main readout).
- shrinkage LDA: z-score with training statistics, then scikit-learn's LinearDiscriminantAnalysis(solver="lsqr",
  shrinkage="auto"), whose shrinkage is the Ledoit-Wolf estimate (the paper's second readout; needs scikit-learn).
"""
from __future__ import annotations

import torch


def uniform_u2_mapping(K: int = 8) -> torch.Tensor:
    """Code -> uniform bin index (uniform codes in increasing order, then one shared non-uniform bin)."""
    uniform = [c for c in range(1 << K) if sum(((c >> i) & 1) != ((c >> ((i + 1) % K)) & 1) for i in range(K)) <= 2]
    mapping = torch.full((1 << K,), len(uniform), dtype=torch.long)
    for idx, code in enumerate(uniform):
        mapping[code] = idx
    return mapping


def uniform_u2(descriptors: torch.Tensor, K: int = 8) -> torch.Tensor:
    """(N, n * 2^K) concatenated code histograms -> (N, n * (K(K-1)+3)) uniform-pattern histograms."""
    mapping = uniform_u2_mapping(K).to(descriptors.device)
    n_bins = 1 << K; n_out = int(mapping.max()) + 1
    x = descriptors.reshape(descriptors.shape[0], -1, n_bins)
    out = torch.zeros(x.shape[0], x.shape[1], n_out, dtype=x.dtype, device=x.device)
    out.index_add_(2, mapping, x)
    return out.reshape(x.shape[0], -1)


def power_normalize(descriptors: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    y = torch.sign(descriptors) * descriptors.abs().pow(alpha)
    return y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)


def postprocess(descriptors: torch.Tensor, alpha: float = 0.7, K: int = 8) -> torch.Tensor:
    """The paper's postprocessing: uniform-u2 projection followed by power normalization."""
    return power_normalize(uniform_u2(descriptors, K), alpha)


@torch.no_grad()
def standardized_1nn(train: torch.Tensor, train_labels: torch.Tensor, test: torch.Tensor, batch_size: int = 1024):
    """Predicted labels of `test` by exact Euclidean 1-NN after z-scoring with the training statistics."""
    train, test = train.double(), test.double()
    mean, std = train.mean(0), train.std(0, unbiased=False)
    std[std < 1e-8] = 1.0
    a = ((train - mean) / std).float(); b = ((test - mean) / std).float()
    idx = torch.cat([torch.cdist(b[s:s + batch_size], a).argmin(1) for s in range(0, len(b), batch_size)])
    return train_labels[idx]


def shrinkage_lda(train: torch.Tensor, train_labels, test: torch.Tensor):
    """Predicted labels of `test` by shrinkage LDA after z-scoring with the training statistics, as in the paper:
    StandardScaler, then LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"). Needs scikit-learn."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
    except ImportError as err:
        raise ImportError('shrinkage_lda needs scikit-learn: pip install ".[lda]"') from err
    xtr = train.detach().cpu().double().numpy(); xte = test.detach().cpu().double().numpy()
    is_tensor = torch.is_tensor(train_labels)
    ytr = train_labels.detach().cpu().numpy() if is_tensor else train_labels
    scaler = StandardScaler().fit(xtr)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(scaler.transform(xtr), ytr)
    pred = lda.predict(scaler.transform(xte))
    return torch.as_tensor(pred) if is_tensor else pred
