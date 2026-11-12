"""Reference training loop with the paper's recipe.

Adam with component-specific learning rates (radii 2e-3; color transforms, angles and class proxies 1e-3; thresholds
and quantization 5e-4), weight decay 1e-4 on the color transforms and proxies only, a step schedule that halves every
rate once (after 6, 8 or 12 epochs for 8-, 12- or 20-epoch runs), no data augmentation, and checkpoint selection by
standardized 1-NN accuracy (raw descriptors, reference set = the remaining training images) on a stratified 15% holdout,
from the second epoch on.
"""
from __future__ import annotations

import copy

import torch

from .losses import ProxyCosineLoss, flex_loss
from .postprocess import standardized_1nn

STEP_AFTER = {8: 6, 12: 8, 20: 12}


def stratified_holdout(labels: torch.Tensor, fraction: float = 0.15, seed: int = 42):
    """Indices (train, validation) with `fraction` of every class held out."""
    g = torch.Generator().manual_seed(seed); tr, va = [], []
    for c in torch.unique(labels):
        idx = torch.nonzero(labels == c).flatten(); idx = idx[torch.randperm(len(idx), generator=g)]
        n = max(1, int(round(fraction * len(idx)))) if len(idx) > 1 else 0
        va.append(idx[:n]); tr.append(idx[n:])
    return torch.cat(tr), torch.cat(va)


def parameter_groups(model, proxy_loss):
    groups = {"radius": ([], 2e-3, 0.0), "color": ([], 1e-3, 1e-4), "angles": ([], 1e-3, 0.0),
              "threshold": ([], 5e-4, 0.0), "quantization": ([], 5e-4, 0.0)}
    for b, br in model.branches.items():
        groups["radius"][0].append(br.radius_param); groups["angles"][0].append(br.angle_logits)
        groups["threshold"][0].append(br.threshold); groups["color"][0].append(model.color[b].weight)
        if br.input_quantization:
            groups["quantization"][0].append(br.quantization_param)
    out = [{"params": [p for p in ps if p.requires_grad], "lr": lr, "weight_decay": wd, "name": n}
           for n, (ps, lr, wd) in groups.items()]
    out.append({"params": list(proxy_loss.parameters()), "lr": 1e-3, "weight_decay": 1e-4, "name": "proxies"})
    return [g for g in out if g["params"]]


def fit(model, images: torch.Tensor, labels: torch.Tensor, epochs: int = 8, batch_size: int = 8, seed: int = 42,
        holdout: float = 0.15, min_selection_epoch: int = 2, device=None, verbose: bool = True):
    """Train `model` (a FLEXLBP layer) on images (N, 3, H, W) in [0, 1] and integer labels; returns the model with
    the checkpoint of best validation 1-NN accuracy, and the per-epoch history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = model.to(device); labels = labels.long()
    tr, va = stratified_holdout(labels, holdout, seed)
    proxy = ProxyCosineLoss(int(labels.max()) + 1, model.dim).to(device)
    opt = torch.optim.Adam(parameter_groups(model, proxy))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_AFTER.get(epochs, max(1, round(0.6 * epochs))), gamma=0.5)
    g = torch.Generator().manual_seed(seed)
    best, best_acc, history = copy.deepcopy(model.state_dict()), -1.0, []
    for epoch in range(epochs):
        model.train(); order = tr[torch.randperm(len(tr), generator=g)]; total = 0.0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            x, y = images[idx].to(device), labels[idx].to(device)
            loss = flex_loss(model, proxy, model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss) * len(idx)
        sched.step()
        model.eval()
        if len(va):
            pred = standardized_1nn(model.describe(images[tr]), labels[tr], model.describe(images[va]))
            acc = 100.0 * float((pred == labels[va]).float().mean())
        else:
            acc = float(epoch)
        history.append({"epoch": epoch + 1, "loss": total / len(tr), "val_1nn": acc})
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}: loss {total / len(tr):.4f}, validation 1-NN {acc:.2f}%")
        if acc > best_acc and (epoch + 1 >= min_selection_epoch or epoch + 1 == epochs):
            best_acc, best = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, history
