"""Train FLEX-LBP on a folder of images and evaluate it by standardized 1-NN.

    pip install .        # once, from the repository root (or run with PYTHONPATH=.)
    python examples/train_folder.py --train data/train --test data/test --learn geometry --template wide

Each folder holds one subfolder per class; images are converted to RGB and resized to --size x --size. The trained
layer is saved with its settings; reload it with
    ck = torch.load(path); layer = FLEXLBP(template=ck["template"], learn=ck["learn"]); layer.load_state_dict(ck["state_dict"])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flexlbp import FLEXLBP, postprocess, standardized_1nn
from flexlbp.train import fit

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm")


def load_folder(root, size, classes=None):
    """Images (N, 3, size, size) in [0, 1], labels, class names (sorted subfolder names unless given)."""
    root = Path(root)
    classes = classes or sorted(d.name for d in root.iterdir() if d.is_dir())
    xs, ys = [], []
    for c, name in enumerate(classes):
        for f in sorted((root / name).iterdir()):
            if f.suffix.lower() in EXTENSIONS:
                img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
                xs.append(torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)); ys.append(c)
    return torch.stack(xs), torch.tensor(ys), classes


def main():
    ap = argparse.ArgumentParser(description="Train FLEX-LBP on a folder of images and evaluate it by standardized 1-NN.")
    ap.add_argument("--train", required=True, help="training folder (one subfolder per class)")
    ap.add_argument("--test", required=True, help="test folder (same class subfolders)")
    ap.add_argument("--learn", default="geometry", help="preset: fixed, color, radius, threshold, geometry, mixer_q, all_joint")
    ap.add_argument("--template", default="wide", help="radius template: wide or compact")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="flexlbp_model.pt")
    a = ap.parse_args()

    xtr, ytr, classes = load_folder(a.train, a.size)
    xte, yte, _ = load_folder(a.test, a.size, classes)
    print(f"{len(classes)} classes, {len(xtr)} training and {len(xte)} test images")
    torch.manual_seed(a.seed)
    layer = FLEXLBP(template=a.template, learn=a.learn)
    if a.learn != "fixed":
        layer, _ = fit(layer, xtr, ytr, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed)
    layer.eval()
    ftr, fte = postprocess(layer.describe(xtr).double()), postprocess(layer.describe(xte).double())
    acc = 100.0 * float((standardized_1nn(ftr, ytr, fte) == yte).double().mean())
    print(f"test accuracy (standardized 1-NN): {acc:.2f}%")
    torch.save({"state_dict": layer.state_dict(), "template": a.template, "learn": a.learn, "classes": classes}, a.out)
    print(json.dumps(layer.parameters_summary(), indent=1))


if __name__ == "__main__":
    main()
