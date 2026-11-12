"""Reproduction of the paper's research implementation (optional: needs files that are not part of this package).

Trained research checkpoints are converted with flexlbp.convert; descriptors and parameter gradients are compared with
the research model on the same images, and one reported accuracy is recomputed from the images (NewBarkTex, Geometry,
seed 42: 84.19% with u2 + power normalization and standardized 1-NN).

The research files are located by environment variables:
    FLEXLBP_RESEARCH_ROOT  the research repository: train_multibranch.py and the run directories listed in RUNS
    FLEXLBP_IMAGE_CACHE    the cached benchmark images NB.npz, USP.npz, KTH.npz and TC32.npz
                           (default: $FLEXLBP_RESEARCH_ROOT/reports/lbp_family_baselines/cache)
The tests also need PyYAML and NumPy. When anything is missing, they are skipped with the reason.
"""
import copy
import os
import sys
import unittest
from pathlib import Path

import torch

import flexlbp
from flexlbp.convert import from_research_state

try:
    import numpy as np
    import yaml
    MISSING_MODULES = ""
except ImportError as error:  # optional dependencies of these tests only
    MISSING_MODULES = str(error)


def _env_path(name):
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _exists(path, kind="any"):
    """Whether path exists (as a file or directory if asked); unreadable locations count as absent."""
    try:
        return path.is_file() if kind == "file" else path.is_dir() if kind == "dir" else path.exists()
    except OSError:
        return False


ROOT = _env_path("FLEXLBP_RESEARCH_ROOT")
CACHE = _env_path("FLEXLBP_IMAGE_CACHE") or (ROOT / "reports" / "lbp_family_baselines" / "cache" if ROOT else None)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUPS = ("color", "radius", "angles", "threshold", "quantization")
RUNS = {  # name: (run directory relative to FLEXLBP_RESEARCH_ROOT, cached images, support size override)
    "NB geometry s42": ("results_controlled_accuracy_push/cap_proxy_newbark_geometry", "NB", None),
    "USP all joint": ("results_controlled_accuracy_push_e20/cap20_proxy_usptex_all_joint", "USP", None),
    "USP mixer+Q": ("results_controlled_accuracy_push_e20/cap20_proxy_usptex_mixer_plus_Q_probe", "USP", None),
    "KTH fixed": ("results_controlled_accuracy_push/cap_proxy_kth_sa_fixed", "KTH", None),
    "TC32 geometry bit-flip": ("results_v11_night/cap_proxy_outex32_geometry_bitflip_s42", "TC32", None),
    "TC32 geometry median support": ("results_controlled_accuracy_push/cap_proxy_outex32_geometry", "TC32", 3),
}


def skip_reason():
    """Why the reproduction tests cannot run here (empty string when they can)."""
    if MISSING_MODULES:
        return f"the reproduction tests need PyYAML and NumPy ({MISSING_MODULES})"
    if ROOT is None:
        return ("research checkpoints not available: set FLEXLBP_RESEARCH_ROOT to the research repository "
                "(and FLEXLBP_IMAGE_CACHE to the cached images) to run the reproduction tests")
    if not _exists(ROOT / "train_multibranch.py", "file"):
        return f"FLEXLBP_RESEARCH_ROOT does not contain the research code (train_multibranch.py): {ROOT}"
    if not _exists(CACHE, "dir"):
        return f"cached benchmark images not found: {CACHE} (set FLEXLBP_IMAGE_CACHE)"
    return ""


SKIP_REASON = skip_reason()


def checkpoint(run_dir):
    return run_dir / ("best_model.pth" if _exists(run_dir / "best_model.pth") else "final_model.pth")


def missing_files(run, cache):
    """The files a comparison needs that are absent."""
    run_dir = ROOT / run
    out = [str(p) for p in (run_dir / "config.yaml", CACHE / f"{cache}.npz") if not _exists(p)]
    if not _exists(checkpoint(run_dir)):
        out.insert(1, str(run_dir / "best_model.pth") + " (or final_model.pth)")
    return out


def research_pair(run, support=None):
    """(research model, FLEXLBP layer) with the same trained parameters."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from train_multibranch import build_model
    cfg = yaml.safe_load((ROOT / run / "config.yaml").read_text())
    if support:
        cfg = copy.deepcopy(cfg)
        for b in ("fine", "mid", "coarse"):
            cfg["model"][f"branch_{b}"]["support_size"] = support
    sd = torch.load(checkpoint(ROOT / run), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd); sd = sd.get("model_state_dict", sd)
    ref = build_model(cfg, DEV); missing = ref.load_state_dict(sd, strict=False).missing_keys
    assert not missing, missing
    return ref.eval(), from_research_state(cfg, sd, learn=GROUPS).to(DEV).eval()


def cached_images(name, split="xte", n=None):
    z = np.load(CACHE / f"{name}.npz")
    x = torch.from_numpy(z[split][:n]).permute(0, 3, 1, 2).float().div(255.0)
    return x, torch.from_numpy(z["y" + split[1:]][:n])


def research_gradients(model, w, x):
    params = {}
    for b in ("fine", "mid", "coarse"):
        br = getattr(model, f"branch_{b}")
        params.update({f"branches.{b}.radius_param": br.radius_param, f"branches.{b}.angle_logits": br.angle_increments,
                       f"branches.{b}.threshold": br.threshold_k, f"color.{b}.weight": getattr(model, f"mixer_{b}").mixer})
        if hasattr(br, "quantization_param"):
            params[f"branches.{b}.quantization_param"] = br.quantization_param
    for p in params.values():
        p.requires_grad_(True); p.grad = None
    (model.extract_descriptor(x, normalize=True) * w).sum().backward()
    return {k: p.grad.detach().clone() for k, p in params.items()}


@unittest.skipIf(bool(SKIP_REASON), SKIP_REASON)
class TestReproduction(unittest.TestCase):
    def test_descriptors_and_gradients(self):
        w = torch.randn(13824, generator=torch.Generator().manual_seed(0)).to(DEV)
        for name, (run, cache, support) in RUNS.items():
            with self.subTest(name):
                absent = missing_files(run, cache)
                if absent:
                    self.skipTest(f"{name}: research files not found: {', '.join(absent)}")
                ref, layer = research_pair(run, support)
                x = cached_images(cache, n=4)[0].to(DEV)
                with torch.no_grad():
                    diff = float((ref.extract_descriptor(x, normalize=True) - layer(x)).abs().max())
                self.assertLess(diff, 1e-6, f"{name}: descriptor max |diff| = {diff:.2e}")
                g_ref = research_gradients(ref, w, x)
                (layer(x) * w).sum().backward()
                for k, g in g_ref.items():
                    g_new = dict(layer.named_parameters())[k].grad
                    rel = float((g_new - g).abs().max() / g.abs().max().clamp_min(1e-12))
                    self.assertLess(rel, 1e-4, f"{name}: gradient of {k}, relative max |diff| = {rel:.2e}")
                print(f"  {name}: descriptor max |diff| {diff:.1e}; gradients of {len(g_ref)} tensors match")

    def test_reported_accuracy(self):
        run, cache, _ = RUNS["NB geometry s42"]
        absent = missing_files(run, cache)
        if absent:
            self.skipTest(f"research files not found: {', '.join(absent)}")
        _, layer = research_pair(run)
        (xtr, ytr), (xte, yte) = cached_images(cache, "xtr"), cached_images(cache, "xte")
        ftr = flexlbp.postprocess(layer.describe(xtr).double()); fte = flexlbp.postprocess(layer.describe(xte).double())
        acc = 100.0 * float((flexlbp.standardized_1nn(ftr, ytr, fte) == yte).double().mean())
        print(f"  NewBarkTex Geometry seed 42: {acc:.2f}% (paper: 84.19%)")
        self.assertEqual(round(acc, 2), 84.19)


if __name__ == "__main__":
    unittest.main()
