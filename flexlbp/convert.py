"""Load checkpoints trained with the paper's research code into the FLEXLBP layer (for exact reproduction).

The research run directories contain config.yaml and best_model.pth (or final_model.pth). The branch parameters
map one to one: radius_param, angle_increments -> angle_logits, threshold_k -> threshold, quantization_param, and the
per-branch color mixers mixer_<branch>.mixer -> color.<branch>.weight.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .layer import BRANCHES, FLEXLBP

SHARED_KEYS = ("radius_directions", "radius_axis_temperature", "support_size", "hist_ste_mode", "enable_learnable_quantization",
               "init_quantization_levels", "quantization_levels_min", "quantization_levels_max")


def from_research_run(run_dir, learn="fixed", map_location="cpu") -> FLEXLBP:
    """Build a FLEXLBP layer with the bounds, operator settings and learned parameters of a research run directory."""
    import yaml

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ck = run_dir / "best_model.pth" if (run_dir / "best_model.pth").exists() else run_dir / "final_model.pth"
    sd = torch.load(ck, map_location=map_location, weights_only=False)
    sd = sd.get("state_dict", sd); sd = sd.get("model_state_dict", sd)
    return from_research_state(cfg, sd, learn=learn)


def from_research_state(cfg: dict, state: dict, learn="fixed") -> FLEXLBP:
    m = cfg["model"]
    bounds = {b: (float(m[f"branch_{b}"]["init_radius"]), float(m[f"branch_{b}"]["radius_min"]), float(m[f"branch_{b}"]["radius_max"]))
              for b in BRANCHES}
    first = m["branch_fine"]
    if not m.get("color_mixer", {}).get("per_branch", False) or m.get("branch_micro") or m.get("branch_macro"):
        raise ValueError("only the paper's three branches with one color transform each are supported")
    for b in BRANCHES:
        bc = m[f"branch_{b}"]
        if not bc.get("enable_magnitude_histogram", False) or bc.get("pattern_mode", "lbp") != "lbp" or bc.get("enable_center_histogram", False):
            raise ValueError("only the paper's sign-plus-magnitude operator is supported")
        if any(bc.get(k) != first.get(k) for k in SHARED_KEYS):
            raise ValueError("the branches must share the operator settings " + ", ".join(SHARED_KEYS))
    layer = FLEXLBP(template=bounds, learn=learn, K=int(m.get("K", 8)), radius_directions=int(first.get("radius_directions", 1)),
                    axis_temperature=float(first.get("radius_axis_temperature", 8.0)), tau=float(m.get("temperature", 10.0)),
                    support_size=int(first.get("support_size", 1)), histogram_estimator=first.get("hist_ste_mode", "adjacent"),
                    padding=cfg.get("border_handling", {}).get("padding_mode", "reflect"), input_quantization=bool(first.get("enable_learnable_quantization", False)),
                    quantization_init=float(first.get("init_quantization_levels", 256.0)),
                    quantization_range=(float(first.get("quantization_levels_min", 8.0)), float(first.get("quantization_levels_max", 256.0))))
    new = {}
    for b in BRANCHES:
        new[f"branches.{b}.radius_param"] = state[f"branch_{b}.radius_param"]
        new[f"branches.{b}.angle_logits"] = state[f"branch_{b}.angle_increments"]
        new[f"branches.{b}.threshold"] = state[f"branch_{b}.threshold_k"]
        if f"branch_{b}.quantization_param" in state:
            new[f"branches.{b}.quantization_param"] = state[f"branch_{b}.quantization_param"]
        new[f"color.{b}.weight"] = state[f"mixer_{b}.mixer"]
    missing, unexpected = layer.load_state_dict(new, strict=False)
    missing = [k for k in missing if not k.endswith(("basis_angles", "init_radii", "standard_angles"))]
    if missing or unexpected:
        raise RuntimeError(f"conversion mismatch: missing {missing}, unexpected {unexpected}")
    return layer
