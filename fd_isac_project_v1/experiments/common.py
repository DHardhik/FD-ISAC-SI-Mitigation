"""Shared helpers for the experiment scripts (not an experiment itself)."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                    # project root -> i_config, ii_channels, ...

import numpy as np
import matplotlib
from i_config import Cfg
from ii_channels import draw_all
from v_algorithm import solve_fd_isac, alphas_from_varrho
from vii_baselines import nsp_design, complete_baseline

RESULTS = os.path.join(ROOT, "results")


def parse_args(default_mc=20, description=""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--mc", type=int, default=default_mc,
                    help=f"Monte-Carlo trials (paper: 1000; default {default_mc} to keep run time short)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open plot windows instead of only saving PNGs")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    return args


def get_plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    return plt


def save(fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)
    return path


def label_varrho(vr):
    return "varrho = inf (comm only)" if np.isinf(vr) else f"varrho = {vr:g}"


def run_proposed(ch, cfg, varrho, rng):
    """Proposed design; the comm-only bound (varrho = inf) gets a larger iteration cap (slow BCD, see notes)."""
    c = Cfg(**{**cfg.__dict__, "max_iter": 200}) if np.isinf(varrho) else cfg
    return solve_fd_isac(ch, c, a=alphas_from_varrho(varrho), rng=rng)


def run_nsp(ch, cfg, power_split=0.5):
    return complete_baseline(ch, cfg, *nsp_design(ch, cfg, power_split))
