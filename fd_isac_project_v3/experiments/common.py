"""Shared helpers for the experiment scripts (not an experiment itself). v2."""
import argparse
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                    # project root -> i_config, ii_channels, ...

import numpy as np
import matplotlib
from i_config import Cfg
from ii_channels import draw_all
from v_algorithm import solve_fd_isac, solve_multistart, alphas_from_varrho
from vii_baselines import nsp_design, complete_baseline

RESULTS = os.path.join(ROOT, "results")
TAG = ""


def parse_args(default_mc=20, description=""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--mc", type=int, default=default_mc,
                    help=f"Monte-Carlo trials (paper: 1000; default {default_mc} to keep run time short)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open plot windows instead of only saving PNGs")
    # ---- v2/v3 options (defaults are the v3 calibrated set-up) ----
    ap.add_argument("--starts", type=int, default=1, help="number of RANDOM starts of the BCD (+ 2 warm starts)")
    ap.add_argument("--no-warm", action="store_true", help="disable the radar-only / comm-only warm starts")
    ap.add_argument("--beta", type=float, default=None, help="penalty parameter (default: 1e-11 on the normalised SI channel for --solver v3, 1e-25 otherwise)")
    ap.add_argument("--loss", type=float, default=20.0,
                    help="extra path loss [dB] on the user links (v3 default 20 -> half-duplex ~10 bps/Hz; v2 used 0)")
    ap.add_argument("--echo", type=float, default=-32.0,
                    help="radar echo per RX antenna re. noise [dB] (v3 default -32); use --echo nan for the RCS formula")
    ap.add_argument("--kappa", type=float, default=16.0, help="radar-weight scale (v3 default 16 = alpha_radar 1 per unit-power beam)")
    ap.add_argument("--radar-weights", choices=["v3", "v2", "raw"], default="v3",
                    help="v3: kappa-scaled unit-power gains; v2: v2's --norm-radar; raw: alpha_radar = 1 on mW gains (v1)")
    ap.add_argument("--norm-radar", action="store_true", help="(v2 compatibility) same as --radar-weights v2")
    ap.add_argument("--solver", choices=["v3", "v2", "paper"], default="v3",
                    help="v3: exact solve, normalised SI penalty (default); v2: exact projection; paper: dense solve of Eqs (34),(43)")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel worker processes (exp_si_sweep)")
    ap.add_argument("--tag", type=str, default="", help="suffix for output file names, e.g. --tag test")
    args, _ = ap.parse_known_args()
    global TAG
    TAG = args.tag
    if not args.show:
        matplotlib.use("Agg")
    return args


def make_cfg(args, **kw):
    echo = None if np.isnan(args.echo) else args.echo
    return Cfg(seed=args.seed, comm_extra_loss_dB=args.loss, max_iter=args.max_iter, echo_dB=echo,
               kappa_radar=args.kappa, solver=args.solver, **kw)


def get_plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    return plt


def save(fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    if TAG:
        root, ext = os.path.splitext(name)
        name = f"{root}_{TAG}{ext}"
    path = os.path.join(RESULTS, name)
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)
    return path


def label_varrho(vr):
    return "varrho = inf (comm only)" if np.isinf(vr) else f"varrho = {vr:g}"


def radar_mode(args):
    if args is None:
        return "v3"
    return "v2" if args.norm_radar else args.radar_weights


def alphas(varrho, cfg, ch, args=None):
    mode = radar_mode(args)
    norm = {"v3": "v3", "v2": True, "raw": False}[mode]
    return alphas_from_varrho(varrho, cfg, normalize_radar=norm, ch=ch)


def run_proposed(ch, cfg, varrho, rng, args=None):
    """Proposed design (Algorithm 1) with multi-start / warm start."""
    n_random = 1 if args is None else args.starts
    warm = True if args is None else (not args.no_warm)
    beta = None if args is None else args.beta
    return solve_multistart(ch, cfg, a=alphas(varrho, cfg, ch, args), rng=rng, n_random=n_random, warm=warm, beta=beta)


def run_nsp(ch, cfg, power_split=0.5, null_mode="channel"):
    return complete_baseline(ch, cfg, *nsp_design(ch, cfg, power_split, null_mode=null_mode))


def robust_stats(x, axis=-1):
    """median and 25/75 percentiles -- v2 uses these instead of the mean of dB values (outlier-sensitive)."""
    return np.median(x, axis=axis), np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)
