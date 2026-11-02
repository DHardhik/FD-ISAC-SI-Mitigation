"""Experiment: convergence  --  reproduces Fig. 2.   (v3)
(a) objective (23) vs iteration, (b) relative difference (51) vs iteration,
for (Nt = 16, SI = 60 dB) and (Nt = 8, SI = 20 dB), varrho = 1. Single start (plain Algorithm 1) so the curve is
that of one BCD run. v2: the first two iterations are not plotted in (a): they still contain the huge SI penalty of
the random starting point (about -1e13) which made v1's panel look flat.
Run:  python experiments/exp_convergence.py [--mc 50] [--show]
"""
from common import *

CASES = [(16, 60, "Nt=16, SI = 60 dB"), (8, 20, "Nt=8, SI = 20 dB")]
SKIP = 2


def main():
    args = parse_args(default_mc=50, description=__doc__)
    plt = get_plt()
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.5))
    for (Nt, si, lab), mk in zip(CASES, ("o", "s")):
        cfg = make_cfg(args, Nt=Nt, Nr=Nt)
        rng = np.random.default_rng(args.seed)
        ch = draw_all(cfg, si, rng)
        out = solve_fd_isac(ch, cfg, a=alphas(1.0, cfg, ch, args), rng=rng, beta=args.beta)
        f, z = out["history"], out["zeta"]
        ax[0].plot(np.arange(SKIP, len(f)), f[SKIP:], marker=mk, ms=4, label=lab)
        ax[1].semilogy(np.arange(1, len(z) + 1), np.maximum(z, 1e-16), marker=mk, ms=4, label=lab)
        print(f"{lab}: {out['n_iter']} iterations, converged={out['converged']}, "
              f"monotone={bool(np.all(np.diff(f) >= -1e-9 * np.abs(f[:-1])))}")
        its = []
        for t in range(args.mc):
            r = np.random.default_rng(1000 + t)
            chm = draw_all(cfg, si, r)
            its.append(solve_fd_isac(chm, cfg, a=alphas(1.0, cfg, chm, args), rng=r, beta=args.beta)["n_iter"])
        print(f"   mean iterations over {args.mc} trials: {np.mean(its):.1f}  (min {min(its)}, max {max(its)})")
    ax[0].set_ylabel("objective (23)"); ax[0].set_xlabel("number of iterations"); ax[0].legend(); ax[0].set_title("(a)")
    ax[1].axhline(Cfg().tol, color="k", ls="--", lw=0.8, label="tol")
    ax[1].set_ylabel("relative difference $\\zeta$"); ax[1].set_xlabel("number of iterations"); ax[1].legend(); ax[1].set_title("(b)")
    save(fig, "fig2_convergence.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
