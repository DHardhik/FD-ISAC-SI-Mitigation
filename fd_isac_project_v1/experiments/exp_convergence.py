"""Experiment: convergence  --  reproduces Fig. 2.
(a) objective (23) vs iteration, (b) relative difference (51) vs iteration,
for (Nt = 16, SI = 60 dB) and (Nt = 8, SI = 20 dB), varrho = 1.
Run:  python experiments/exp_convergence.py [--mc 50] [--show]
"""
from common import *

CASES = [(16, 60, "Nt=16, SI = 60 dB"), (8, 20, "Nt=8, SI = 20 dB")]


def main():
    args = parse_args(default_mc=50, description=__doc__)
    plt = get_plt()
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.5))
    for (Nt, si, lab), mk in zip(CASES, ("o", "s")):
        cfg = Cfg(Nt=Nt, Nr=Nt, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        out = solve_fd_isac(draw_all(cfg, si, rng), cfg, a=alphas_from_varrho(1.0), rng=rng)
        f, z = out["history"], out["zeta"]
        # f[0] is the random starting point (hugely negative because of the 1/(2*beta) penalty) -> plot from iteration 1
        ax[0].plot(np.arange(1, len(f)), f[1:], marker=mk, ms=4, label=lab)
        ax[1].semilogy(np.arange(1, len(z) + 1), np.maximum(z, 1e-16), marker=mk, ms=4, label=lab)
        print(f"{lab}: {out['n_iter']} iterations, converged={out['converged']}, "
              f"monotone={bool(np.all(np.diff(f) >= -1e-9 * np.abs(f[:-1])))}")

        # average number of iterations over Monte-Carlo trials
        its = []
        for t in range(args.mc):
            r = np.random.default_rng(1000 + t)
            its.append(solve_fd_isac(draw_all(cfg, si, r), cfg, a=alphas_from_varrho(1.0), rng=r)["n_iter"])
        print(f"   mean iterations over {args.mc} trials: {np.mean(its):.1f}  (min {min(its)}, max {max(its)})")
    ax[0].set_ylabel("objective (23)"); ax[0].set_xlabel("number of iterations"); ax[0].legend(); ax[0].set_title("(a)")
    ax[1].axhline(Cfg().tol, color="k", ls="--", lw=0.8, label="tol")
    ax[1].set_ylabel("relative difference $\\zeta$"); ax[1].set_xlabel("number of iterations"); ax[1].legend(); ax[1].set_title("(b)")
    save(fig, "fig2_convergence.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
