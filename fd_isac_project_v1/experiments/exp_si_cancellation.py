"""Experiment: SI cancellation  --  reproduces Figs. 3 and 4.
Fig 3: residual SI power (re. noise floor) vs SI power, for varrho in {1, 10, 100, 1e3, inf} and NSP.
Fig 4: SoI/SI power ratio vs SI power.  Averages are taken in dB over Monte-Carlo trials.
Run:  python experiments/exp_si_cancellation.py [--mc 30] [--show]
"""
from common import *

VARRHOS = (1, 10, 100, 1e3, np.inf)


def main():
    args = parse_args(default_mc=20, description=__doc__)
    plt = get_plt()
    cfg = Cfg(seed=args.seed)
    si_list = np.array(cfg.si_sweep_dB, dtype=float)
    names = [label_varrho(v) for v in VARRHOS] + ["NSP"]
    res = np.zeros((len(names), len(si_list), args.mc))
    soi = np.zeros_like(res)
    for i, si in enumerate(si_list):
        for t in range(args.mc):
            rng = np.random.default_rng(10_000 * (i + 1) + t)
            ch = draw_all(cfg, si, rng)                                 # same channel for all schemes
            for k, vr in enumerate(VARRHOS):
                m = run_proposed(ch, cfg, vr, rng)["metrics"]
                res[k, i, t], soi[k, i, t] = m["res_SI_dB"], m["soi_si_dB"]
            m = run_nsp(ch, cfg)["metrics"]
            res[-1, i, t], soi[-1, i, t] = m["res_SI_dB"], m["soi_si_dB"]
        print(f"SI = {si:4.0f} dB done")
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, "si_cancellation.npz"), si=si_list, res=res, soi=soi, names=names)

    for data, ylabel, fname, ttl in ((res, "Residual SI [dB re. noise floor]", "fig3_residual_SI.png", "Fig. 3"),
                                      (soi, "SoI over SI power [dB]", "fig4_SoI_over_SI.png", "Fig. 4")):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for k, nm in enumerate(names):
            ax.plot(si_list, data[k].mean(axis=1), marker="o", ms=3, ls="--" if nm == "NSP" else "-", label=nm)
        ax.set_xlabel("SI power [dB]"); ax.set_ylabel(ylabel); ax.set_title(f"{ttl} (mean of dB over {args.mc} trials)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, fname)
    if args.show:
        plt.show()
    print("\nMean residual SI [dB] at SI = 60 dB:", {n: round(float(res[k, -1].mean()), 1) for k, n in enumerate(names)})


if __name__ == "__main__":
    main()
