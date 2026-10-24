"""Experiment: sum-rate  --  reproduces Fig. 7.
Average FD sum-rate Ru + Rd [bps/Hz] vs SI power for varrho in {1, 10, 100, 1e3, inf}, NSP, and the
half-duplex system (delta = 0.5; no SI, so its curve is flat).
Run:  python experiments/exp_sumrate.py [--mc 30] [--show]
"""
from common import *
from vii_baselines import hd_rates, half_duplex_sumrate

VARRHOS = (1, 10, 100, 1e3, np.inf)


def main():
    args = parse_args(default_mc=20, description=__doc__)
    plt = get_plt()
    cfg = Cfg(seed=args.seed)
    si_list = np.array(cfg.si_sweep_dB, dtype=float)
    names = [label_varrho(v) for v in VARRHOS] + ["NSP"]
    sr = np.zeros((len(names), len(si_list), args.mc))
    hd = np.zeros((len(si_list), args.mc))
    for i, si in enumerate(si_list):
        for t in range(args.mc):
            rng = np.random.default_rng(20_000 * (i + 1) + t)
            ch = draw_all(cfg, si, rng)
            for k, vr in enumerate(VARRHOS):
                m = run_proposed(ch, cfg, vr, rng)["metrics"]
                sr[k, i, t] = m["Ru"] + m["Rd"]
            m = run_nsp(ch, cfg)["metrics"]
            sr[-1, i, t] = m["Ru"] + m["Rd"]
            hd[i, t] = half_duplex_sumrate(*hd_rates(ch, cfg), delta=cfg.delta_hd)
        print(f"SI = {si:4.0f} dB done")
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, "sumrate.npz"), si=si_list, sr=sr, hd=hd, names=names)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for k, nm in enumerate(names):
        ax.plot(si_list, sr[k].mean(axis=1), marker="o", ms=3, ls="--" if nm == "NSP" else "-", label=nm)
    ax.plot(si_list, hd.mean(axis=1), color="k", marker="+", label=f"Half-duplex (delta={cfg.delta_hd:g})")
    ax.set_xlabel("SI power [dB]"); ax.set_ylabel("Average sum-rate [bps/Hz] per unit time-frequency resource")
    ax.set_title(f"Fig. 7 ({args.mc} trials)"); ax.legend(fontsize=7)
    save(fig, "fig7_sumrate.png")
    if args.show:
        plt.show()
    print("Mean sum-rate at SI = 60 dB:", {n: round(float(sr[k, -1].mean()), 2) for k, n in enumerate(names)},
          " half-duplex:", round(float(hd[-1].mean()), 2))


if __name__ == "__main__":
    main()
