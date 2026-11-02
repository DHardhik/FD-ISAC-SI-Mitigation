"""Experiment: beampatterns  --  reproduces Fig. 5.   (v3)
TX beampattern |b(theta)^H p|^2 (left) and RX beampattern |w^H a(theta)|^2 (right) of the optimised design (red)
vs the radar-only pattern (blue) for varrho = 1, 100, 0.01 at 60 dB SI (Nt = Nr = 16).
v2: patterns are in dB RELATIVE to the radar-only peak (0 dB), since the paper does not state its normalisation;
use --absolute for the v1 scale. Compare shapes/lobes.
Run:  python experiments/exp_beampattern.py [--loss 15] [--absolute] [--show]
"""
from common import *
from ii_channels import steering
from vii_baselines import radar_only


def main():
    args = parse_args(default_mc=1, description=__doc__)
    absolute = "--absolute" in sys.argv
    plt = get_plt()
    cfg = make_cfg(args)
    ch = draw_all(cfg, 60, np.random.default_rng(args.seed + 1))
    th = np.linspace(-100, 100, 1001)
    B = np.stack([steering(t, cfg.Nt) for t in th])
    A = np.stack([steering(t, cfg.Nr) for t in th])
    p_r, w_r = radar_only(cfg)
    tx_r = 10 * np.log10(np.abs(B.conj() @ p_r) ** 2 + 1e-12)
    rx_r = 10 * np.log10(np.abs(A.conj() @ w_r) ** 2 + 1e-12)
    off_tx, off_rx = (0.0, 0.0) if absolute else (tx_r.max(), rx_r.max())

    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
    for row, vr in enumerate((1.0, 100.0, 0.01)):
        out = run_proposed(ch, cfg, vr, np.random.default_rng(args.seed + 2), args)
        p, w = out["p"], out["w"]
        tx = 10 * np.log10(np.abs(B.conj() @ p) ** 2 + 1e-12)
        rx = 10 * np.log10(np.abs(A.conj() @ w) ** 2 + 1e-12)
        i_d = int(np.argmin(np.abs(th - cfg.dl_user[0]))); i_u = int(np.argmin(np.abs(th - cfg.ul_user[0])))
        print(f"varrho = {vr:g}: residual SI = {out['metrics']['res_SI_dB']:.1f} dB | TX level at DL user "
              f"{tx[i_d] - tx.max():.1f} dB below peak (radar-only: {tx_r[i_d] - tx_r.max():.1f}) | "
              f"RX level at UL user {rx[i_u] - rx.max():.1f} dB below peak (radar-only: {rx_r[i_u] - rx_r.max():.1f})")
        for col, (opt, ref, ttl, user, off) in enumerate(((tx, tx_r, "TX", cfg.dl_user[0], off_tx),
                                                          (rx, rx_r, "RX", cfg.ul_user[0], off_rx))):
            ax = axs[row, col]
            ax.plot(th, ref - off, color="tab:blue", lw=1, label="Radar-only")
            ax.plot(th, opt - off, color="tab:red", lw=1, label=f"Optimized {ttl} beampattern")
            ax.axvline(cfg.target[0], color="k", ls="--", lw=0.8, label=f"Target at {cfg.target[0]:g}°")
            ax.axvline(user, color="tab:red", ls="-.", lw=0.8, label=f"{'DL' if ttl == 'TX' else 'UL'} user at {user:g}°")
            ax.set_ylim(-40, (ref - off).max() + 5); ax.set_title(f"{ttl}: varrho = {vr:g}", fontsize=9)
            ax.set_ylabel(f"{ttl} beampattern [dB{'' if absolute else ' rel. to radar-only peak'}]", fontsize=8)
            if row == 0:
                ax.legend(fontsize=6, loc="lower right")
    for ax in axs[-1]:
        ax.set_xlabel("Azimuth angle [deg]")
    save(fig, "fig5_beampatterns.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
