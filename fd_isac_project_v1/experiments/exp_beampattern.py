"""Experiment: beampatterns  --  reproduces Fig. 5.
TX beampattern 10log10|b(theta)^H p|^2 (left) and RX beampattern 10log10|w^H a(theta)|^2 (right) of the optimised
design (red) vs the radar-only pattern (blue) for varrho = 1, 100, 0.01 at 60 dB SI (Nt = Nr = 16).
Absolute dB scale differs from the paper's (normalisation not stated); compare the SHAPE and the lobes.
Run:  python experiments/exp_beampattern.py [--show]
"""
from common import *
from ii_channels import steering
from vii_baselines import radar_only


def main():
    args = parse_args(default_mc=1, description=__doc__)
    plt = get_plt()
    cfg = Cfg(seed=args.seed)
    ch = draw_all(cfg, 60, np.random.default_rng(args.seed + 1))
    th = np.linspace(-100, 100, 1001)
    B = np.stack([steering(t, cfg.Nt) for t in th])            # [T x Nt]
    A = np.stack([steering(t, cfg.Nr) for t in th])
    p_r, w_r = radar_only(cfg)
    tx_r = 10 * np.log10(np.abs(B.conj() @ p_r) ** 2 + 1e-12)
    rx_r = 10 * np.log10(np.abs(A.conj() @ w_r) ** 2 + 1e-12)

    cases = [(1.0, "(a),(d)"), (100.0, "(b),(e)"), (0.01, "(c),(f)")]
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
    for row, (vr, tag) in enumerate(cases):
        out = run_proposed(ch, cfg, vr, np.random.default_rng(args.seed + 2))
        p, w = out["p"], out["w"]
        tx = 10 * np.log10(np.abs(B.conj() @ p) ** 2 + 1e-12)
        rx = 10 * np.log10(np.abs(A.conj() @ w) ** 2 + 1e-12)
        print(f"varrho = {vr:g}: residual SI = {out['metrics']['res_SI_dB']:.2f} dB  (iterations {out['n_iter']})")
        for col, (opt, ref, ttl, user) in enumerate(((tx, tx_r, "TX", cfg.dl_user[0]), (rx, rx_r, "RX", cfg.ul_user[0]))):
            ax = axs[row, col]
            ax.plot(th, ref, color="tab:blue", lw=1, label="Radar-only")
            ax.plot(th, opt, color="tab:red", lw=1, label=f"Optimized {ttl} beampattern")
            ax.axvline(cfg.target[0], color="k", ls="--", lw=0.8, label=f"Target at {cfg.target[0]:g}°")
            ax.axvline(user, color="tab:red", ls="-.", lw=0.8, label=f"{'DL' if ttl == 'TX' else 'UL'} user at {user:g}°")
            ax.set_ylim(-40, ref.max() + 5); ax.set_title(f"{ttl}: varrho = {vr:g}", fontsize=9)
            ax.set_ylabel(f"{ttl} beampattern [dB]")
            if row == 0:
                ax.legend(fontsize=6, loc="lower right")
    for ax in axs[-1]:
        ax.set_xlabel("Azimuth angle [deg]")
    save(fig, "fig5_beampatterns.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
