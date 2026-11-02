"""Experiment: sensing  --  reproduces Figs. 8 and 9.
Fig 8: range-velocity maps (proposed, NSP, radar-only, communication-only), 60 dB SI, Nt = Nr = 16,
       M = 512 blocks, N = 1024 symbols, varrho = 1.
Fig 9: normalised angle spectrum P(theta) of the four methods.
Also prints the success rate of range-velocity detection over Monte-Carlo trials.
v2: zero-padded Doppler FFT (no more 19.07 vs 21.46 m/s bin split), multi-start BCD.
v3: calibrated echo + conjugate-steering fix -> the proposed design no longer nulls the target.
Run:  python experiments/exp_sensing.py [--mc 20] [--show]
"""
from common import *
from vi_sensing import (simulate_rx_stream, range_doppler_map, estimate_peak, to_velocity_positive,
                        angle_spectrum, normalise_dB)
from vii_baselines import radar_only, comm_only

N_RANGE = 10        # range bins shown (0 .. 67.5 m, bin = 7.5 m)


def designs(ch, cfg, rng, args=None):
    prop = run_proposed(ch, cfg, 1.0, rng, args)
    return {"Proposed method": prop,
            "NSP method": run_nsp(ch, cfg),
            "Radar-only method": complete_baseline(ch, cfg, *radar_only(cfg)),
            "Communication-only method": complete_baseline(ch, cfg, *comm_only(cfg))}


def main():
    args = parse_args(default_mc=20, description=__doc__)
    plt = get_plt()
    cfg = make_cfg(args)
    rng = np.random.default_rng(args.seed + 1)
    ch = draw_all(cfg, 60, rng)
    ds = designs(ch, cfg, np.random.default_rng(args.seed + 2), args)
    i_true = int(round(2 * cfg.target[1] / (3e8 * cfg.Ts)))

    # ---------------- Fig 8: range-velocity maps ----------------
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.6), constrained_layout=True)
    for ax, (name, d) in zip(axs.ravel(), ds.items()):
        s, sd = simulate_rx_stream(ch, d["w"], d["p"], d["wu"], cfg, rng=np.random.default_rng(args.seed + 3))
        P, r_ax, v_ax = range_doppler_map(s, sd, cfg, n_range=N_RANGE)
        r_hat, v_hat, _ = estimate_peak(P, r_ax, v_ax)
        Pv, v_pos = to_velocity_positive(P, v_ax, v_max=60.0)
        im = ax.pcolormesh(r_ax, v_pos, 10 * np.log10(Pv + 1e-30), shading="nearest", cmap="viridis")
        ax.plot([r_hat], [v_hat], "ys", ms=7, mfc="none")
        ax.set_title(f"{name}\npeak: r = {r_hat:.1f} m, v = {v_hat:.2f} m/s", fontsize=9)
        ax.set_xlabel("Range [m]"); ax.set_ylabel("Velocity [m/s]")
        fig.colorbar(im, ax=ax, label="dB")
        print(f"{name:27s} peak at r = {r_hat:5.1f} m, v = {v_hat:6.2f} m/s")
    fig.suptitle(f"Fig. 8: true target r = {cfg.target[1]} m, v = {cfg.target[2]} m/s (SI = 60 dB)")
    save(fig, "fig8_range_velocity.png")

    # ---------------- Fig 9: angle spectrum ----------------
    th = np.linspace(-90, 90, 721)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, d), c in zip(ds.items(), ("tab:red", "tab:purple", "k", "tab:blue")):
        ax.plot(th, normalise_dB(angle_spectrum(d["w"], ch["Hsi"], d["p"], th, cfg.Nr)), color=c, lw=1, label=name)
    ax.axvline(cfg.target[0], color="r", ls=":", label=f"Radar target at {cfg.target[0]:g}°")
    ax.axvline(cfg.ul_user[0], color="k", ls="--", label=f"UL user at {cfg.ul_user[0]:g}°")
    ax.set_ylim(-40, 3); ax.set_xlabel("theta [deg]"); ax.set_ylabel("P(theta) [dB]")
    ax.set_title("Fig. 9: angle estimation (SI = 60 dB)"); ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig9_angle.png")

    # ---------------- Monte-Carlo detection rate ----------------
    hits = {k: 0 for k in ds}
    ang_ok = {k: 0 for k in ds}
    for t in range(args.mc):
        r = np.random.default_rng(30_000 + t)
        chm = draw_all(cfg, 60, r)
        for name, d in designs(chm, cfg, r, args).items():
            s, sd = simulate_rx_stream(chm, d["w"], d["p"], d["wu"], cfg, rng=r)
            P, r_ax, v_ax = range_doppler_map(s, sd, cfg, n_range=N_RANGE)
            r_hat, v_hat, _ = estimate_peak(P, r_ax, v_ax)
            hits[name] += int(abs(r_hat - cfg.target[1]) < 1e-6 and abs(v_hat - cfg.target[2]) < 1.5)
            ang = th[np.argmax(angle_spectrum(d["w"], chm["Hsi"], d["p"], th, cfg.Nr))]
            ang_ok[name] += int(abs(ang - cfg.target[0]) < 2.0)
    print(f"\nDetection rate over {args.mc} trials (peak at true range bin and |v err| < 1.5 m/s | AoA within 2 deg):")
    for k in ds:
        print(f"  {k:27s} range-velocity {hits[k] / args.mc:5.2f}   angle {ang_ok[k] / args.mc:5.2f}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
