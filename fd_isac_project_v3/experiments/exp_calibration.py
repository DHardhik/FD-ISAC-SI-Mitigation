"""Experiment: calibration sweep (added in v2; in v3 the chosen value, 20 dB, is the default `--loss`).
The paper does not give every parameter, and with our v1 assumptions the user links have a much higher SNR than the
paper's (half-duplex sum-rate ~16.6 bps/Hz here vs ~10 in Fig. 7; comm-only FD ~32 vs ~15). This script sweeps an
EXTRA path loss on the user links (`Cfg.comm_extra_loss_dB`) and reports, at SI = 60 dB:
  * half-duplex rate   (paper: about 10 bps/Hz)
  * FD sum-rate at varrho = inf   (paper: about 15)
  * FD sum-rate at varrho = 1
  * radar gain Gt / (Nt Pd) at varrho = 1  and  how far the TX lobe at the DL user is below the peak (varrho = 1;
    the paper's Fig. 5(a) shows the DL lobe at about the same level as the target lobe)
Use the loss that best matches the paper's numbers as `--loss` for the other experiments.
Run:  python experiments/exp_calibration.py [--mc 10] [--losses 0 5 10 15 20 25 30]
"""
from common import *
from ii_channels import steering
from vii_baselines import hd_rates, half_duplex_sumrate


def main():
    args = parse_args(default_mc=10, description=__doc__)
    losses = [0, 5, 10, 15, 20, 25, 30]
    if "--losses" in sys.argv:
        i = sys.argv.index("--losses") + 1
        losses = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            losses.append(float(sys.argv[i])); i += 1
    plt = get_plt()
    rows = []
    for L in losses:
        cfg = replace(make_cfg(args), comm_extra_loss_dB=L)
        b_dl = steering(cfg.dl_user[0], cfg.Nt)
        acc = dict(hd=[], inf=[], one=[], gt=[], dl=[], ru=[], rd=[])
        for t in range(args.mc):
            rng = np.random.default_rng(40_000 + t)
            ch = draw_all(cfg, 60, rng)
            acc["hd"].append(half_duplex_sumrate(*hd_rates(ch, cfg), delta=cfg.delta_hd))
            m = run_proposed(ch, cfg, np.inf, rng, args)["metrics"]; acc["inf"].append(m["Ru"] + m["Rd"])
            o = run_proposed(ch, cfg, 1.0, rng, args); m = o["metrics"]
            acc["one"].append(m["Ru"] + m["Rd"]); acc["ru"].append(m["Ru"]); acc["rd"].append(m["Rd"])
            acc["gt"].append(m["Gt"] / (cfg.Nt * cfg.Pd))
            b_t = steering(cfg.target[0], cfg.Nt)
            acc["dl"].append(10 * np.log10(abs(np.vdot(b_dl, o["p"])) ** 2 / abs(np.vdot(b_t, o["p"])) ** 2))
        r = {k: float(np.median(v)) for k, v in acc.items()}
        rows.append((L, r))
        print(f"loss {L:4.0f} dB | HD {r['hd']:5.1f} | FD varrho=inf {r['inf']:5.1f} | FD varrho=1 {r['one']:5.1f} "
              f"(Ru {r['ru']:.1f}, Rd {r['rd']:.1f}) | Gt/(Nt Pd) {r['gt']:.2f} | DL lobe vs target lobe {r['dl']:6.1f} dB")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    L = [x[0] for x in rows]
    ax[0].plot(L, [x[1]["hd"] for x in rows], "k+-", label="half-duplex")
    ax[0].plot(L, [x[1]["inf"] for x in rows], "o-", label="FD, varrho = inf")
    ax[0].plot(L, [x[1]["one"] for x in rows], "s-", label="FD, varrho = 1")
    ax[0].axhline(10, color="k", ls=":", lw=0.8); ax[0].axhline(15, color="tab:blue", ls=":", lw=0.8)
    ax[0].set_xlabel("extra user-link loss [dB]"); ax[0].set_ylabel("sum-rate [bps/Hz]")
    ax[0].set_title("dotted: approx. paper values (HD ~10, FD varrho=inf ~15)"); ax[0].legend()
    ax[1].plot(L, [x[1]["dl"] for x in rows], "o-", color="tab:red")
    ax[1].axhline(0, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("extra user-link loss [dB]"); ax[1].set_ylabel("TX level at DL user rel. to target lobe [dB]")
    ax[1].set_title("varrho = 1 (paper Fig. 5a: DL lobe about 10 dB below the target lobe)")
    save(fig, "calibration.png")
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, "calibration.npz"), losses=L, **{k: [x[1][k] for x in rows] for k in rows[0][1]})
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
