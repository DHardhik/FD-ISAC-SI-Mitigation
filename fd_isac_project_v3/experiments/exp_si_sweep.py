"""Experiment: SI-power sweep  --  reproduces Figs. 3, 4 and 7 from ONE set of Monte-Carlo runs.   (v3)
Fig 3: residual SI power (re. noise floor) vs SI power, varrho in {1, 10, 100, 1e3, inf} and NSP.
Fig 4: SoI/SI power ratio vs SI power.
Fig 7: average FD sum-rate Ru + Rd vs SI power, plus half-duplex (delta = 0.5).
v3: the three figures share the same channel draws (v2 ran the whole sweep twice); trials run on several CPU cores
(--jobs); medians are used for the dB quantities (Figs 3, 4) and means for the rates (Fig 7, as in the paper).
Run:  python experiments/exp_si_sweep.py [--mc 100] [--jobs 4] [--show]
"""
from common import *
from multiprocessing import Pool
from vii_baselines import hd_rates, half_duplex_sumrate

VARRHOS = (1, 10, 100, 1e3, np.inf)
NAMES = [label_varrho(v) for v in VARRHOS] + ["NSP method"]


def one_trial(job):
    """All schemes on one channel draw. Returns (res[K], soi[K], sr[K], hd)."""
    i, si, t, args = job
    cfg = make_cfg(args)
    rng = np.random.default_rng(10_000 * (i + 1) + t)
    ch = draw_all(cfg, si, rng)
    res, soi, sr = [], [], []
    for vr in VARRHOS:
        m = run_proposed(ch, cfg, vr, rng, args)["metrics"]
        res.append(m["res_SI_dB"]); soi.append(m["soi_si_dB"]); sr.append(m["Ru"] + m["Rd"])
    m = run_nsp(ch, cfg)["metrics"]
    res.append(m["res_SI_dB"]); soi.append(m["soi_si_dB"]); sr.append(m["Ru"] + m["Rd"])
    return i, t, res, soi, sr, half_duplex_sumrate(*hd_rates(ch, cfg), delta=cfg.delta_hd)


def run_sweep(args):
    cfg = make_cfg(args)
    si_list = np.array(cfg.si_sweep_dB, dtype=float)
    K = len(NAMES)
    res = np.zeros((K, len(si_list), args.mc)); soi = np.zeros_like(res); sr = np.zeros_like(res)
    hd = np.zeros((len(si_list), args.mc))
    jobs = [(i, si, t, args) for i, si in enumerate(si_list) for t in range(args.mc)]
    done = 0
    with Pool(args.jobs) as pool:
        for i, t, r, s, q, h in pool.imap_unordered(one_trial, jobs, chunksize=2):
            res[:, i, t], soi[:, i, t], sr[:, i, t], hd[i, t] = r, s, q, h
            done += 1
            if done % args.mc == 0:
                print(f"{done}/{len(jobs)} trials done", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, f"si_sweep{('_' + TAG) if TAG else ''}.npz"),
             si=si_list, res=res, soi=soi, sr=sr, hd=hd, names=NAMES)
    return si_list, res, soi, sr, hd


def plot_all(args, si_list, res, soi, sr, hd):
    plt = get_plt()
    styles = dict(zip(NAMES, ["<-", "s-", ">-", "o-", "D-", "^--"]))
    for data, ylabel, fname, ttl in ((res, "Residual SI [dB]", "fig3_residual_SI.png", "Fig. 3"),
                                     (soi, "SoI over SI [dB]", "fig4_SoI_over_SI.png", "Fig. 4")):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for k, nm in enumerate(NAMES):
            med, lo, hi = robust_stats(data[k], axis=1)
            ax.plot(si_list, med, styles[nm], ms=4, label=nm)
            if not nm.startswith("NSP"):
                ax.fill_between(si_list, lo, hi, alpha=0.08)
        ax.set_xlabel("SI power [dB]"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ttl}: median and 25-75% band over {args.mc} trials")
        ax.legend(fontsize=7, ncol=2)
        save(fig, fname)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for k, nm in enumerate(NAMES):
        ax.plot(si_list, sr[k].mean(axis=1), styles[nm], ms=4, label=nm)
    ax.plot(si_list, hd.mean(axis=1), color="tab:brown", marker="d", ms=4, label="Half-Duplex")
    ax.set_xlabel("SI power [dB]"); ax.set_ylabel("Average sum rate [bps/Hz] in one unit time-frequency resource")
    ax.set_title(f"Fig. 7 ({args.mc} trials)"); ax.legend(fontsize=7, ncol=2)
    save(fig, "fig7_sumrate.png")
    if args.show:
        plt.show()
    print("\nMedian residual SI [dB] at SI = 20 / 60 dB:")
    for k, n in enumerate(NAMES):
        print(f"  {n:26s} {np.median(res[k, 0]):8.1f} {np.median(res[k, -1]):8.1f}")
    print("Mean sum-rate [bps/Hz] at SI = 20 / 60 dB:")
    for k, n in enumerate(NAMES):
        print(f"  {n:26s} {sr[k, 0].mean():6.2f} {sr[k, -1].mean():6.2f}")
    print(f"  {'Half-duplex':26s} {hd[0].mean():6.2f} {hd[-1].mean():6.2f}")


def main():
    args = parse_args(default_mc=20, description=__doc__)
    plot_all(args, *run_sweep(args))


if __name__ == "__main__":
    main()
