"""Algorithm 1 (penalty-based BCD) -- Liu et al., JSAC 2023."""
import numpy as np
from iii_metrics import (mse_bs, mse_dl, objective_23, rate_ul, rate_dl, beampattern_gains,
                     residual_SI_dB, soi_over_si_dB, sinr_ul)
from iv_updates import (Zt_matrix, Zr_matrix, update_rho, update_wu, update_w, update_ud, update_p)


def alphas_from_varrho(varrho: float):
    """varrho = alpha_com / alpha_radar (Sec V-A), alpha_1 = alpha_2 = alpha_com, alpha_3 = alpha_4 = alpha_radar.
    varrho = inf -> communications only (alpha_3 = alpha_4 = 0). Otherwise alpha_radar = 1."""
    if np.isinf(varrho):
        return (1.0, 1.0, 0.0, 0.0)
    return (float(varrho), float(varrho), 1.0, 1.0)


def _cr(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def random_init(cfg, rng):
    """Random feasible start: ||p||^2 = Pd (needed so the bound (21) is tight from iteration 0),
    ||w|| = 1, ||wu||^2 = Pu."""
    p = _cr(rng, cfg.Nt); p *= np.sqrt(cfg.Pd) / np.linalg.norm(p)
    w = _cr(rng, cfg.Nr); w /= np.linalg.norm(w)
    wu = _cr(rng, cfg.Nu); wu *= np.sqrt(cfg.Pu) / np.linalg.norm(wu)
    ud = _cr(rng, cfg.Nd)
    return dict(p=p, w=w, wu=wu, ud=ud)


def solve_fd_isac(ch: dict, cfg, a=(1, 1, 1, 1), init=None, rng=None, log=False, beta=None):
    """
    ch : dict(Hu, Hd, Hr, Hsi, H)
    a  : (alpha1..alpha4); with varrho = a_com/a_radar -> a = alphas_from_varrho(varrho)
    init : optional dict(p, w, wu, ud); otherwise random (uses rng, default seeded by cfg.seed)
    beta : penalty parameter override (default cfg.beta)

    Per iteration (Algorithm 1, lines 3-8):  rho_u, rho_d -> wu -> w -> ud -> p
    Convergence: zeta = |f_n - f_{n-1}| / |f_{n-1}| <= cfg.tol  (Eq 51), where f = objective (23) evaluated
    with rho at its optimum for the current variables (== f_g in the paper; guaranteed non-decreasing).
    After the loop: w <- w/||w|| (Eq 44).

    Returns dict(p, w, wu, ud, rho_u, rho_d, history, zeta, n_iter, converged, w_raw, metrics).
    """
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    beta = cfg.beta if beta is None else beta
    Hu, Hd, H, Hsi = ch["Hu"], ch["Hd"], ch["H"], ch["Hsi"]
    a1, a2, a3, a4 = a
    Zt = Zt_matrix(cfg.target[0], cfg.Nt)
    Zr = Zr_matrix(cfg.target[0], cfg.Nr)

    st = random_init(cfg, rng) if init is None else {k: np.array(v, dtype=complex) for k, v in init.items()}
    p, w, wu, ud = st["p"], st["w"], st["wu"], st["ud"]

    def f_g(p, w, wu, ud):
        ru = update_rho(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u))
        rd = update_rho(mse_dl(ud, p, Hd, cfg.sigma2_d))
        return objective_23(p, w, wu, ud, ru, rd, ch, cfg, a, beta=beta)

    history, zeta = [f_g(p, w, wu, ud)], []
    converged = False
    for it in range(1, cfg.max_iter + 1):
        rho_u = update_rho(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u))                       # line 3
        rho_d = update_rho(mse_dl(ud, p, Hd, cfg.sigma2_d))                             # line 4
        wu = update_wu(w, Hu, rho_u, a1, cfg.Pu)                                        # line 5
        w = update_w(p, wu, Hu, H, Hsi, rho_u, a1, a4, Zr, beta, cfg.sigma2_u)          # line 6
        ud = update_ud(p, Hd, cfg.sigma2_d)                                             # line 7
        p = update_p(w, ud, wu, Hu, H, Hd, Hsi, rho_u, rho_d, a1, a2, a3, Zt, beta,     # line 8
                     cfg.Pd, cfg.sigma2_u)
        history.append(f_g(p, w, wu, ud))
        zeta.append(abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-300))     # Eq (51)
        if log:
            print(f"it {it:3d}  f_g = {history[-1]: .6e}  zeta = {zeta[-1]:.3e}")
        if zeta[-1] <= cfg.tol:
            converged = True
            break

    w_raw = w.copy()
    w = w / np.linalg.norm(w)                                                           # Eq (44)

    rho_u = update_rho(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u))
    rho_d = update_rho(mse_dl(ud, p, Hd, cfg.sigma2_d))
    Gt, Gr = beampattern_gains(p, w, cfg.target[0], cfg.Nt, cfg.Nr)
    metrics = dict(
        Ru=rate_ul(w, p, wu, Hu, H, cfg.sigma2_u), Rd=rate_dl(ud, p, Hd, cfg.sigma2_d), Gt=Gt, Gr=Gr,
        res_SI_dB=residual_SI_dB(w, Hsi, p, cfg.noise_dBm),
        soi_si_dB=soi_over_si_dB(w, ch["Hr"], Hu, wu, Hsi, p),
    )
    return dict(p=p, w=w, wu=wu, ud=ud, rho_u=rho_u, rho_d=rho_d, history=np.array(history),
                zeta=np.array(zeta), n_iter=len(zeta), converged=converged, w_raw=w_raw, metrics=metrics)


if __name__ == "__main__":
    # python algorithm.py
    from i_config import Cfg
    from ii_channels import draw_all
    cfg = Cfg()
    rng = np.random.default_rng(0)

    print("== Fig. 2-like convergence ==")
    for Nt, si in [(16, 60), (8, 20)]:
        c = Cfg(Nt=Nt, Nr=Nt)
        ch = draw_all(c, si, np.random.default_rng(1))
        out = solve_fd_isac(ch, c, a=alphas_from_varrho(1.0), rng=np.random.default_rng(2))
        d = np.diff(out["history"])
        print(f"Nt={Nt}, SI={si} dB: iters={out['n_iter']}, converged={out['converged']}, "
              f"min f-increment={d.min():.2e}, ||p||^2/Pd={np.linalg.norm(out['p'])**2/c.Pd:.4f}")

    print("\n== varrho sweep @ SI = 60 dB (cf. Figs 3-5) ==")
    ch = draw_all(cfg, 60, np.random.default_rng(1))
    for vr in (0.01, 1, 100, 1e3, np.inf):
        out = solve_fd_isac(ch, cfg, a=alphas_from_varrho(vr), rng=np.random.default_rng(2))
        m = out["metrics"]
        print(f"varrho={vr:>6}: it={out['n_iter']:3d} Pres={m['res_SI_dB']:8.1f} dB  SoI/SI={m['soi_si_dB']:7.1f} dB  "
              f"Ru={m['Ru']:.2f} Rd={m['Rd']:.2f} Gt={m['Gt']:.1f} Gr={m['Gr']:.2f}")
