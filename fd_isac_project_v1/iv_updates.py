"""Closed-form BCD block updates (Algorithm 1). All vectors are 1-D complex arrays.

Objective being MAXIMISED (natural log, see metrics.objective_23):
  f = a1(ln rho_u - rho_u E_BS) + a2(ln rho_d - rho_d E_d)
      + a3 p^H Zt p + a4 w^H Zr w - (1/(2 beta)) |w^H Hsi p|^2

Each block sets the gradient w.r.t. the conjugate variable to zero (Wirtinger calculus).

NUMERICAL NOTE (important): with beta = 1e-25 the penalty weight c = 1/(2 beta) = 5e24 makes the
system matrices have condition numbers ~1e24+, so np.linalg.solve / eigh on them loses everything.
In both w- and p-updates the penalty is a RANK-ONE term c v v^H added to a well-conditioned
matrix B. We therefore use Sherman-Morrison:
      (B + c v v^H)^{-1} r = B^{-1} [ r - v (v^H B^{-1} r) / (1/c + v^H B^{-1} v) ]
which is exact and stable as c -> infinity (it projects out the SI direction).
"""
import numpy as np
from ii_channels import steering


def Zt_matrix(theta_r, Nt):
    """Zt = b b^H - Vt I, Vt = ||b||^2 = Nt. Eq (21). Negative semi-definite (null along b)."""
    b = steering(theta_r, Nt)
    return np.outer(b, b.conj()) - Nt * np.eye(Nt)


def Zr_matrix(theta_r, Nr):
    """Zr = a a^H - Vr I. Eq (22)."""
    a = steering(theta_r, Nr)
    return np.outer(a, a.conj()) - Nr * np.eye(Nr)


def update_rho(mse):
    """rho* = 1/E. Eqs (26),(27). (Exact for natural log; see metrics.objective_23 note.)"""
    return 1.0 / mse


def update_wu(w, Hu, rho_u, a1, Pu):
    """Eqs (28)-(32).
    Minimise a1 rho_u |g^H wu - 1|^2 + mu(||wu||^2 - Pu),  g = Hu^H w.
      grad:  a1 rho_u (g g^H wu - g) + mu wu = 0
      =>     wu = (a1 rho_u g g^H + mu I)^{-1} a1 rho_u g = a1 rho_u g / (mu + a1 rho_u ||g||^2)  (Sherman-Morrison)
    (Paper's Eq 30/32 omits rho_u -- it must be there, derived from a1*rho_u*E_BS.)
    Because wu is a scalar multiple of g, the KKT condition ||wu||^2 = Pu has a CLOSED FORM, no bisection:
      mu = 0 solution is g/||g||^2 (makes the MSE term zero). If it violates the power budget,
      wu = sqrt(Pu) g/||g||  (and mu = a1 rho_u (||g||/sqrt(Pu) - ||g||^2) > 0)."""
    g = Hu.conj().T @ w
    gn2 = np.vdot(g, g).real
    if 1.0 / gn2 <= Pu:                    # mu = 0 feasible
        return g / gn2
    return np.sqrt(Pu) * g / np.sqrt(gn2)


def update_w(p, wu, Hu, H, Hsi, rho_u, a1, a4, Zr, beta, sigma2_u):
    """Eqs (33)-(35). Unconstrained minimisation over w of
        a1 rho_u E_BS(w) - a4 w^H Zr w + (1/(2 beta)) |w^H Hsi p|^2
    E_BS(w) = w^H M w - 2Re{w^H Hu wu} + 1,  M = Hu wu wu^H Hu^H + H p p^H H^H + s2 I.
    Gradient = 0:  [a1 rho_u M - a4 Zr + c v v^H] w = a1 rho_u Hu wu,   v = Hsi p, c = 1/(2 beta).
    (This equals the paper's Eq 34-35 after multiplying both sides by 2.)
    Stable solve via Sherman-Morrison (see module note). The caller normalises w at the very end (Eq 44)."""
    Nr = Hu.shape[0]
    hu_wu = Hu @ wu
    h_p = H @ p
    M = np.outer(hu_wu, hu_wu.conj()) + np.outer(h_p, h_p.conj()) + sigma2_u * np.eye(Nr)
    B = a1 * rho_u * M - a4 * Zr
    r = a1 * rho_u * hu_wu
    v = Hsi @ p
    c = 1.0 / (2.0 * beta)
    Binv_r = np.linalg.solve(B, r)
    Binv_v = np.linalg.solve(B, v)
    coef = np.vdot(v, Binv_r) / (1.0 / c + np.vdot(v, Binv_v))
    return Binv_r - Binv_v * coef


def update_ud(p, Hd, sigma2_d):
    """MMSE combiner, Eq (37): (Hd p p^H Hd^H + s2 I)^{-1} Hd p = Hd p / (||Hd p||^2 + s2) (Sherman-Morrison)."""
    x = Hd @ p
    return x / (np.vdot(x, x).real + sigma2_d)


def _find_gamma(n2, lo, Pd, iters):
    """Root of n2(Gamma) = Pd for a strictly decreasing n2 on (lo, inf); returns Gamma."""
    lo = lo
    hi = lo + 1.0
    while n2(hi) > Pd:
        hi = lo + 2.0 * (hi - lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if n2(mid) > Pd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def update_p(w, ud, wu, Hu, H, Hd, Hsi, rho_u, rho_d, a1, a2, a3, Zt, beta, Pd, sigma2_u,
             bisect_iters=200, return_gamma=False):
    """Eqs (38)-(43). Minimise over p, subject to ||p||^2 = Pd (paper Eq 40):
        a1 rho_u |w^H H p|^2 + a2 rho_d E_d(p) - a3 p^H Zt p + c |w^H Hsi p|^2 + Gamma(||p||^2 - Pd)
    Gradient = 0:  [B0 + c v v^H + Gamma I] p = r
        B0 = a1 rho_u H^H w w^H H + a2 rho_d Hd^H ud ud^H Hd - a3 Zt      (PSD, Nt x Nt)
        r  = a2 rho_d Hd^H ud,   v = Hsi^H w,   c = 1/(2 beta)
    (Paper's Eq 42 writes Hd ud ud^H Hd^H -- dimensionally wrong; it must be Hd^H ud ud^H Hd,
     and '-a3 Zt I' should be '-a3 Zt'.)

    WHY EQUALITY: the bound Gt <= p^H Zt p + Vt*Pd (Eq 21) is tight only when ||p||^2 = Pd. With an
    inequality, the surrogate's -a3 Vt ||p||^2 term just shrinks p (we observed ||p||^2 ~ 0.1 Pd) and the
    convergence proof of Sec. IV breaks. Hence Gamma is free in sign; ||p(Gamma)||^2 is decreasing on
    (-lambda_min, inf), so we bisect.

    TWO NUMERICAL PATHS for the huge penalty c:
      * c ||v||^2 huge (beta ~ 1e-25): c -> infinity enforces v^H p = 0, i.e. p lives in the (Nt-1)-dim
        orthogonal complement of v. We work in an orthonormal basis U of that subspace (exact, stable).
      * moderate c: Sherman-Morrison in the eigenbasis of B0.
    One eigh per call, then each bisection step is O(Nt).
    wu, Hu, sigma2_u are unused (they only enter constants of the p-subproblem); kept for signature parity."""
    hw = H.conj().T @ w
    hdu = Hd.conj().T @ ud
    B0 = a1 * rho_u * np.outer(hw, hw.conj()) + a2 * rho_d * np.outer(hdu, hdu.conj()) - a3 * Zt
    B0 = (B0 + B0.conj().T) / 2
    r = a2 * rho_d * hdu
    v = Hsi.conj().T @ w
    c = 1.0 / (2.0 * beta)
    vn2 = np.vdot(v, v).real
    scale = max(1.0, np.linalg.eigvalsh(B0).max())
    eps = 1e-12 * scale

    if c * vn2 > 1e12 * scale:                       # ---- projected path (v^H p = 0) ----
        Qf, _ = np.linalg.qr(v[:, None], mode="complete")
        U = Qf[:, 1:]                                # orthonormal basis of v-perp, Nt x (Nt-1)
        lam, Qq = np.linalg.eigh(U.conj().T @ B0 @ U)
        rt = Qq.conj().T @ (U.conj().T @ r)
        n2 = lambda G: float(np.sum(np.abs(rt) ** 2 / (lam + G) ** 2))
        Gam = _find_gamma(n2, -lam.min() + eps, Pd, bisect_iters)
        p = U @ (Qq @ (rt / (lam + Gam)))
        q_min = U @ Qq[:, 0]
    else:                                            # ---- Sherman-Morrison path ----
        lam, Q = np.linalg.eigh(B0)
        rt, vt = Q.conj().T @ r, Q.conj().T @ v

        def p_tilde(G):
            d = lam + G
            x = rt / d
            s = np.vdot(vt, x)
            t = np.sum(np.abs(vt) ** 2 / d)
            return x - (vt / d) * s / (1.0 / c + t)

        n2 = lambda G: float(np.linalg.norm(p_tilde(G)) ** 2)
        Gam = _find_gamma(n2, -lam.min() + eps, Pd, bisect_iters)
        p = Q @ p_tilde(Gam)
        q_min = Q[:, 0]
    # Degenerate / 'hard' case (e.g. a3 = 0, so B0 has a null space and r has no component there):
    # the stationary solution can have ||p||^2 < Pd at Gamma = -lambda_min. The remaining power is put on the
    # minimum-eigenvalue direction (zero extra cost when lambda_min = 0). Rescaling p instead would NOT be
    # optimal and breaks monotonicity.
    n2p = np.linalg.norm(p) ** 2
    if n2p < Pd * (1 - 1e-10):
        b_ = np.vdot(q_min, p).real
        p = p + (-b_ + np.sqrt(b_ ** 2 + (Pd - n2p))) * q_min
    p = p * (np.sqrt(Pd) / np.linalg.norm(p))        # remove residual bisection error
    return (p, Gam) if return_gamma else p


if __name__ == "__main__":
    # Self-checks: python updates.py
    from i_config import Cfg
    from ii_channels import draw_all
    from iii_metrics import (mse_bs, mse_dl, objective_23, residual_SI_dB, soi_over_si_dB,
                         rate_ul, rate_dl, beampattern_gains)

    cfg = Cfg(); rng = np.random.default_rng(3)
    cr = lambda n: (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    al = (1.0, 1.0, 1.0, 1.0)
    Zt, Zr = Zt_matrix(cfg.target[0], cfg.Nt), Zr_matrix(cfg.target[0], cfg.Nr)
    print("Zt, Zr negative semi-definite:", np.linalg.eigvalsh(Zt).max() < 1e-9, np.linalg.eigvalsh(Zr).max() < 1e-9)

    for beta in (1e-3, 1e-25):
        ch = draw_all(cfg, 40, rng)
        Hu, Hd, H, Hsi = ch["Hu"], ch["Hd"], ch["H"], ch["Hsi"]
        p = cr(cfg.Nt); p *= np.sqrt(cfg.Pd) / np.linalg.norm(p)
        w = cr(cfg.Nr); w /= np.linalg.norm(w)
        wu = cr(cfg.Nu); wu *= np.sqrt(cfg.Pu) / np.linalg.norm(wu)
        ud = cr(cfg.Nd)

        def F(p, w, wu, ud):   # objective with rho at its optimum for the given variables
            ru = update_rho(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u)); rd = update_rho(mse_dl(ud, p, Hd, cfg.sigma2_d))
            return objective_23(p, w, wu, ud, ru, rd, ch, cfg, al, beta=beta)

        hist = []
        for it in range(30):
            ru = update_rho(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u)); rd = update_rho(mse_dl(ud, p, Hd, cfg.sigma2_d))
            wu = update_wu(w, Hu, ru, al[0], cfg.Pu)
            w = update_w(p, wu, Hu, H, Hsi, ru, al[0], al[3], Zr, beta, cfg.sigma2_u)
            ud = update_ud(p, Hd, cfg.sigma2_d)
            p = update_p(w, ud, wu, Hu, H, Hd, Hsi, ru, rd, al[0], al[1], al[2], Zt, beta, cfg.Pd, cfg.sigma2_u)
            hist.append(F(p, w, wu, ud))
        d = np.diff(hist)
        wn = w / np.linalg.norm(w)
        print(f"\nbeta={beta:g}: f23 first/last = {hist[0]:.4f} / {hist[-1]:.4f}; "
              f"min increment = {d.min():.3e} (should be >= ~0)")
        print("  ||p||^2/Pd =", round(np.linalg.norm(p) ** 2 / cfg.Pd, 6), " ||wu||^2/Pu =", round(np.linalg.norm(wu) ** 2 / cfg.Pu, 6))
        print("  residual SI [dB] =", round(residual_SI_dB(wn, Hsi, p, cfg.noise_dBm), 1),
              " SoI/SI [dB] =", round(soi_over_si_dB(wn, ch['Hr'], Hu, wu, Hsi, p), 1),
              " Ru, Rd =", round(rate_ul(wn, p, wu, Hu, H, cfg.sigma2_u), 2), round(rate_dl(ud, p, Hd, cfg.sigma2_d), 2),
              " Gt, Gr =", [round(x, 2) for x in beampattern_gains(p, wn, cfg.target[0], cfg.Nt, cfg.Nr)])

        # random-perturbation optimality checks (each block should beat nearby points)
        ud = update_ud(p, Hd, cfg.sigma2_d)
        f0 = F(p, w, wu, ud); worse = {"w": 0, "p": 0, "wu": 0, "ud": 0}
        for _ in range(200):
            e = 1e-3
            if F(p, w + e * np.linalg.norm(w) * cr(cfg.Nr), wu, ud) > f0 + 1e-9 * abs(f0): worse["w"] += 1
            pp = p + e * np.linalg.norm(p) * cr(cfg.Nt); pp *= min(1.0, np.sqrt(cfg.Pd) / np.linalg.norm(pp))
            if F(pp, w, wu, ud) > f0 + 1e-9 * abs(f0): worse["p"] += 1
            wp = wu + e * np.linalg.norm(wu) * cr(cfg.Nu); wp *= min(1.0, np.sqrt(cfg.Pu) / np.linalg.norm(wp))
            if F(p, w, wp, ud) > f0 + 1e-9 * abs(f0): worse["wu"] += 1
            if F(p, w, wu, ud + e * np.linalg.norm(ud) * cr(cfg.Nd)) > f0 + 1e-9 * abs(f0): worse["ud"] += 1
        print("  perturbations that BEAT the solution (want 0 each):", worse)
