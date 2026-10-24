"""All performance metrics. Independent of the solver (reused in tests and experiments).

Conventions: vectors are 1-D complex arrays; powers in mW; `alphas` = (a1, a2, a3, a4);
H = Hr + Hsi (aggregate channel seen by the transceiver RX in Eq. 3).
"""
import numpy as np
from ii_channels import steering


# ----------------------------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------------------------
def mse_bs(w, p, wu, Hu, H, sigma2_u):
    """E_BS, Eq (18). Compact form of the expanded expression in the paper:
        E_BS = |w^H Hu wu - 1|^2 + |w^H H p|^2 + sigma_u^2 ||w||^2
    (expanding gives exactly Eq 18: wuHuHw.. - 2Re{w^H Hu wu} + w^H(H p p^H H^H + s2 I)w + 1).
    Both the radar return (Hr p) and the SI (Hsi p) are inside H p -> treated as interference."""
    s = np.vdot(w, Hu @ wu)
    i = np.vdot(w, H @ p)
    return float(abs(s - 1) ** 2 + abs(i) ** 2 + sigma2_u * np.vdot(w, w).real)


def mse_dl(ud, p, Hd, sigma2_d):
    """E_d, Eq (19): |ud^H Hd p - 1|^2 + sigma_d^2 ||ud||^2."""
    s = np.vdot(ud, Hd @ p)
    return float(abs(s - 1) ** 2 + sigma2_d * np.vdot(ud, ud).real)


def sinr_ul(w, p, wu, Hu, H, sigma2_u):
    """gamma_u, Eq (12)."""
    num = abs(np.vdot(w, Hu @ wu)) ** 2
    den = abs(np.vdot(w, H @ p)) ** 2 + np.vdot(w, w).real * sigma2_u
    return float(num / den)


def sinr_dl(ud, p, Hd, sigma2_d):
    """gamma_d, Eq (14)."""
    num = abs(np.vdot(ud, Hd @ p)) ** 2
    den = np.vdot(ud, ud).real * sigma2_d
    return float(num / den)


def rate_ul(w, p, wu, Hu, H, sigma2_u):
    """Ru = log2(1 + gamma_u) [bits/s/Hz], Eqs (11),(12)."""
    return float(np.log2(1 + sinr_ul(w, p, wu, Hu, H, sigma2_u)))


def rate_dl(ud, p, Hd, sigma2_d):
    """Rd = log2(1 + gamma_d) [bits/s/Hz], Eqs (13),(14)."""
    return float(np.log2(1 + sinr_dl(ud, p, Hd, sigma2_d)))


def beampattern_gains(p, w, theta_r, Nt, Nr):
    """Gt = |b^H(theta_r) p|^2, Gr = |w^H a(theta_r)|^2. Eq (15)."""
    b = steering(theta_r, Nt)
    a = steering(theta_r, Nr)
    return float(abs(np.vdot(b, p)) ** 2), float(abs(np.vdot(w, a)) ** 2)


def residual_SI_dB(w, Hsi, p, noise_dBm):
    """Pres[dB] = 10log10(|w^H Hsi p|^2 / 1mW) - Pnoise[dBm]. Sec V-A.
    Powers are already in mW in this code base, so the '/1mW' is implicit. Negative => below noise floor."""
    return float(10 * np.log10(abs(np.vdot(w, Hsi @ p)) ** 2 + 1e-300) - noise_dBm)


def soi_over_si_dB(w, Hr, Hu, wu, Hsi, p):
    """P_SoI/SI = 10log10((|w^H Hr p|^2 + |w^H Hu wu|^2) / |w^H Hsi p|^2). Sec V-A.
    Positive => signal of interest (echo + uplink) is not drowned by SI."""
    soi = abs(np.vdot(w, Hr @ p)) ** 2 + abs(np.vdot(w, Hu @ wu)) ** 2
    si = abs(np.vdot(w, Hsi @ p)) ** 2
    return float(10 * np.log10(soi / (si + 1e-300)))


def penalty_term(w, p, Hsi, beta):
    """(1/(2 beta)) |w^H Hsi p|^2 -- the SI penalty in (17a)/(20)/(23)."""
    return float(abs(np.vdot(w, Hsi @ p)) ** 2 / (2 * beta))


# ----------------------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------------------
def objective_17(p, w, wu, ud, ch, cfg, alphas, beta=None):
    """Original penalised objective (17a):
        a1 Ru + a2 Rd + a3 Gt + a4 Gr - (1/(2 beta)) |w^H Hsi p|^2     (rates in bits)"""
    a1, a2, a3, a4 = alphas
    beta = cfg.beta if beta is None else beta
    Ru = rate_ul(w, p, wu, ch["Hu"], ch["H"], cfg.sigma2_u)
    Rd = rate_dl(ud, p, ch["Hd"], cfg.sigma2_d)
    Gt, Gr = beampattern_gains(p, w, cfg.target[0], cfg.Nt, cfg.Nr)
    return a1 * Ru + a2 * Rd + a3 * Gt + a4 * Gr - penalty_term(w, p, ch["Hsi"], beta)


def objective_23(p, w, wu, ud, rho_u, rho_d, ch, cfg, alphas, beta=None, log=np.log):
    """WMMSE-transformed objective (23) -- track THIS for convergence (Eq 51):
        a1(log rho_u - rho_u E_BS) + a2(log rho_d - rho_d E_d)
        + a3 p^H Zt p + a4 w^H Zr w - (1/(2 beta)) |w^H Hsi p|^2
    where p^H Zt p = |b^H p|^2 - Vt ||p||^2  and  w^H Zr w = |w^H a|^2 - Vr ||w||^2  (Eqs 21-22,
    constants Vt*Pd, Vr dropped).

    NOTE ON log: with natural log, rho* = 1/E exactly (Eqs 26-27). The paper writes log2 in (20) but
    keeps rho* = 1/E, which is only exact for ln. Default here is np.log (=> rates in nats, i.e.
    a1, a2 effectively scaled by ln2 w.r.t. bits). Pass log=np.log2 only if you also change the rho
    update to 1/(E ln2)."""
    a1, a2, a3, a4 = alphas
    beta = cfg.beta if beta is None else beta
    Vt, Vr = float(cfg.Nt), float(cfg.Nr)          # ||b||^2 = Nt, ||a||^2 = Nr for a ULA
    E_bs = mse_bs(w, p, wu, ch["Hu"], ch["H"], cfg.sigma2_u)
    E_d = mse_dl(ud, p, ch["Hd"], cfg.sigma2_d)
    Gt, Gr = beampattern_gains(p, w, cfg.target[0], cfg.Nt, cfg.Nr)
    ptZp = Gt - Vt * np.vdot(p, p).real
    wZw = Gr - Vr * np.vdot(w, w).real
    return float(a1 * (log(rho_u) - rho_u * E_bs) + a2 * (log(rho_d) - rho_d * E_d)
                 + a3 * ptZp + a4 * wZw - penalty_term(w, p, ch["Hsi"], beta))


if __name__ == "__main__":
    # self-checks: python metrics.py
    from i_config import Cfg
    from ii_channels import draw_all
    cfg = Cfg(); rng = np.random.default_rng(1)
    ch = draw_all(cfg, 40, rng)
    cr = lambda n: (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    p = cr(cfg.Nt); p *= np.sqrt(cfg.Pd) / np.linalg.norm(p)
    wu = cr(cfg.Nu); wu *= np.sqrt(cfg.Pu) / np.linalg.norm(wu)
    w = cr(cfg.Nr); w /= np.linalg.norm(w)
    ud = cr(cfg.Nd)
    Hu, H, Hd = ch["Hu"], ch["H"], ch["Hd"]

    # 1) compact MSE == expanded Eq (18)
    e18 = (np.vdot(w, Hu @ wu) * np.vdot(Hu @ wu, w)).real - 2 * np.vdot(w, Hu @ wu).real \
        + (np.vdot(w, H @ p) * np.vdot(H @ p, w)).real + cfg.sigma2_u * np.vdot(w, w).real + 1
    print("Eq18 compact vs expanded:", np.isclose(mse_bs(w, p, wu, Hu, H, cfg.sigma2_u), e18))

    # 2) -ln E <= ln(1+gamma) always; equality at the MMSE combiner
    ud_mmse = np.linalg.solve(np.outer(Hd @ p, (Hd @ p).conj()) + cfg.sigma2_d * np.eye(cfg.Nd), Hd @ p)
    for name, u in [("random ud", ud), ("MMSE ud", ud_mmse)]:
        E = mse_dl(u, p, Hd, cfg.sigma2_d); g = sinr_dl(u, p, Hd, cfg.sigma2_d)
        print(f"{name}: -lnE={-np.log(E):.6f}  ln(1+gamma)={np.log1p(g):.6f}")
    A = np.outer(H @ p, (H @ p).conj()) + np.outer(Hu @ wu, (Hu @ wu).conj()) + cfg.sigma2_u * np.eye(cfg.Nr)
    w_mmse = np.linalg.solve(A, Hu @ wu)
    E = mse_bs(w_mmse, p, wu, Hu, H, cfg.sigma2_u)
    print("UL MMSE w: -lnE =", round(-np.log(E), 6), " ln(1+gamma_u) =",
          round(np.log1p(sinr_ul(w_mmse, p, wu, Hu, H, cfg.sigma2_u)), 6))

    # 3) objectives run
    Eb = mse_bs(w, p, wu, Hu, H, cfg.sigma2_u); Ed = mse_dl(ud, p, Hd, cfg.sigma2_d)
    print("f17 =", objective_17(p, w, wu, ud, ch, cfg, (1, 1, 1, 1), beta=1.0))
    print("f23 =", objective_23(p, w, wu, ud, 1 / Eb, 1 / Ed, ch, cfg, (1, 1, 1, 1), beta=1.0))
    print("Residual SI [dB]:", residual_SI_dB(w, ch["Hsi"], p, cfg.noise_dBm),
          " SoI/SI [dB]:", soi_over_si_dB(w, ch["Hr"], Hu, wu, ch["Hsi"], p))
