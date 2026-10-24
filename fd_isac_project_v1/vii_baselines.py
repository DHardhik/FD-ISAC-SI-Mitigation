"""Comparison schemes (Sec. V): NSP, radar-only, communication-only, half-duplex.

All designs return (p, w): TX precoder with ||p||^2 = Pd and RX beamformer with ||w|| = 1.
`complete_baseline` then fills in the uplink precoder wu and downlink combiner ud so every scheme
can be scored with the same metrics as the proposed algorithm.
"""
import numpy as np
from ii_channels import steering
from iii_metrics import (rate_ul, rate_dl, beampattern_gains, residual_SI_dB, soi_over_si_dB)
from iv_updates import update_ud


def _unit(x):
    return x / np.linalg.norm(x)


def radar_only(cfg):
    """p_r = b(theta_r) (scaled to Pd), w_r = a(theta_r) (unit norm). Sec V-D."""
    p = np.sqrt(cfg.Pd) * _unit(steering(cfg.target[0], cfg.Nt))
    w = _unit(steering(cfg.target[0], cfg.Nr))
    return p, w


def comm_only(cfg, paper_literal=False):
    """Communication-only: TX beam to the DOWNLINK user, RX beam to the UPLINK user.
    The paper writes p_u = b(theta_u), w_u = a(theta_u). Since p is the downlink precoder, I read the TX angle
    as theta_d; set paper_literal=True to use theta_u for both (exactly as printed)."""
    th_tx = cfg.ul_user[0] if paper_literal else cfg.dl_user[0]
    p = np.sqrt(cfg.Pd) * _unit(steering(th_tx, cfg.Nt))
    w = _unit(steering(cfg.ul_user[0], cfg.Nr))
    return p, w


def nsp_design(ch, cfg, power_split=0.5, null_dl=True):
    """Null-space-projection (NSP) baseline of [28]-[30], as described in Sec. I of the paper.
    TX: weighted sum of a radar beam and a downlink-communication beam,
          p = sqrt(Pd) * normalise( sqrt(xi) b_hat(theta_r) + sqrt(1-xi) b_hat(theta_d) ),  xi = power_split.
    RX: steer to the target and project onto the null space of the interference columns
          C = [Hsi p, a(theta_d)]  (SI vector, and the downlink-beam direction if null_dl),
          w = normalise( (I - C C^+) a(theta_r) ).
    Because Hsi p is nulled exactly, the residual SI is at numerical precision for any SI level
    (the flat NSP line in Fig. 3). INTERPRETATION NOTE: refs [28]-[30] are only described in words in this paper;
    the choice of xi and of the null set are my reading -- state them in your report."""
    xi = power_split
    b_t = _unit(steering(cfg.target[0], cfg.Nt))
    b_d = _unit(steering(cfg.dl_user[0], cfg.Nt))
    p = np.sqrt(xi) * b_t + np.sqrt(1 - xi) * b_d
    p = np.sqrt(cfg.Pd) * _unit(p)
    cols = [ch["Hsi"] @ p]
    if null_dl:
        cols.append(steering(cfg.dl_user[0], cfg.Nr))
    C = np.stack(cols, axis=1)
    P = np.eye(cfg.Nr) - C @ np.linalg.pinv(C)
    w = _unit(P @ steering(cfg.target[0], cfg.Nr))
    return p, w


def complete_baseline(ch, cfg, p, w):
    """Fill in wu (MRT w.r.t. the effective uplink channel w^H Hu, power Pu) and ud (MMSE), then compute
    the same metrics as the proposed solver. Returns dict(p, w, wu, ud, metrics)."""
    g = ch["Hu"].conj().T @ w
    wu = np.sqrt(cfg.Pu) * _unit(g)
    ud = update_ud(p, ch["Hd"], cfg.sigma2_d)
    Gt, Gr = beampattern_gains(p, w, cfg.target[0], cfg.Nt, cfg.Nr)
    m = dict(Ru=rate_ul(w, p, wu, ch["Hu"], ch["H"], cfg.sigma2_u),
             Rd=rate_dl(ud, p, ch["Hd"], cfg.sigma2_d), Gt=Gt, Gr=Gr,
             res_SI_dB=residual_SI_dB(w, ch["Hsi"], p, cfg.noise_dBm),
             soi_si_dB=soi_over_si_dB(w, ch["Hr"], ch["Hu"], wu, ch["Hsi"], p))
    return dict(p=p, w=w, wu=wu, ud=ud, metrics=m)


def half_duplex_sumrate(Rdl, Rul, delta=0.5):
    """Rhd = delta*Rdl + (1-delta)*Rul  (sum-rate per unit time-frequency resource). Sec V-C."""
    return delta * Rdl + (1 - delta) * Rul


def hd_rates(ch, cfg):
    """Single-link rates of a half-duplex system (no SI exists): beamform with the dominant singular
    vectors of Hd (downlink, power Pd) and Hu (uplink, power Pu), MRC at the receiver.
    Returns (Rdl, Rul) in bits/s/Hz."""
    sd = np.linalg.svd(ch["Hd"], compute_uv=False)[0]
    su = np.linalg.svd(ch["Hu"], compute_uv=False)[0]
    return (float(np.log2(1 + cfg.Pd * sd ** 2 / cfg.sigma2_d)),
            float(np.log2(1 + cfg.Pu * su ** 2 / cfg.sigma2_u)))


if __name__ == "__main__":
    from i_config import Cfg
    from ii_channels import draw_all
    from v_algorithm import solve_fd_isac, alphas_from_varrho
    cfg = Cfg()
    ch = draw_all(cfg, 60, np.random.default_rng(1))
    rows = {"radar-only": complete_baseline(ch, cfg, *radar_only(cfg)),
            "comm-only": complete_baseline(ch, cfg, *comm_only(cfg)),
            "NSP xi=0.5": complete_baseline(ch, cfg, *nsp_design(ch, cfg, 0.5))}
    out = solve_fd_isac(ch, cfg, a=alphas_from_varrho(1.0), rng=np.random.default_rng(2))
    rows["proposed varrho=1"] = out
    print(f"{'scheme':20s}{'Pres[dB]':>10s}{'SoI/SI':>9s}{'Ru':>7s}{'Rd':>7s}{'Gt':>9s}{'Gr':>7s}")
    for k, v in rows.items():
        m = v["metrics"]
        print(f"{k:20s}{m['res_SI_dB']:10.1f}{m['soi_si_dB']:9.1f}{m['Ru']:7.2f}{m['Rd']:7.2f}{m['Gt']:9.1f}{m['Gr']:7.2f}")
    Rdl, Rul = hd_rates(ch, cfg)
    print("half-duplex: Rdl, Rul =", round(Rdl, 2), round(Rul, 2), " Rhd =", round(half_duplex_sumrate(Rdl, Rul, 0.5), 2))
