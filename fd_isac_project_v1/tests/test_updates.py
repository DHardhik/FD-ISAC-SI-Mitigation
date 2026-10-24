"""Verify the closed-form BCD updates against independent numerical checks.

Run from the project folder (the one containing config.py):
    python -m pytest -v            (or)   python tests/test_updates.py

Module-name tolerant: if you renamed files (e.g. updates.py -> iv_updates.py, algorithm.py -> v_algorithm.py)
the helper `_imp` below tries the alternatives.
Project module names: i_config, ii_channels, iii_metrics, iv_updates, v_algorithm, vi_sensing, vii_baselines.
"""
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _imp(*names):
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"none of {names} found in {ROOT}")


config = _imp("i_config", "config")
channels = _imp("ii_channels", "channels")
metrics = _imp("iii_metrics", "metrics")
upd = _imp("iv_updates", "updates")
alg = _imp("v_algorithm", "algorithm")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def cr(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def make_instance(seed=0, si_dB=40.0, beta=1e-3, alphas=(1.0, 1.0, 1.0, 1.0)):
    """Random channel + random feasible variables + FIXED rho (as inside one BCD sweep)."""
    cfg = config.Cfg()
    rng = np.random.default_rng(seed)
    ch = channels.draw_all(cfg, si_dB, rng)
    p = cr(rng, cfg.Nt); p *= np.sqrt(cfg.Pd) / np.linalg.norm(p)
    w = cr(rng, cfg.Nr); w /= np.linalg.norm(w)
    wu = cr(rng, cfg.Nu); wu *= np.sqrt(cfg.Pu) / np.linalg.norm(wu)
    ud = cr(rng, cfg.Nd)
    ru = upd.update_rho(metrics.mse_bs(w, p, wu, ch["Hu"], ch["H"], cfg.sigma2_u))
    rd = upd.update_rho(metrics.mse_dl(ud, p, ch["Hd"], cfg.sigma2_d))
    Zt = upd.Zt_matrix(cfg.target[0], cfg.Nt)
    Zr = upd.Zr_matrix(cfg.target[0], cfg.Nr)
    return SimpleNamespace(cfg=cfg, rng=rng, ch=ch, p=p, w=w, wu=wu, ud=ud, ru=ru, rd=rd,
                           Zt=Zt, Zr=Zr, beta=beta, al=alphas)


def F(I, p=None, w=None, wu=None, ud=None):
    """Objective (23) with rho FIXED at I.ru / I.rd (what each BCD block really optimises)."""
    return metrics.objective_23(I.p if p is None else p, I.w if w is None else w,
                                I.wu if wu is None else wu, I.ud if ud is None else ud,
                                I.ru, I.rd, I.ch, I.cfg, I.al, beta=I.beta)


def num_grad(f, x):
    """Central-difference gradient w.r.t. [Re x_k, Im x_k] interleaved."""
    eps = 1e-6 * max(np.linalg.norm(x), 1e-12)
    g = np.zeros(2 * len(x))
    for k in range(len(x)):
        for j, d in enumerate((1.0, 1j)):
            e = np.zeros(len(x), complex); e[k] = d * eps
            g[2 * k + j] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


def interleave(x):
    out = np.zeros(2 * len(x)); out[0::2] = x.real; out[1::2] = x.imag
    return out


# ----------------------------------------------------------------------------------------------
# Z matrices
# ----------------------------------------------------------------------------------------------
def test_Z_matrices_nsd_and_null_along_steering():
    cfg = config.Cfg()
    for Z, N in ((upd.Zt_matrix(cfg.target[0], cfg.Nt), cfg.Nt), (upd.Zr_matrix(cfg.target[0], cfg.Nr), cfg.Nr)):
        assert np.linalg.eigvalsh(Z).max() < 1e-9               # negative semi-definite
        a = channels.steering(cfg.target[0], N)
        assert np.linalg.norm(Z @ a) < 1e-9 * N * np.linalg.norm(a)   # null along the steering vector


# ----------------------------------------------------------------------------------------------
# rate <-> MSE (WMMSE) equivalence
# ----------------------------------------------------------------------------------------------
def test_rate_mse_equivalence():
    """With rho = 1/E:  -ln E <= ln(1+SINR) always, with EQUALITY at the MMSE receiver. Eqs (18)-(20)."""
    I = make_instance(1)
    cfg, ch = I.cfg, I.ch
    Hd, Hu, H = ch["Hd"], ch["Hu"], ch["H"]
    # downlink: random ud -> strict inequality; MMSE ud -> equality
    E = metrics.mse_dl(I.ud, I.p, Hd, cfg.sigma2_d); g = metrics.sinr_dl(I.ud, I.p, Hd, cfg.sigma2_d)
    assert -np.log(E) <= np.log1p(g) + 1e-12
    u = upd.update_ud(I.p, Hd, cfg.sigma2_d)
    E = metrics.mse_dl(u, I.p, Hd, cfg.sigma2_d); g = metrics.sinr_dl(u, I.p, Hd, cfg.sigma2_d)
    assert -np.log(E) == pytest.approx(np.log1p(g), rel=1e-9)
    # uplink: MMSE w = (Hu wu wu^H Hu^H + H p p^H H^H + s2 I)^{-1} Hu wu
    hu, hp = Hu @ I.wu, H @ I.p
    wm = np.linalg.solve(np.outer(hu, hu.conj()) + np.outer(hp, hp.conj()) + cfg.sigma2_u * np.eye(cfg.Nr), hu)
    E = metrics.mse_bs(wm, I.p, I.wu, Hu, H, cfg.sigma2_u); g = metrics.sinr_ul(wm, I.p, I.wu, Hu, H, cfg.sigma2_u)
    assert -np.log(E) == pytest.approx(np.log1p(g), rel=1e-9)
    # compact MSE == expanded Eq (18)
    w = I.w
    e18 = (np.vdot(w, hu) * np.vdot(hu, w)).real - 2 * np.vdot(w, hu).real \
        + (np.vdot(w, hp) * np.vdot(hp, w)).real + cfg.sigma2_u * np.vdot(w, w).real + 1
    assert metrics.mse_bs(w, I.p, I.wu, Hu, H, cfg.sigma2_u) == pytest.approx(e18, rel=1e-9)


# ----------------------------------------------------------------------------------------------
# block: ud
# ----------------------------------------------------------------------------------------------
def test_ud_is_mmse():
    I = make_instance(2)
    u = upd.update_ud(I.p, I.ch["Hd"], I.cfg.sigma2_d)
    E0 = metrics.mse_dl(u, I.p, I.ch["Hd"], I.cfg.sigma2_d)
    for _ in range(200):
        du = 1e-3 * np.linalg.norm(u) * cr(I.rng, I.cfg.Nd)
        assert metrics.mse_dl(u + du, I.p, I.ch["Hd"], I.cfg.sigma2_d) >= E0 - 1e-15
    # equals the paper's Eq (37) with explicit inverse
    Hdp = I.ch["Hd"] @ I.p
    u37 = np.linalg.solve(np.outer(Hdp, Hdp.conj()) + I.cfg.sigma2_d * np.eye(I.cfg.Nd), Hdp)
    assert np.allclose(u, u37, rtol=1e-8)


# ----------------------------------------------------------------------------------------------
# block: wu
# ----------------------------------------------------------------------------------------------
def test_wu_kkt():
    I = make_instance(3)
    cfg = I.cfg
    wu = upd.update_wu(I.w, I.ch["Hu"], I.ru, I.al[0], cfg.Pu)
    n2 = np.linalg.norm(wu) ** 2
    assert n2 <= cfg.Pu * (1 + 1e-9)                                  # feasible
    g = I.ch["Hu"].conj().T @ I.w
    # either constraint active or the MSE term is exactly zero (mu = 0 solution)
    assert n2 == pytest.approx(cfg.Pu, rel=1e-9) or abs(np.vdot(g, wu) - 1) < 1e-9
    f0 = F(I, wu=wu)
    for _ in range(300):
        cand = wu + 1e-3 * np.linalg.norm(wu) * cr(I.rng, cfg.Nu)
        cand *= min(1.0, np.sqrt(cfg.Pu) / np.linalg.norm(cand))       # project back into the feasible ball
        assert F(I, wu=cand) <= f0 + 1e-9 * abs(f0)


# ----------------------------------------------------------------------------------------------
# block: w
# ----------------------------------------------------------------------------------------------
def test_w_stationarity():
    """Gradient of the w-block objective is ~0 at the closed-form w (moderate beta so finite differences are sane)."""
    I = make_instance(4, beta=1e-3)
    w = upd.update_w(I.p, I.wu, I.ch["Hu"], I.ch["H"], I.ch["Hsi"], I.ru, I.al[0], I.al[3], I.Zr, I.beta, I.cfg.sigma2_u)
    g0 = np.linalg.norm(num_grad(lambda x: F(I, w=x), w))
    g1 = np.linalg.norm(num_grad(lambda x: F(I, w=x), w + 0.1 * cr(I.rng, I.cfg.Nr)))
    assert g0 < 1e-5 * g1


def test_w_matches_paper_eq34():
    """Compare the Sherman-Morrison implementation with a literal transcription of Eqs (34)-(35) (moderate beta)."""
    I = make_instance(5, beta=1e-3)
    ch, cfg = I.ch, I.cfg
    Hu, H, Hsi = ch["Hu"], ch["H"], ch["Hsi"]
    a1, a4, ru, b = I.al[0], I.al[3], I.ru, I.beta
    Xu = a1 * ru * (np.outer(Hu @ I.wu, (Hu @ I.wu).conj()) + np.outer(H @ I.p, (H @ I.p).conj())
                    + cfg.sigma2_u * np.eye(cfg.Nr)) - a4 * I.Zr                       # Eq (35)
    Hsip = Hsi @ I.p
    w34 = np.linalg.solve(2 * Xu + (1 / b) * np.outer(Hsip, Hsip.conj()), 2 * a1 * ru * (Hu @ I.wu))  # Eq (34)
    w = upd.update_w(I.p, I.wu, Hu, H, Hsi, ru, a1, a4, I.Zr, b, cfg.sigma2_u)
    assert np.allclose(w, w34, rtol=1e-6, atol=1e-9 * np.linalg.norm(w34))


# ----------------------------------------------------------------------------------------------
# block: p
# ----------------------------------------------------------------------------------------------
def test_p_power_constraint_and_stationarity():
    """(a) ||p||^2 = Pd  (b) Lagrangian gradient = 0 with the returned Gamma  (c) matches literal Eq (43)."""
    I = make_instance(6, beta=1e-3)
    ch, cfg = I.ch, I.cfg
    args = (I.w, I.ud, I.wu, ch["Hu"], ch["H"], ch["Hd"], ch["Hsi"], I.ru, I.rd, I.al[0], I.al[1], I.al[2],
            I.Zt, I.beta, cfg.Pd, cfg.sigma2_u)
    p, Gam = upd.update_p(*args, return_gamma=True)
    assert np.linalg.norm(p) ** 2 == pytest.approx(cfg.Pd, rel=1e-9)

    def lag_grad(x):   # gradient of  -F(p) + Gamma ||p||^2
        return -num_grad(lambda y: F(I, p=y), x) + 2 * Gam * interleave(x)
    g0 = np.linalg.norm(lag_grad(p))
    g1 = np.linalg.norm(num_grad(lambda y: F(I, p=y), p + 0.1 * np.linalg.norm(p) * cr(I.rng, cfg.Nt)))
    assert g0 < 1e-4 * g1

    # literal Eq (43) with the dimension typo of Eq (42) corrected
    a1, a2, a3, ru, rd, b = I.al[0], I.al[1], I.al[2], I.ru, I.rd, I.beta
    hw = ch["H"].conj().T @ I.w; hdu = ch["Hd"].conj().T @ I.ud; v = ch["Hsi"].conj().T @ I.w
    Xd = a1 * ru * np.outer(hw, hw.conj()) + a2 * rd * np.outer(hdu, hdu.conj()) - a3 * I.Zt
    p43 = np.linalg.solve(2 * Xd + (1 / b) * np.outer(v, v.conj()) + 2 * Gam * np.eye(cfg.Nt), 2 * a2 * rd * hdu)
    assert np.allclose(p, p43, rtol=1e-5, atol=1e-8 * np.linalg.norm(p43))


def test_p_projected_path_tiny_beta():
    """beta = 1e-25: p must be (numerically) orthogonal to Hsi^H w, on the power sphere, and locally optimal."""
    I = make_instance(7, beta=1e-25)
    ch, cfg = I.ch, I.cfg
    p = upd.update_p(I.w, I.ud, I.wu, ch["Hu"], ch["H"], ch["Hd"], ch["Hsi"], I.ru, I.rd,
                     I.al[0], I.al[1], I.al[2], I.Zt, I.beta, cfg.Pd, cfg.sigma2_u)
    v = ch["Hsi"].conj().T @ I.w
    assert np.linalg.norm(p) ** 2 == pytest.approx(cfg.Pd, rel=1e-9)
    assert abs(np.vdot(v, p)) < 1e-10 * np.linalg.norm(v) * np.linalg.norm(p)
    f0 = F(I, p=p)
    vn = v / np.linalg.norm(v)
    for _ in range(300):
        d = cr(I.rng, cfg.Nt); d -= vn * np.vdot(vn, d)                 # stay in v-perp
        cand = p + 1e-3 * np.linalg.norm(p) * d
        cand *= np.sqrt(cfg.Pd) / np.linalg.norm(cand)                   # stay on the sphere
        assert F(I, p=cand) <= f0 + 1e-9 * abs(f0)


# ----------------------------------------------------------------------------------------------
# whole algorithm
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("varrho", [1.0, 100.0, np.inf])
def test_monotone_objective(varrho):
    """On a random instance, f_g per iteration must be non-decreasing (Sec. IV-A)."""
    cfg = config.Cfg(max_iter=40)
    ch = channels.draw_all(cfg, 50, np.random.default_rng(11))
    out = alg.solve_fd_isac(ch, cfg, a=alg.alphas_from_varrho(varrho), rng=np.random.default_rng(12))
    f = out["history"]
    assert np.all(np.diff(f) >= -1e-9 * np.abs(f[:-1]))
    assert np.linalg.norm(out["p"]) ** 2 == pytest.approx(cfg.Pd, rel=1e-6)
    assert np.linalg.norm(out["w"]) == pytest.approx(1.0, rel=1e-9)
    assert np.linalg.norm(out["wu"]) ** 2 <= cfg.Pu * (1 + 1e-9)


def test_si_goes_down_with_beta():
    cfg = config.Cfg(max_iter=60)
    ch = channels.draw_all(cfg, 50, np.random.default_rng(21))
    res = []
    for beta in (1e-1, 1e-3, 1e-6, 1e-12, 1e-25):  # large beta: penalty negligible
        out = alg.solve_fd_isac(ch, cfg, a=alg.alphas_from_varrho(1.0), rng=np.random.default_rng(22), beta=beta)
        res.append(out["metrics"]["res_SI_dB"])
    print("residual SI [dB] vs beta:", np.round(res, 1))
    assert np.all(np.diff(res[1:]) < 0)      # once the penalty matters, smaller beta -> stronger SI suppression
    assert res[-1] < res[0] - 100
    assert res[-1] < -100                    # and far below the noise floor at beta = 1e-25


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
