"""Channel models. Eqs (5)-(10).

Conventions (match the paper):
  * Hu  : Nr x Nu   (uplink user -> transceiver RX)
  * Hd  : Nd x Nt   (transceiver TX -> downlink user)
  * Hr  : Nr x Nt   (radar echo, mono-static: AoA = AoD = theta_r)
  * Hsi : Nr x Nt   (TX -> RX self-interference, same Rician structure as Gu)
  * H = Hr + Hsi
All large-scale gains come from Cfg (units: mW powers, so |H p|^2 is in mW).
"""
import numpy as np


def steering(theta_deg: float, N: int, delta: float = 0.5) -> np.ndarray:
    """ULA steering vector a(theta) in C^N: [1, e^{j2 pi delta sin(th)}, ..., e^{j2 pi (N-1) delta sin(th)}]^T.
    Eq (7) items 2-3. The same function is used for b(.) (just pass Nt / Nu / Nd)."""
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * delta * n * np.sin(np.deg2rad(theta_deg)))


def pathloss_dB(d: float, lam: float, n: float, d0: float = 1.0) -> float:
    """eta(d) = -20log10(lam/(4 pi d0)) + 10 n log10(d/d0)  [dB, a LOSS -> positive number]. Sec V."""
    return -20 * np.log10(lam / (4 * np.pi * d0)) + 10 * n * np.log10(d / d0)


def rician(Nrx, Ntx, theta_rx, theta_tx, kappa, rng, delta: float = 0.5) -> np.ndarray:
    """Small-scale Rician matrix, Eq (6)-(7):
        G = sqrt(k/(k+1)) * a(th_rx) b^T(th_tx) + sqrt(1/(k+1)) * G_tilde,   [G_tilde]_ij ~ CN(0,1)
    Note b^T is a plain TRANSPOSE (no conjugate), so LoS = outer(a_rx, a_tx).
    E|G_ij|^2 = 1 for any kappa."""
    a_rx = steering(theta_rx, Nrx, delta)
    a_tx = steering(theta_tx, Ntx, delta)
    G_los = np.outer(a_rx, a_tx)
    G_nlos = (rng.standard_normal((Nrx, Ntx)) + 1j * rng.standard_normal((Nrx, Ntx))) / np.sqrt(2)
    return np.sqrt(kappa / (kappa + 1)) * G_los + np.sqrt(1 / (kappa + 1)) * G_nlos


def comm_channel(cfg, user, rng, Nrx, Ntx, uplink: bool) -> np.ndarray:
    """H = sqrt(eta) * G. Eq (5).  user = (theta_deg, range_m, velocity).
    ASSUMPTION: AoD at the user side equals AoA at the transceiver (theta'_u = theta_u), i.e. the
    user and transceiver see each other along the same bearing. `uplink` only documents which
    side is the transceiver (Nrx for uplink, Ntx for downlink); the geometry is symmetric."""
    theta, d, _ = user
    eta = cfg.eta_comm(d)                       # linear POWER gain
    G = rician(Nrx, Ntx, theta, theta, cfg.kappa_comm, rng, cfg.delta_ant)
    return np.sqrt(eta) * G


def radar_channel(cfg, t: float = 0.0) -> np.ndarray:
    """Hr(t) = eta_r e^{j 2 pi fd t} a(th_r) b^T(th_r), Nr x Nt. Eqs (8),(9), fd = 2 v fc / c.
    eta_r is an AMPLITUDE (Eq 9 has a sqrt)."""
    th_r = cfg.target[0]
    a = steering(th_r, cfg.Nr, cfg.delta_ant)
    b = steering(th_r, cfg.Nt, cfg.delta_ant)
    return cfg.eta_radar() * np.exp(1j * 2 * np.pi * cfg.doppler_hz * t) * np.outer(a, b)


def si_channel(cfg, si_power_dB_over_noise: float, rng) -> np.ndarray:
    """Hsi = sqrt(eta_si) * Gsi, Eq (10). Gsi is Rician (same structure as Gu) with LoS at theta_si = 0 deg
    and K = cfg.kappa_si.

    SCALING (state this in your report): eta_si is chosen so that if the TX vector p has
    ||p||^2 = Pd and 'typical' (unstructured) entries, the SI power per RX antenna is
    `si_power_dB_over_noise` dB above the noise floor. Because the LoS part is rank-1, a p that is
    aligned with b(theta_si) can see up to Nt x more SI -- that is precisely what the algorithm must avoid."""
    eta = cfg.eta_si(si_power_dB_over_noise)
    G = rician(cfg.Nr, cfg.Nt, cfg.theta_si, cfg.theta_si, cfg.kappa_si, rng, cfg.delta_ant)
    return np.sqrt(eta) * G


def draw_all(cfg, si_dB, rng):
    """One Monte-Carlo channel realisation. Returns dict(Hu, Hd, Hr, Hsi, H=Hr+Hsi)."""
    Hu = comm_channel(cfg, cfg.ul_user, rng, cfg.Nr, cfg.Nu, uplink=True)    # Nr x Nu
    Hd = comm_channel(cfg, cfg.dl_user, rng, cfg.Nd, cfg.Nt, uplink=False)   # Nd x Nt
    Hr = radar_channel(cfg)
    Hsi = si_channel(cfg, si_dB, rng)
    return dict(Hu=Hu, Hd=Hd, Hr=Hr, Hsi=Hsi, H=Hr + Hsi)


if __name__ == "__main__":
    # quick self-checks: python channels.py
    from i_config import Cfg
    cfg = Cfg()
    rng = np.random.default_rng(cfg.seed)
    ch = draw_all(cfg, 60, rng)
    print({k: v.shape for k, v in ch.items()})
    # 1) E|G_ij|^2 = 1
    Gs = np.stack([rician(8, 8, 20, -10, cfg.kappa_si, rng) for _ in range(2000)])
    print("E|G|^2 (should be ~1):", np.mean(np.abs(Gs) ** 2).round(3))
    # 2) fd = 320 Hz
    print("Doppler [Hz]:", cfg.doppler_hz)
    # 3) link budget flavour: SI vs echo per-RX-antenna power with ||p||^2=Pd, p = b(theta_r)/||b||*sqrt(Pd)
    p = steering(cfg.target[0], cfg.Nt) / np.sqrt(cfg.Nt) * np.sqrt(cfg.Pd)
    si = np.mean(np.abs(ch["Hsi"] @ p) ** 2); echo = np.mean(np.abs(ch["Hr"] @ p) ** 2)
    print("SI/echo before beamforming [dB]:", round(10 * np.log10(si / echo), 1), "(paper link budget: 134.15)")
    print("SI above noise [dB]:", round(10 * np.log10(si / cfg.sigma2_u), 1))
