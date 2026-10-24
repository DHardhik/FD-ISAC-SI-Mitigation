"""Range-Doppler and angle sensing. Eqs (53),(54), Sec. V-D, Figs 8-9."""
import numpy as np
from ii_channels import steering

C_LIGHT = 3e8


def qpsk(n, rng):
    """Unit-energy i.i.d. QPSK symbols (Sec. V)."""
    return (rng.choice([-1.0, 1.0], n) + 1j * rng.choice([-1.0, 1.0], n)) / np.sqrt(2)


def block_duration(cfg, N=None):
    """One block of N symbols lasts N*Ts."""
    return (cfg.N_fast if N is None else N) * cfg.Ts


def range_bin(cfg):
    """Round-trip delay in samples: i_tau = round(2 r / (c Ts)). (r=7.5 m, Ts=50 ns -> i_tau = 1)."""
    return int(round(2 * cfg.target[1] / (C_LIGHT * cfg.Ts)))


def simulate_rx_stream(ch, w, p, wu, cfg, N=None, M=None, rng=None, include_SI=True):
    """Discrete-time stream after the RX beamformer, Eq (53). Returns (s_hat [N x M], sd [N x M]).

        s_hat[n,m] = w^H Hu wu su[n,m]                         (uplink data; ~uncorrelated with sd)
                   + eta_r e^{j 2 pi fd (N Ts) m} (w^H a(th_r)) (b^T(th_r) p) sd[n - i_tau, m]   (radar echo)
                   + w^H Hsi p sd[n, m]                        (residual SI, zero delay)   [include_SI]
                   + w^H nu[n,m]                               (noise, variance sigma_u^2 ||w||^2)

    DOPPLER PHASE: Eq (53) prints e^{j2pi (fd Ts) m}; the phase must advance per BLOCK, i.e. by fd*(N*Ts)*m.
    With this scaling the velocity grid is v_j = j*lambda/(2 M N Ts) = j*2.39 m/s, so the true 20 m/s falls
    between bins 8 (19.07 m/s) and 9 (21.46 m/s); Fig. 8 reports 21.46, you may get either neighbour.
    Note: Hr already contains eta_r and a b^T, so the echo coefficient is w^H Hr(t=0) p."""
    N = cfg.N_fast if N is None else N
    M = cfg.M_slow if M is None else M
    rng = np.random.default_rng() if rng is None else rng
    sd = np.stack([qpsk(N, rng) for _ in range(M)], axis=1)
    su = np.stack([qpsk(N, rng) for _ in range(M)], axis=1)
    i_tau = range_bin(cfg)
    T_blk = block_duration(cfg, N)
    echo_coef = np.vdot(w, ch["Hr"] @ p)
    doppler = np.exp(1j * 2 * np.pi * cfg.doppler_hz * T_blk * np.arange(M))[None, :]
    s_hat = np.vdot(w, ch["Hu"] @ wu) * su + echo_coef * doppler * np.roll(sd, i_tau, axis=0)
    if include_SI:
        s_hat = s_hat + np.vdot(w, ch["Hsi"] @ p) * sd
    noise = (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / np.sqrt(2)
    s_hat = s_hat + np.sqrt(cfg.sigma2_u) * np.linalg.norm(w) * noise
    return s_hat, sd


def range_doppler_map(s_hat, sd, cfg, n_range=64):
    """Matched filter along FAST time n, then M-point DFT along SLOW time m (Sec. V-D).
    R[k,m] = sum_n s_hat[n,m] conj(sd[n-k,m])  (circular cross-correlation via FFT), then FFT over m.
    Returns (power_map [n_range x M] with Doppler axis ordered 0..M-1, range_axis [m], velocity_axis [m/s])
    where velocity_axis[j] is the velocity of Doppler bin j (negative velocities wrap to the end)."""
    N, M = s_hat.shape
    R = np.fft.ifft(np.fft.fft(s_hat, axis=0) * np.conj(np.fft.fft(sd, axis=0)), axis=0)   # correlation over n
    RD = np.fft.fft(R[:n_range, :], axis=1) / (N * M)                                      # DFT over m
    T_blk = N * cfg.Ts
    v_axis = np.fft.fftfreq(M, d=T_blk) * cfg.lam / 2
    r_axis = np.arange(n_range) * C_LIGHT * cfg.Ts / 2
    return np.abs(RD) ** 2, r_axis, v_axis


def estimate_peak(power_map, r_axis, v_axis):
    """(r_hat, v_hat) of the strongest cell. Bin resolution: dr = c Ts/2 = 7.5 m, dv = lambda/(2 M N Ts)."""
    k, j = np.unravel_index(np.argmax(power_map), power_map.shape)
    return float(r_axis[k]), float(v_axis[j]), (int(k), int(j))


def to_velocity_positive(power_map, v_axis, v_max=60.0):
    """Reorder to increasing velocity and crop to [0, v_max] for plotting like Fig. 8 (rows: Doppler, cols: range)."""
    order = np.argsort(v_axis)
    v_sorted, P = v_axis[order], power_map[:, order]
    sel = (v_sorted >= 0) & (v_sorted <= v_max)
    return P[:, sel].T, v_sorted[sel]          # shape [velocity x range]


def angle_spectrum(w, Hsi, p, thetas_deg, Nr):
    """Eq (54) as printed: P(theta) = |w^H (a(theta) + Hsi p)|^2 -- the receive pattern plus the residual-SI leakage.
    Returns linear power for each angle. Use normalise_dB() for Fig. 9."""
    leak = np.vdot(w, Hsi @ p)
    return np.array([abs(np.vdot(w, steering(t, Nr)) + leak) ** 2 for t in thetas_deg])


def normalise_dB(P):
    return 10 * np.log10(P / P.max() + 1e-300)


if __name__ == "__main__":
    from i_config import Cfg
    from ii_channels import draw_all
    from v_algorithm import solve_fd_isac, alphas_from_varrho
    from vii_baselines import radar_only, comm_only, nsp_design, complete_baseline
    cfg = Cfg()
    ch = draw_all(cfg, 60, np.random.default_rng(1))
    print(f"i_tau = {range_bin(cfg)} (range bin), fd = {cfg.doppler_hz} Hz")
    prop = solve_fd_isac(ch, cfg, a=alphas_from_varrho(1.0), rng=np.random.default_rng(2))
    designs = {"proposed": prop, "NSP": complete_baseline(ch, cfg, *nsp_design(ch, cfg)),
               "radar-only": complete_baseline(ch, cfg, *radar_only(cfg)),
               "comm-only": complete_baseline(ch, cfg, *comm_only(cfg))}
    th = np.linspace(-90, 90, 721)
    for name, d in designs.items():
        s, sd = simulate_rx_stream(ch, d["w"], d["p"], d["wu"], cfg, rng=np.random.default_rng(5))
        P, r_ax, v_ax = range_doppler_map(s, sd, cfg)
        r_hat, v_hat, (k, j) = estimate_peak(P, r_ax, v_ax)
        P_ang = angle_spectrum(d["w"], ch["Hsi"], d["p"], th, cfg.Nr)
        print(f"{name:11s}: peak at r = {r_hat:5.1f} m, v = {v_hat:6.2f} m/s | angle argmax = {th[np.argmax(P_ang)]:6.1f} deg")
