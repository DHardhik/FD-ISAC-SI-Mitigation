"""Simulation constants for the FD-ISAC paper (Liu et al., JSAC 2023), Sec. V.

Tags:
  [PAPER]   value stated in the paper
  [DERIVED] follows directly from paper values
  [ASSUMED] NOT given in the paper -> standard choice, state it in your report

UNIT CONVENTION: all powers in mW, all angles in degrees (convert inside functions).
"""
from dataclasses import dataclass, field
import numpy as np

C_LIGHT = 3e8  # m/s


def dbm2mw(x_dbm: float) -> float:
    return 10.0 ** (x_dbm / 10.0)


def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


@dataclass
class Cfg:
    # ---------------- Array / system sizes ----------------
    Nt: int = 16                 # [PAPER] TX antennas at ISAC transceiver
    Nr: int = 16                 # [PAPER] RX antennas at ISAC transceiver
    Nu: int = 2                  # [PAPER] uplink user antennas
    Nd: int = 2                  # [PAPER] downlink user antennas
    delta_ant: float = 0.5       # [PAPER] ULA spacing in wavelengths

    # ---------------- Carrier / bandwidth ----------------
    fc: float = 2.4e9            # [PAPER] carrier frequency (Hz)
    bw: float = 20e6             # [PAPER] bandwidth Omega (Hz); Ts = 1/bw

    # ---------------- Powers ----------------
    Pd_dBm: float = 20.0         # [PAPER] max TX power at transceiver
    Pu_dBm: float = 10.0         # [PAPER] max TX power at uplink user
    noise_dBm: float = -94.0     # [PAPER] noise floor at ISAC receiver
                                 #   (= -174 + 10log10(20e6) + 7 dB NF -> NF = 7 dB)
    # [ASSUMED] downlink user has the same noise floor as the ISAC receiver
    noise_dl_dBm: float = -94.0

    # ---------------- Path loss ----------------
    n_pl: float = 2.2            # [PAPER] path-loss exponent
    d0: float = 1.0              # [PAPER] reference distance (m)

    # ---------------- Geometry: (theta [deg], range [m], velocity [m/s]) ----------------
    target: tuple = (45.0, 7.5, 20.0)      # [PAPER]
    ul_user: tuple = (-50.0, 10.0, 0.0)    # [PAPER]
    dl_user: tuple = (-30.0, 100.0, 0.0)   # [PAPER]
    theta_si: float = 0.0        # [DERIVED] Fig. 9 text: "the angle 0 deg, where the SI locates"

    # ---------------- Rician / RCS (NOT in paper) ----------------
    kappa_comm: float = 1.0      # [ASSUMED] Rician K for Hu, Hd (linear; 1 = 0 dB, LoS & NLoS equal power)
    kappa_si: float = 10.0       # [ASSUMED] Rician K for Hsi (linear; 10 = 10 dB). Near-field SI is LoS-dominated
                                 #   (Duarte et al. [19] measure strong K); keep >= kappa_comm
    rcs: float = 1.0             # [ASSUMED] radar cross-section sigma_r (m^2); common small-target value

    # ---------------- SI level (Sec. V-A sweeps) ----------------
    si_dB: float = 60.0          # [PAPER] SI power above noise floor, swept 10..60 dB
    si_sweep_dB: tuple = (20, 25, 30, 35, 40, 45, 50, 55, 60)   # [DERIVED] x-axis of Figs. 3, 4, 7
    # [ASSUMED] meaning of "SI power": total SI power per RX antenna, with ||p||^2 = Pd,
    #   equals noise + si_dB. See eta_si() below for how it fixes the SI large-scale gain.

    # ---------------- Algorithm parameters ----------------
    beta: float = 1e-25          # [PAPER] penalty parameter (Algorithm 1, line 1)
    varrho: float = 1.0          # [PAPER] alpha_com / alpha_radar
    varrho_list: tuple = (0.01, 1, 10, 100, 1e3)   # [DERIVED] values used in Figs. 3-5, 7
    tol: float = 1e-5            # [PAPER] epsilon in Fig. 2
    max_iter: int = 100          # [ASSUMED] safety cap (paper converges in ~25 its)
    bisect_iters: int = 60       # [ASSUMED] iterations for mu / Gamma bisection (I1, I2)
    bisect_tol: float = 1e-12    # [ASSUMED]
    n_mc: int = 1000             # [PAPER] Monte Carlo trials (use ~50 while debugging)
    seed: int = 0                # [ASSUMED]

    # ---------------- Sensing (Sec. V-D) ----------------
    N_fast: int = 1024           # [PAPER] symbols per block (fast time)
    M_slow: int = 512            # [PAPER] number of blocks (slow time)
    delta_hd: float = 0.5        # [PAPER] half-duplex time-frequency split

    # ======================= derived helpers =======================
    @property
    def lam(self) -> float:
        return C_LIGHT / self.fc

    @property
    def Ts(self) -> float:
        return 1.0 / self.bw

    @property
    def Pd(self) -> float:           # mW
        return dbm2mw(self.Pd_dBm)

    @property
    def Pu(self) -> float:           # mW
        return dbm2mw(self.Pu_dBm)

    @property
    def sigma2_u(self) -> float:     # mW, noise variance per RX antenna at transceiver
        return dbm2mw(self.noise_dBm)

    @property
    def sigma2_d(self) -> float:     # mW, noise variance per antenna at DL user
        return dbm2mw(self.noise_dl_dBm)

    @property
    def doppler_hz(self) -> float:
        """fd = 2 v fc / c  (paper: 320 Hz for v = 20 m/s). Sanity check this!"""
        return 2.0 * self.target[2] * self.fc / C_LIGHT

    def pathloss_db(self, d: float) -> float:
        """eta(d) [dB] = -20log10(lam/(4 pi d0)) + 10 n log10(d/d0). Returns a POSITIVE loss in dB
        (i.e. the paper's eta(d) as written is a loss; large-scale gain = 10^(-PL/10))."""
        return -20 * np.log10(self.lam / (4 * np.pi * self.d0)) + 10 * self.n_pl * np.log10(d / self.d0)

    def eta_comm(self, d: float) -> float:
        """Linear large-scale POWER gain for Hu / Hd (use sqrt() in H = sqrt(eta) G)."""
        return db2lin(-self.pathloss_db(d))

    def eta_radar(self) -> float:
        """Amplitude factor eta_r from Eq. (9): sqrt(lam^2 sigma_r / ((4 pi)^3 r^4))."""
        r = self.target[1]
        return float(np.sqrt(self.lam ** 2 * self.rcs / ((4 * np.pi) ** 3 * r ** 4)))

    def eta_si(self, si_dB: float | None = None) -> float:
        """Linear large-scale POWER gain for Hsi so that, with ||p||^2 = Pd,
        E[|[Hsi p]_i|^2] = noise * 10^(si_dB/10) per RX antenna.
        Assumes E|[Gsi]_ij|^2 = 1 (true for the Rician model in Eq. 6)
        => E|[Hsi p]_i|^2 = eta_si * Pd  =>  eta_si = noise*10^(si/10) / Pd."""
        s = self.si_dB if si_dB is None else si_dB
        return self.sigma2_u * db2lin(s) / self.Pd

    def __post_init__(self):
        assert self.Nt > 1 and self.Nr > 1
        assert abs(self.doppler_hz - 320.0) < 1.0 or self.target[2] != 20.0
