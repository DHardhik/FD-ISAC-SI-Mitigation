"""Simulation constants for the FD-ISAC paper (Liu et al., JSAC 2023), Sec. V.

Tags:
  [PAPER]   value stated in the paper
  [DERIVED] follows directly from paper values
  [ASSUMED] NOT given in the paper -> standard choice, state it in your report

UNIT CONVENTION: all powers in mW, all angles in degrees (convert inside functions).
"""
from dataclasses import dataclass, field
import numpy as np

# Constants - (1)
C_LIGHT = 3e8  # m/s


def dbm2mw(x_dbm: float) -> float:
    return 10.0 ** (x_dbm / 10.0)


def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


@dataclass
class Cfg:
    # ---------------- Array / system sizes (5)----------------
    Nt: int = 16                 # [PAPER] TX antennas at ISAC transceiver
    Nr: int = 16                 # [PAPER] RX antennas at ISAC transceiver
    Nu: int = 2                  # [PAPER] uplink user antennas
    Nd: int = 2                  # [PAPER] downlink user antennas
    delta_ant: float = 0.5       # [PAPER] ULA spacing in wavelengths

    # ---------------- Carrier / bandwidth (2)----------------
    fc: float = 2.4e9            # [PAPER] carrier frequency (Hz)
    bw: float = 20e6             # [PAPER] bandwidth Omega (Hz); Ts = 1/bw

    # ---------------- Powers (4)----------------
    Pd_dBm: float = 20.0         # [PAPER] max TX power at transceiver
    Pu_dBm: float = 10.0         # [PAPER] max TX power at uplink user
    noise_dBm: float = -94.0     # [PAPER] noise floor at ISAC receiver
                                 #   (= -174 + 10log10(20e6) + 7 dB NF -> NF = 7 dB)
    # [ASSUMED] downlink user has the same noise floor as the ISAC receiver
    noise_dl_dBm: float = -94.0

    # ---------------- Path loss (2)----------------
    n_pl: float = 2.2            # [PAPER] path-loss exponent
    d0: float = 1.0              # [PAPER] reference distance (m)

    # ---------------- Geometry: (theta [deg], range [m], velocity [m/s]) (4)----------------
    target: tuple = (45.0, 7.5, 20.0)      # [PAPER] contains angle, distance, speed
    ul_user: tuple = (-50.0, 10.0, 0.0)    # [PAPER]
    dl_user: tuple = (-30.0, 100.0, 0.0)   # [PAPER]
    theta_si: float = 0.0        # [DERIVED] Fig. 9 text: "the angle 0 deg, where the SI locates"

    # ---------------- Rician / RCS (NOT in paper) (5)----------------
    kappa_comm: float = 10.0     # [v3 ASSUMED, v2: 1.0] Rician K for Hu, Hd (linear; 1 = 0 dB, LoS & NLoS equal power)
    kappa_si: float = 10.0       # [ASSUMED] Rician K for Hsi (linear; 10 = 10 dB). Near-field SI is LoS-dominated
                                 #   (Duarte et al. [19] measure strong K); keep >= kappa_comm
    comm_extra_loss_dB: float = 20.0  # [v3 CALIBRATED] extra path loss on Hu, Hd only. 20 dB makes the half-duplex
                                      #   sum-rate ~10 bps/Hz, as in Fig. 7 of the paper (v2 default was 0).
    rcs: float = 1.0             # [ASSUMED] radar cross-section sigma_r (m^2); only used if echo_dB is None
    echo_dB: float | None = -32.0  # [v3 CALIBRATED] radar echo power per RX antenna, relative to the noise floor,
                                 #   for an unstructured p with ||p||^2 = Pd (same convention as the SI level).
                                 #   Derivation (paper footnote 1): SI/echo = 134.15 dB before, and <= 80 dB after
                                 #   the analog cancellation; with the largest digital SI of 60 dB above noise this
                                 #   gives an echo of 60 - 80 = -20 dB when the TX beam points at the target
                                 #   (array gain Nt = 12 dB), i.e. -32 dB for an unstructured p.
                                 #   v2 used the radar equation with RCS = 1 m^2 at 7.5 m, which made the echo
                                 #   ~+20 dB above noise per antenna -> the uplink receiver nulled the target.
                                 #   Set echo_dB=None to go back to the RCS formula (Eq. 9).

    # ---------------- SI level (Sec. V-A sweeps) (2)----------------
    si_dB: float = 60.0          # [PAPER] SI power above noise floor, swept 10..60 dB
    si_sweep_dB: tuple = (20, 25, 30, 35, 40, 45, 50, 55, 60)   # [DERIVED] x-axis of Figs. 3, 4, 7
    # [ASSUMED] meaning of "SI power": total SI power per RX antenna, with ||p||^2 = Pd,
    #   equals noise + si_dB. See eta_si() below for how it fixes the SI large-scale gain.

    # ---------------- Algorithm parameters (14)----------------
    beta: float = 1e-25          # [PAPER] penalty parameter (Algorithm 1, line 1)
    solver: str = "v3"           # [v3] how the w- and p-blocks (Eqs 34, 43) are solved:
                                 #   "v3"    : exact Sherman-Morrison solve of the penalised equations, with the
                                 #             penalty applied to the NORMALISED SI channel (see penalty_norm).
                                 #   "v2"    : v2's exact projection (residual SI pinned at the float64 floor for
                                 #             every varrho -> Figs. 3/4 curves collapse onto each other).
                                 #   "paper" : dense np.linalg.solve of Eqs (34),(43) exactly as printed. With
                                 #             beta = 1e-25 the matrix has condition number ~1e25+, so round-off
                                 #             corrupts the beamformers (noisy RX beampattern). Kept for study only.
    penalty_norm: bool = True    # [v3] penalty (1/(2 beta)) |w^H Gsi p|^2 on the unit-gain SI matrix Gsi = Hsi /
                                 #   sqrt(eta_si) instead of Hsi. This makes beta dimensionless and independent of the
                                 #   SI level, so the BCD suppresses the SI by a fixed DEPTH and the residual grows with
                                 #   the SI power, as in Fig. 3. Without it, 1/(2 beta) = 5e24 multiplies a quantity
                                 #   whose size depends on the (unstated) unit system.
    beta_v3: float = 1e-6        # [v3 CALIBRATED] beta used with penalty_norm=True. At SI = 60 dB it gives about
                                 #   -99 / -199 dB residual SI for varrho = 0.01 / 100 (Fig. 5 channel); the paper
                                 #   quotes -108.48 / -197.37 dB in Sec. V-B.
    radar_convention: str = "alpha_com=1"  # [v3 ASSUMED] varrho = alpha_com/alpha_radar is a RATIO; v3 fixes
                                 #   alpha_com = 1 and sets alpha_radar = 1/varrho (v2 fixed alpha_radar = 1).
                                 #   Only the overall scale of the objective changes, so the beamformers are the same,
                                 #   but the penalty is then relatively stronger for communication-heavy designs, which
                                 #   gives the varrho ordering of Fig. 3 (varrho = 0.01 worst, inf best).
    kappa_radar: float = 16.0    # [v3 ASSUMED] scale of the radar weights: alpha_3 = kappa/(Nt Pd), alpha_4 =
                                 #   kappa * s_w / Nr. kappa = Nt = 16 is the paper's alpha_radar = 1 with Gt, Gr
                                 #   measured for unit-power beamformers (max 16 = 12 dBi). v2 used kappa = 1.
    varrho: float = 1.0          # [PAPER] alpha_com / alpha_radar
    varrho_list: tuple = (0.01, 1, 10, 100, 1e3)   # [DERIVED] values used in Figs. 3-5, 7
    tol: float = 1e-5            # [PAPER] epsilon in Fig. 2
    max_iter: int = 100          # [ASSUMED] safety cap (paper converges in ~25 its)
    bisect_iters: int = 60       # [ASSUMED] iterations for mu / Gamma bisection (I1, I2)
    bisect_tol: float = 1e-12    # [ASSUMED]
    n_mc: int = 1000             # [PAPER] Monte Carlo trials (use ~50 while debugging)
    seed: int = 0                # [ASSUMED]

    # ---------------- Sensing (Sec. V-D) (3)----------------
    N_fast: int = 1024           # [PAPER] symbols per block (fast time)
    M_slow: int = 512            # [PAPER] number of blocks (slow time)
    delta_hd: float = 0.5        # [PAPER] half-duplex time-frequency split

    # ======================= derived helpers (11)=======================
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
        return db2lin(-(self.pathloss_db(d) + self.comm_extra_loss_dB))

    def eta_radar(self) -> float:
        """Amplitude factor eta_r of Eq. (8).
        v3 (default, echo_dB set): eta_r^2 * Pd = noise * 10^(echo_dB/10), i.e. the echo per RX antenna for an
        unstructured p is echo_dB above the noise floor (see the echo_dB comment).
        echo_dB = None: radar equation, Eq. (9): sqrt(lam^2 sigma_r / ((4 pi)^3 r^4)) (v1/v2 behaviour)."""
        if self.echo_dB is not None:
            return float(np.sqrt(self.sigma2_u * db2lin(self.echo_dB) / self.Pd))
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
