# FD-ISAC v3: Joint Transmit/Receive Beamforming in Full-Duplex ISAC

Python implementation of

> Z. Liu, S. Aditya, H. Li, B. Clerckx, "Joint Transmit and Receive Beamforming Design in Full-Duplex
> Integrated Sensing and Communications," *IEEE JSAC*, vol. 41, no. 9, pp. 2907-2919, Sep. 2023.

It covers the penalty-based BCD algorithm (Algorithm 1), the baselines (NSP, radar-only, communication-only,
half-duplex) and the experiments for Figs. 2, 3, 4, 5, 7, 8 and 9. Results in `results/` were produced with
**100 Monte-Carlo trials** (`python run_all.py --mc 100`).

---

## 0. Why v2 did not look like the paper, and what v3 changes

The number of Monte-Carlo trials was **not** the problem. 100 trials is enough to see the paper's trends. The
differences came from one real bug in the channel model and from several modelling choices the paper leaves open.
v3 fixes the bug and recalibrates the open choices against numbers that the paper does state.

### 0.1 Drawbacks of v2

| # | Drawback in v2 | Root cause (where v2 went wrong) | Effect on the v2 figures | v3 fix |
|---|---|---|---|---|
| 1 | **Transpose vs conjugate bug in the channels** | `ii_channels.rician` and `radar_channel` built the LoS part as `a(θ) b^T(θ)` (as Eqs. 7-8 are printed), but the beampattern metric is `|b^H(θ) p|²` (Eq. 15). With a plain transpose, the beam that serves a user at θ appears at **-θ** in the beampattern, and the radar-only beam `p = b(θr)` sees almost no echo (`b^T b ≈ 0`). | Fig. 5: the TX "DL lobe" sat near +30° instead of -30°. NSP's TX beam missed the DL user, so NSP's sum-rate was 3.6 bps/Hz, far **below** half-duplex (paper: NSP ≈ half-duplex). | LoS = `a(θrx) b^H(θtx)` everywhere, radar channel included. Test `test_los_channel_uses_conjugate_steering`. |
| 2 | **Radar echo far too strong** | Echo from the radar equation (Eq. 9) with RCS = 1 m² at 7.5 m: about **+28 dB above noise per antenna** (+52 dB after both beams). The uplink SINR (Eq. 12) treats the echo as interference, so the optimiser **nulled the target** in `w`. | Fig. 8/9: the proposed method failed to detect the target (detection rate **0.00**; peak at r = 30 m, v = -156 m/s). | Echo set from the paper's own link budget (footnote 1): SI/echo ≤ 80 dB after analog cancellation with SI ≤ 60 dB above noise gives an echo of about -20 dB with the TX beam on the target, i.e. `echo_dB = -32` for an unstructured `p`. Proposed detection rate is now 1.00. |
| 3 | **Residual SI pinned at the float64 floor** | `update_p` switched to an exact projection (`v^H p = 0`) whenever `c‖v‖²` was large, and `beta = 1e-25` always triggered it. The penalty then acts as a hard constraint and no trade-off is left. | Figs. 3 and 4: all ϱ curves on top of each other at -250 to -300 dB; the ϱ ordering of the paper is invisible. | Exact Sherman-Morrison solve **without** the projection switch, and the penalty applied to the **unit-gain** SI matrix `Gsi = Hsi/√η_si` (Section 0.2). The residual now reflects the real penalty trade-off: it rises with SI power and depends on ϱ. |
| 4 | **Penalty parameter had no physical meaning** | `1/(2β) = 5e24` multiplies `|w^H Hsi p|²`, whose size depends on the unit system (mW vs W, unnormalised `w`). The paper does not give its units. | Any β either does nothing or nulls exactly; nothing in between. | β is now dimensionless (normalised SI channel). `β = 1e-6` is calibrated so that, at 60 dB SI, ϱ = 0.01 and ϱ = 100 give about -99 / -199 dB (the paper quotes -108.48 / -197.37 dB in Sec. V-B). |
| 5 | **Radar weights too weak; wrong ϱ convention** | v2 used α_com = ϱ, α_radar = 1 with radar gains normalised to [0, 1] (κ = 1). The radar terms (at most 2) were tiny next to the rates (about 20 nats). | Fig. 7: ϱ = 1, 10, 100, 1000, ∞ all at 19.3 to 19.7 bps/Hz, essentially the comm-only bound. Fig. 3 ordering (if it had been visible) would have been reversed. | α_com = 1, α_radar = 1/ϱ, with Gt and Gr measured for unit-power beamformers (max 16 = 12 dBi), i.e. the paper's α_radar = 1 on a dBi scale (`kappa_radar = 16`). |
| 6 | **User links too scattered** | Rician K = 1 (0 dB) for Hu and Hd: half of the channel energy is NLoS, so any RX beam already collects the uplink. | Fig. 5: at ϱ = 1 the RX beam had no lobe at the UL user. | K = 10 (10 dB) for the user links (LoS-dominant, like the user lobes in Fig. 5). Still an assumption; see Section 4. |
| 7 | Calibration was opt-in | The calibrated set-up needed `--loss 20 --norm-radar --tag calibrated`; the defaults were the uncalibrated v1 values, and `results/` mixed both runs. | Easy to run the wrong configuration. | The calibrated set-up is the default. Every knob is still a flag (Section 2). |
| 8 | Figs. 3/4 and Fig. 7 ran the same sweep twice, sequentially | Two scripts drew the same channels separately. | About twice the run time. | One script, `exp_si_sweep.py`, produces Figs. 3, 4 and 7 from the same trials and runs on all CPU cores (`--jobs`). |
| 9 | **Latent bug in the p-update (hard case)** | The bisection for Γ was bounded by -λ_min(B0) instead of -λ_min(B0 + c v v^H). When the optimum lay in between, the code fell into a fake "hard case" and added power along a direction with a huge SI penalty. v2 hid this because the projection path was almost always taken. | Occasional non-monotone BCD steps (objective dropping by ~1e12), especially for ϱ = ∞ where B0 is singular. | True bound from the secular equation; hard-case power goes to an eigen-direction orthogonal to `Hsi^H w`. 0 non-monotone runs out of 240 in a stress test. |
| 10 | Cosmetic | Two NSP variants in every plot; Fig. 8 panel titles overlapped the axes; the y-axis of the proposed panel stretched to -156 m/s. | Cluttered figures. | One NSP curve (paper's description), constrained layout. |

### 0.2 What v3 does, in one paragraph

Algorithm 1 is unchanged: ρ → ω_u → w → u_d → p, each block in closed form (Eqs. 26, 27, 32, 34, 37, 43), with
‖p‖² = P_d and the final normalisation of `w` (Eq. 44). The changes are in the **model and calibration**:
(i) conjugate steering on the transmit side (bug fix); (ii) an echo level taken from the paper's link budget;
(iii) the SI penalty on the unit-gain SI channel with a calibrated β, solved exactly; (iv) α_com = 1,
α_radar = 1/ϱ with dBi-scaled radar gains; (v) LoS-dominant user links; (vi) a 20 dB extra user-link loss so that
half-duplex is about 10 bps/Hz as in Fig. 7 (already found in v2).

### 0.3 v3 results vs the paper (100 Monte-Carlo trials)

| Figure | Paper | v2 (100 trials) | v3 (100 trials) |
|---|---|---|---|
| Fig. 2 convergence | monotone, < 30 iterations | monotone, ~5 iterations | monotone in every run; mean 49 (Nt = 16, 60 dB) and 28 (Nt = 8, 20 dB) iterations |
| Fig. 3 residual SI | rises with SI power (about -300 to -110 dB); ϱ = 1 highest, ∞ lowest; NSP flat (~-200 dB) | every curve at the float floor (-250 to -300 dB), no ϱ separation | rises with SI power (slope ~1.9 dB/dB): from -280 to -256 dB at 20 dB SI up to -203 to -175 dB at 60 dB SI. Radar-heavy ϱ = 1 and 10 sit above ϱ = 100, 1000, ∞. NSP at the float floor |
| Fig. 4 SoI/SI | falls with SI power, 160-345 dB; comm-heavy ϱ highest | flat at the +300 dB ceiling | falls with SI power, 300 → 212-240 dB; ϱ = 100, 1000, ∞ above ϱ = 1, 10 |
| Fig. 5 beampatterns (60 dB SI) | ϱ = 1: TX target lobe + DL lobe ~10 dB lower, RX target + UL lobes. ϱ = 100: user lobes dominate. ϱ = 0.01: radar-only. Residual SI -131 / -197 / -108 dB | ϱ = 1, 100: TX "DL lobe" at the mirrored angle (+30°), no RX UL lobe at ϱ = 1 | ϱ = 1: TX DL lobe 11.7 dB below the target lobe, RX UL lobe 4.0 dB below. ϱ = 100: TX and RX peaks at the users. ϱ = 0.01: TX identical to radar-only. Residual SI -190 / -199 / -99 dB |
| Fig. 7 sum-rate | HD ≈ 10; NSP ≈ 10.1; ϱ = 1 ≈ 10.5-11, ϱ = 10 ≈ 12, ϱ = 100 ≈ 13, ϱ = 1e3 ≈ 14, ϱ = ∞ ≈ 14-16 | HD 9.9; NSP 3.6 (below HD); ϱ = 1 … ∞ all 19.3-19.7 | HD 10.2; NSP 12.5; ϱ = 1 12.6; ϱ = 10 19.7; ϱ = 100, 1e3, ∞ 20.4 |
| Fig. 8 range-velocity | proposed and NSP find 7.5 m / ~21 m/s; radar-only and comm-only stuck at the SI (0 m, 0 m/s) | proposed **failed** (detection 0.00, peak at 30 m / -156 m/s); NSP 1.00 | proposed 0.88, NSP 1.00, radar-only 0.00, comm-only 0.00 (peaks at 7.5 m / 19.7 m/s and 7.5 m / 20.3 m/s) |
| Fig. 9 angle | proposed peaks at the target with a distinct UL lobe; comm-only peaks at the UL user | proposed angle detection 0.01 | proposed 0.83, NSP 1.00, radar-only 1.00, comm-only 0.00; proposed shows the UL lobe at -50° |

### 0.4 Remaining differences (state these in your report)

1. **Fig. 7, spread across ϱ.** In v3 the ordering is right and ϱ = 1 sits just above NSP and half-duplex, as in the
   paper. But ϱ ≥ 10 saturates near the comm-only bound (≈ 2 × half-duplex), whereas the paper's curves rise slowly
   over three decades of ϱ and its comm-only FD rate is only ~1.6 × half-duplex. With bounded radar gains and
   logarithmic rates, the trade-off in this model happens within about one decade of ϱ. The paper's slower,
   SI-dependent decline (20 → 40 dB) also does not appear, because our residual SI is always far below the noise
   and so never costs rate.
2. **Fig. 3, ordering inside the radar-heavy and comm-heavy groups.** ϱ = 10 lies above ϱ = 1, and ϱ = 100, 1000, ∞
   are within a few dB of each other (the paper separates them by ~30 dB per decade). ϱ = 1 also has a wide 25-75 %
   band because it sits where the optimum switches between a radar-like and a mixed beam. NSP nulls `Hsi p` exactly,
   so its residual is the float64 floor (-300 → -261 dB), not the flat -200 dB line of the paper.
3. **Fig. 5, ϱ = 0.01 RX beam.** It keeps a lobe at the UL user (-2.6 dB) instead of being radar-only. Inside the BCD
   `w` is unnormalised (Eq. 44 normalises only at the end), so the radar term `α4 w^H Zr w` can be reduced by shrinking
   ‖w‖ rather than by turning the beam. This is a property of the paper's formulation; the TX side is radar-only as
   expected.
4. **Fig. 8, proposed detection 0.88 rather than ~1.** At ϱ = 1 the RX beam also serves the uplink user, and the
   uplink data then raises the floor of the range-Doppler map (visible in the panel). The paper argues this floor
   vanishes for large N; with N = 1024 it is still visible in some trials.
5. **Calibrated values are ours.** β = 1e-6 on the normalised SI channel, κ = 16, echo -32 dB, user-link K = 10 dB and
   the 20 dB extra loss are all chosen to match numbers the paper does state (Sec. V-B residuals, Fig. 5 lobes,
   footnote 1, Fig. 7 half-duplex level). They are listed in Section 4 and should be reported as assumptions.

---

## 1. Setup

Requires **Python 3.10 or newer**.

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt  # numpy, scipy, matplotlib, pytest
```

## 2. Running

Run everything from the project root (the folder that contains `i_config.py`).

| Goal | Command | Time (2 cores) |
|---|---|---|
| Check the maths | `python -m pytest -q` | about 1 min |
| Smoke test | `python run_all.py --mc 5` | about 3 min |
| Figures as delivered | `python run_all.py --mc 100` | about 27 min |
| Paper-level statistics | `python run_all.py --mc 1000` | hours |

Single experiments:

```bash
python experiments/exp_convergence.py --mc 100   # Fig. 2
python experiments/exp_beampattern.py            # Fig. 5   (one channel realisation)
python experiments/exp_sensing.py --mc 100       # Figs. 8, 9 + detection rates
python experiments/exp_si_sweep.py --mc 100      # Figs. 3, 4, 7 (parallel, --jobs N)
python experiments/exp_calibration.py --mc 10    # user-link loss sweep (half-duplex level)
```

Options accepted by every experiment (and passed through by `run_all.py`):

| Option | Default (v3) | Meaning |
|---|---|---|
| `--mc N` | per script | Monte-Carlo trials |
| `--solver` | `v3` | `v3`: exact solve, normalised SI penalty. `v2`: v2's exact projection (SI at the float floor). `paper`: dense `np.linalg.solve` of Eqs. (34), (43) as printed (see Section 3) |
| `--beta B` | 1e-6 (`v3`), 1e-25 (others) | penalty parameter |
| `--kappa K` | 16 | radar-weight scale (16 = α_radar 1 on a dBi scale) |
| `--echo E` | -32 | echo per RX antenna re. noise [dB]; `--echo nan` uses the RCS formula (v2) |
| `--loss L` | 20 | extra user-link path loss [dB] (v2 default was 0) |
| `--radar-weights` | `v3` | `v2` = v2's `--norm-radar`, `raw` = v1 |
| `--starts K` | 1 | random BCD starts (radar-only and comm-only warm starts are always added unless `--no-warm`) |
| `--max-iter N` | 100 | BCD iteration cap |
| `--jobs N` | all cores | worker processes for `exp_si_sweep.py` |
| `--tag NAME` | none | suffix for output files |
| `--show` | off | open plot windows |

To reproduce v2-like behaviour for comparison: `python experiments/exp_si_sweep.py --mc 20 --solver v2 --tag v2style`.

## 3. Note on the `paper` solver (why we do not use it by default)

Solving Eqs. (34) and (43) literally, with β = 1e-25 and a dense linear solve, gives residual-SI numbers that look
very much like the paper's Fig. 3 (e.g. ϱ = 1: -258 dB at 20 dB SI rising to -149 dB at 60 dB). But the matrix has a
condition number of about 1e25, so float64 round-off also corrupts the beamformers themselves: the RX beampattern
becomes noisy and loses its lobes. Those residual-SI values are therefore round-off, not a property of the
design. v3 instead obtains the same qualitative behaviour (residual grows with SI power, and is higher for smaller
ϱ) from an exact solve of a well-scaled penalty. The `paper` solver is kept so you can show this in a report.

## 4. Assumptions not given in the paper (see `i_config.py`)

| Quantity | v3 value | v2 value | Reason |
|---|---|---|---|
| Transmit-side steering in LoS / radar channel | `b^H` | `b^T` | consistency with Eq. (15) (bug fix) |
| Radar echo per RX antenna (unstructured p) | -32 dB re. noise | +28 dB (RCS 1 m²) | paper footnote 1 link budget |
| SI penalty | on `Gsi`, β = 1e-6 | on `Hsi`, β = 1e-25 (projection) | dimensionless β, calibrated to Sec. V-B values |
| ϱ convention | α_com = 1, α_radar = 1/ϱ | α_com = ϱ, α_radar = 1 | gives the Fig. 3 ordering |
| Radar gain scale | κ = 16 (dBi scale) | κ = 1 | ϱ = 1 gives both target and user lobes (Fig. 5a) |
| Rician K, user links | 10 dB | 0 dB | LoS-dominant user lobes (Fig. 5) |
| Rician K, SI link | 10 dB | 10 dB | unchanged |
| Extra user-link loss | 20 dB | 0 dB | half-duplex ≈ 10 bps/Hz (Fig. 7) |
| "SI power = X dB" | SI per RX antenna, unstructured p, X dB above noise | same | unchanged |
| NSP | 50/50 TX split (target, DL); RX steered at target, nulls on `Hsi p` and the DL angle | RX at target, also an angle-only variant | Sec. I description of [28]-[30] |
| Iteration cap / tolerance | 100 / 1e-5 | 200 / 1e-5 | paper converges in < 30 iterations |

## 5. Corrections to the printed equations (unchanged from v2)

- Eq. (30)/(32): the factor ρ_u is missing in the ω_u update.
- Eq. (33): α_4 appears twice. Eq. (42): `Hd ud ud^H Hd^H` must be `Hd^H ud ud^H Hd`, and `-α_3 Zt I` is `-α_3 Zt`.
- ‖p‖² = P_d must be an equality (Eq. 40); the bound (21) is tight only then.
- ρ* = 1/E is exact for the natural log, so the objective uses `ln` (rates are reported in bits).
- Eq. (53): the Doppler phase advances per block (`fd · N · Ts · m`), not per sample.
- Eqs. (7)-(8): the transmit-side steering vector must be conjugated (`b^H`), see drawback 1.

## 6. Project layout

```
i_config.py        constants (Sec. V) and every assumption, tagged [PAPER] / [ASSUMED] / [v3 ...]
ii_channels.py     steering vectors, Rician Hu/Hd/Hsi, radar channel          Eqs (5)-(10)
iii_metrics.py     rates, MSEs, beampattern gains, residual SI, objectives     Eqs (11)-(15), (18)-(19), (23)
iv_updates.py      closed-form BCD block updates (+ literal "paper" versions)  Eqs (26),(27),(32),(34),(37),(43)
v_algorithm.py     Algorithm 1, multi-start, alpha / beta conventions          Eqs (44),(51)
vi_sensing.py      range-Doppler map and angle spectrum                        Eqs (53),(54)
vii_baselines.py   NSP, radar-only, comm-only, half-duplex                     Sec. V
run_all.py         runs every experiment, logs to results/logs/
tests/test_updates.py   block-update optimality, monotonicity, and v3 checks
experiments/
    common.py            shared options and helpers
    exp_convergence.py   Fig. 2
    exp_beampattern.py   Fig. 5
    exp_sensing.py       Figs. 8, 9
    exp_si_sweep.py      Figs. 3, 4, 7
    exp_calibration.py   user-link loss sweep
results/           figures (.png), data (.npz), logs
```

## 7. Minimal usage

```python
import numpy as np
from i_config import Cfg
from ii_channels import draw_all
from v_algorithm import solve_multistart, alphas_from_varrho

cfg = Cfg()                                                  # v3 defaults
ch = draw_all(cfg, si_dB=60, rng=np.random.default_rng(0))
a = alphas_from_varrho(1.0, cfg, normalize_radar="v3", ch=ch)  # varrho = 1
out = solve_multistart(ch, cfg, a=a, rng=np.random.default_rng(1))
print(out["metrics"])        # Ru, Rd, Gt, Gr, res_SI_dB, soi_si_dB
```

## 8. Troubleshooting

- `ModuleNotFoundError: i_config`: run from the project root, not from inside `experiments/` or `tests/`.
- `[FAILED]` in `run_all.py`: open `results/logs/<script>.log`.
- On Windows, `exp_si_sweep.py` uses `multiprocessing`; run it as a script (not inside an interactive cell), or use `--jobs 1`.
