# FD-ISAC: Joint Transmit/Receive Beamforming in Full-Duplex ISAC

Python implementation of

> Z. Liu, S. Aditya, H. Li, B. Clerckx, "Joint Transmit and Receive Beamforming Design in Full-Duplex
> Integrated Sensing and Communications," *IEEE JSAC*, vol. 41, no. 9, pp. 2907-2919, Sep. 2023.

It implements the penalty-based block-coordinate-descent (BCD) algorithm (Algorithm 1), the baselines
(NSP, radar-only, communication-only, half-duplex) and the experiments that correspond to Figs. 2, 3, 4, 5, 7, 8, 9.

---

## 0. Versions
v1 -> 

## 1. Setup

Requires **Python 3.10 or newer**.

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt  # numpy, scipy, matplotlib, tqdm, pytest
```

## 2. Quick start (what to run)

Run everything from the project root (the folder that contains `i_config.py`).

| Goal | Command | Time |
|---|---|---|
| 1. Check the maths is right | `python -m pytest -v` | seconds |
| 2. Smoke-test all experiments | `python run_all.py --mc 5 --parallel` | < 1 min |
| 3. Reasonable figures | `python run_all.py --mc 30 --parallel` | a few minutes |
| 4. Paper-level statistics | `python run_all.py --mc 1000 --parallel` | hours |

Figures (PNG) and data (`.npz`) are written to `results/`; each script's printed output is in `results/logs/`.

To run one experiment only:

```bash
python experiments/exp_convergence.py --mc 50      # Fig. 2
python experiments/exp_si_cancellation.py --mc 30  # Figs. 3, 4
python experiments/exp_beampattern.py              # Fig. 5   (single realisation, no --mc)
python experiments/exp_sumrate.py --mc 30          # Fig. 7
python experiments/exp_sensing.py --mc 20          # Figs. 8, 9 (+ detection rates)
```

`--mc N` = number of **Monte-Carlo trials** (random channel draws that are averaged). The paper uses 1000.
Add `--show` to a single script to open the plot window. `--seed S` changes the random seed.

## 3. Project layout

The Roman-numeral prefixes give the reading/build order.

```
i_config.py        simulation constants (Sec. V) + documented assumptions     [PAPER]/[ASSUMED] tags
ii_channels.py     steering vectors, Rician Hu/Hd/Hsi, radar channel Hr       Eqs (5)-(10)
iii_metrics.py     rates, MSEs, beampattern gains, residual SI, objectives     Eqs (11)-(15), (18)-(19), (23)
iv_updates.py      closed-form BCD block updates (rho, wu, w, ud, p)           Eqs (26),(27),(32),(34),(37),(43)
v_algorithm.py     Algorithm 1 loop, convergence test, final normalisation     Eqs (44),(51)
vi_sensing.py      range-Doppler map and angle spectrum                        Eqs (53),(54)
vii_baselines.py   NSP, radar-only, comm-only, half-duplex                     Sec. V
run_all.py         runs every experiment
conftest.py        lets pytest find the modules
tests/
    test_updates.py       verification of every block update and of the whole algorithm
experiments/
    common.py             shared helpers (not an experiment)
    exp_convergence.py    Fig. 2
    exp_si_cancellation.py Figs. 3 and 4
    exp_beampattern.py    Fig. 5
    exp_sumrate.py        Fig. 7
    exp_sensing.py        Figs. 8 and 9
results/           generated figures, data, logs
```

Every module can also be run on its own (`python iii_metrics.py`, `python iv_updates.py`, ...); each has a
short self-check under `if __name__ == "__main__":`.

## 4. Minimal usage example

```python
import numpy as np
from i_config import Cfg
from ii_channels import draw_all
from v_algorithm import solve_fd_isac, alphas_from_varrho

cfg = Cfg()                                         # Nt = Nr = 16, fc = 2.4 GHz, ...
ch = draw_all(cfg, si_dB=60, rng=np.random.default_rng(0))     # channels with 60 dB SI
out = solve_fd_isac(ch, cfg, a=alphas_from_varrho(1.0))        # varrho = alpha_com / alpha_radar
print(out["metrics"])        # Ru, Rd, Gt, Gr, res_SI_dB, soi_si_dB
```

`varrho` is the communication-vs-radar priority: small values favour sensing, large values favour communications,
`np.inf` gives the communications-only upper bound.

## 5. What the tests check (`tests/test_updates.py`)

- Rate-MSE equivalence: `-ln E <= ln(1+SINR)`, with equality at the MMSE receiver.
- Each block (ud, wu, w, p) is optimal: zero gradient, KKT conditions, no feasible perturbation improves it.
- The stable (Sherman-Morrison / projected) implementations equal a literal transcription of the paper's
  Eqs. (34) and (43).
- Objective (23) is non-decreasing for varrho = 1, 100, inf; `||p||^2 = Pd`, `||w|| = 1`, `||wu||^2 <= Pu`.
- Residual SI decreases as the penalty parameter beta decreases.

## 6. Assumptions not given in the paper (see `i_config.py`)

Rician K = 1 (user links) and 10 (SI link); radar cross-section 1 m^2; downlink-user noise floor = -94 dBm;
all powers in mW; "SI power = X dB" means X dB above the noise floor per RX antenna for an unstructured p
with `||p||^2 = Pd`. Both user AoA/AoD angles are taken equal. NSP uses a 50/50 radar/communication power split.
State these in any report.

## 7. Deviations from / corrections to the paper

- Eq. (30)/(32): the factor `rho_u` is missing in the `wu` update. Derived and used here.
- Eq. (33): `alpha_4` appears twice. Eq. (42): `Hd ud ud^H Hd^H` must be `Hd^H ud ud^H Hd`, and `-alpha_3 Zt I` is `-alpha_3 Zt`.
- `||p||^2 = Pd` must be an **equality** constraint (Eq. 40); the bound (21) is tight only then.
- `rho* = 1/E` is exact for the natural log; the code uses `ln` in the objective (rates in nats).
- Doppler phase in Eq. (53) advances per block (`fd * N * Ts * m`), not per sample.
- With `beta = 1e-25` the penalty weight is ~5e24. Plain `solve`/`eigh` fail, so the SI term is handled exactly
  (Sherman-Morrison, or a projection onto the orthogonal complement of `Hsi^H w`).

## 8. Known limitations (honest status)

- **Fig. 2, 8, 9:** reproduce well (monotone convergence; the target is detected at 7.5 m; radar-only and
  comm-only are drowned by SI).
- **Figs. 3, 4:** SI is suppressed far below the noise floor for every varrho (qualitatively right), but the
  varrho-dependence shown in the paper is not reproduced; absolute dB values are numerical-precision-limited.
- **Figs. 5, 7:** the varrho sweep does not yet show the paper's smooth trade-off. For large varrho (>= 100)
  the BCD converges slowly and can stop in poor local optima that depend on the random start.
  Suggested remedies: several random starts and keep the best, or start from the varrho = inf / radar-only solution.
- **Rates:** the half-duplex rate (about 16 bps/Hz) is higher than the paper's (about 10). SNR-related
  assumptions (Rician K, path loss, power units) are not fully specified in the paper.

## 9. Troubleshooting

- `ModuleNotFoundError: i_config` -> run from the project root, not from inside `experiments/` or `tests/`.
- A script prints `[FAILED]` in `run_all.py` -> open `results/logs/<script>.log`.
- Plots do not open -> by design they are only saved; add `--show` when running a single script.
- Delete any stray folder literally named `{experiments,tests}` (left over from an old zip).
