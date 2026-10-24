"""Run every experiment in experiments/ one after another (or in parallel).

    python run_all.py                 # default Monte-Carlo counts, sequential
    python run_all.py --mc 5          # quick smoke test (tiny MC)
    python run_all.py --mc 100 --parallel    # all five at the same time (uses several CPU cores)

Figures and .npz data land in results/. Each script's log goes to results/logs/<name>.log.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp_convergence.py", "exp_si_cancellation.py", "exp_beampattern.py", "exp_sumrate.py", "exp_sensing.py"]
NO_MC = {"exp_beampattern.py"}                  # single-realisation figure, has no meaningful --mc


def cmd_for(script, mc):
    c = [sys.executable, os.path.join(ROOT, "experiments", script)]
    if mc is not None and script not in NO_MC:
        c += ["--mc", str(mc)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=None, help="override the Monte-Carlo count of every script")
    ap.add_argument("--parallel", action="store_true", help="launch all scripts at once")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    t0 = time.time()
    procs = []
    for s in SCRIPTS:
        log = open(os.path.join(ROOT, "results", "logs", s.replace(".py", ".log")), "w")
        print(f"[start] {s}")
        p = subprocess.Popen(cmd_for(s, args.mc), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if args.parallel:
            procs.append((s, p, log))
        else:
            rc = p.wait(); log.close()
            print(f"[{'ok' if rc == 0 else 'FAILED'}] {s}  ({time.time() - t0:.0f}s elapsed)")
    for s, p, log in procs:
        rc = p.wait(); log.close()
        print(f"[{'ok' if rc == 0 else 'FAILED'}] {s}")
    print(f"done in {time.time() - t0:.0f}s -> figures in results/, logs in results/logs/")


if __name__ == "__main__":
    main()