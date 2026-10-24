"""
Runs the pipeline in order:

    python run_all.py          # runs every step that isn't already up to date
    python run_all.py 04       # reruns step 04 onwards, whether up to date or not

Double-clicking this file (or VS Code's Run button) also works. A step counts
as up to date when its outputs are newer than its inputs and its own script,
so rerunning after a failure picks up where it stopped. A quick package
check (00_check_env.py) always runs first. If the project has a .venv, every
step runs with that environment's Python, and everything printed is also
saved to outputs/run_log.txt. A full run takes roughly an hour.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "run_log.txt"
CHECK = "00_check_env.py"
# (script, inputs, outputs); a step with no outputs always runs.
STEPS = [
    ("01_merge_data.py",
     ["data/yelp_academic_dataset_business.csv", "data/yelp_academic_dataset_review.csv"],
     ["data/az_reviews.csv"]),
    ("02_preprocess.py", ["data/az_reviews.csv"],
     ["data/reviews_preprocessed.csv", "outputs/metrics/02_preprocess.json"]),
    ("03_sentiment_eda.py", ["data/reviews_preprocessed.csv"],
     ["outputs/metrics/03_sentiment_eda.json"]),
    ("04_topics.py", ["data/reviews_preprocessed.csv"], ["outputs/metrics/04_topics.json"]),
    ("05_star_predictor.py", ["data/reviews_preprocessed.csv"],
     ["outputs/metrics/05_star_predictor.json"]),
    ("06_closure_risk.py", ["data/reviews_preprocessed.csv"], ["outputs/metrics/06_closure_risk.json"]),
    ("07_summary.py", [], []),
]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def project_python():
    """Use the project's virtual environment if there is one."""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def up_to_date(script, inputs, outputs):
    if not outputs:
        return False
    ins = [ROOT / "src" / script] + [ROOT / p for p in inputs]
    outs = [ROOT / p for p in outputs]
    if not all(p.exists() for p in ins + outs):
        return False
    return min(p.stat().st_mtime for p in outs) > max(p.stat().st_mtime for p in ins)


def main():
    os.chdir(ROOT)  # every script uses paths relative to the repo root
    LOG.parent.mkdir(parents=True, exist_ok=True)
    forced_from = sys.argv[1].zfill(2) if len(sys.argv) > 1 else None
    python = project_python()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

    with open(LOG, "a", encoding="utf-8") as log:
        def emit(text):
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        def run(script):
            emit(f"\n{'=' * 70}\n{script}\n{'=' * 70}")
            t0 = time.time()
            proc = subprocess.Popen([python, f"src/{script}"], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=env, text=True,
                                    encoding="utf-8", errors="replace")
            for line in proc.stdout:
                emit(line.rstrip("\n"))
            code = proc.wait()
            if code != 0:
                emit(f"\nFAILED: {script} (exit code {code}). Fix the error, then run this "
                     f"file again; finished steps are skipped.")
                sys.exit(code)
            emit(f"[{script} finished in {time.time() - t0:,.0f}s]")

        total = time.time()
        emit(f"\n#### Run started {time.strftime('%Y-%m-%d %H:%M:%S')} using {python}")
        run(CHECK)
        for script, inputs, outputs in STEPS:
            if forced_from is not None:
                if script[:2] < forced_from:
                    continue
            elif up_to_date(script, inputs, outputs):
                emit(f"\n[{script} is up to date, skipping]")
                continue
            run(script)
        emit(f"\nPIPELINE COMPLETE in {(time.time() - total) / 60:,.1f} minutes. "
             f"Summary: outputs/RESULTS.md")


if __name__ == "__main__":
    main()
