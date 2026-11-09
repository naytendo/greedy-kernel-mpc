"""Reproduce the scalar kernel-MPC study from local source and reference data."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from time import perf_counter

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "data" / "reference"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "simulate", "smoke"), default="reference")
    parser.add_argument("--profile", choices=("conference", "primary", "extended"), default="conference")
    parser.add_argument("--output", type=Path, help="Output folder; defaults to outputs/<mode-or-profile>")
    parser.add_argument("--random-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)
    if args.random_repetitions < 1:
        parser.error("--random-repetitions must be positive")
    name = args.profile if args.mode == "simulate" else args.mode
    output = (args.output or ROOT / "outputs" / name).resolve()
    if output == ROOT or output == REFERENCE or output in REFERENCE.parents or REFERENCE in output.parents:
        parser.error("Choose an output folder separate from the package source and reference data")
    protected = [ROOT / name for name in ("engine", "analysis", "tests", "data")]
    if any(output == path or path in output.parents for path in protected):
        parser.error("Choose an output folder outside engine, analysis, tests and data")
    output.mkdir(parents=True, exist_ok=True)
    from analysis import run_center_comparison, run_derivative_audit
    from plot_results import render_results
    from verify_results import verify_reference_files, verify_results

    start = perf_counter()
    reference_file_count = verify_reference_files(REFERENCE)
    print(f"Verified {reference_file_count} reference-file checksums.", flush=True)
    with threadpool_limits(limits=1):
        if args.mode == "reference":
            print("Using included archived runs. No MPC simulation is performed in reference mode.", flush=True)
            for path in [REFERENCE / "summary.json", *REFERENCE.glob("*.csv")]:
                shutil.copy2(path, output / path.name)
        else:
            from engine.study import run_study, ALL_KEYS, CONFERENCE_KEYS, PRIMARY_KEYS
            keys = {"conference": CONFERENCE_KEYS, "primary": PRIMARY_KEYS, "extended": ALL_KEYS}[args.profile]
            kwargs = {"keys": keys, "progress": lambda message: print(message, flush=True)}
            if args.mode == "smoke":
                kwargs.update(keys=CONFERENCE_KEYS, weights=(16.,), starts=(.75,), steps=8)
                print("SMOKE TEST: three 8-step runs, not the paper's aggregate experiment.", flush=True)
            else:
                print(f"Recomputing the {args.profile} experiment from the simulated plant.", flush=True)
            run_study(output, **kwargs)
        print("Recomputing the center-placement comparison...", flush=True)
        run_center_comparison(output, output, repetitions=args.random_repetitions, seed=args.seed)
        print("Recomputing derivative certificates and high-precision audits...", flush=True)
        run_derivative_audit(output, output)
        print("Checking dynamics, constraints and actual-input cost accounting...", flush=True)
        verification = verify_results(output, REFERENCE, compare_reference=args.mode != "smoke")
        print("Rendering the five-panel figure and tables...", flush=True)
        figure_info = render_results(output, output)
    seconds = perf_counter() - start
    manifest = {
        "mode": args.mode, "profile": args.profile if args.mode == "simulate" else args.mode,
        "is_paper_experiment": args.mode != "smoke",
        "elapsed_seconds": seconds, "random_repetitions": args.random_repetitions,
        "random_seed": args.seed, "reference_file_count": reference_file_count,
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {p: importlib.metadata.version(p)
                         for p in ("numpy", "scipy", "matplotlib", "threadpoolctl")},
        "figure": figure_info,
        "scope": "A fixed-gain nonlinear MPC study with verified feasible candidates. No global NLP optimality or formal interval certificate is claimed.",
    }
    for filename, content in (("verification.json", verification), ("run-manifest.json", manifest)):
        (output / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(f"Completed in {seconds:.1f} seconds. Outputs: {output}", flush=True)
    print("Main figure: study-summary.pdf (and .png). Metrics: metrics.csv. Checks: verification.json.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
