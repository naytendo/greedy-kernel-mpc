"""Check saved plant trajectories, actual-input accounting and reference metrics."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def verify_reference_files(reference_dir):
    root = Path(reference_dir)
    hashes = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        path = root / name
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference checksum failed: {name}")
    return len(hashes)


def verify_results(data_dir, reference_dir=None, *, compare_reference=True):
    root = Path(data_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    reference = None
    if reference_dir is not None and compare_reference:
        reference = json.loads((Path(reference_dir) / "summary.json").read_text(encoding="utf-8"))
    by_reference = ({(r["input_weight"], r["controller"], r["initial_state"]): r
                     for r in reference["runs"]} if reference else {})
    models = {}
    for d in summary["selected_designs"]:
        centers = np.asarray(d["center_locations"])
        labels = np.asarray(d["training_labels"])
        def kernel(x, centers=centers):
            t = np.sqrt(5) / .5 * np.abs(np.asarray(x).reshape(-1, 1) - centers)
            return (1 + t + t * t / 3) * np.exp(-t)
        alpha = cho_solve(cho_factor(kernel(centers), lower=True), labels)
        offset = float((kernel([0]) @ alpha)[0])
        models[d["key"]] = (kernel, alpha, offset)
    maximum_constraint_excess = 0.
    maximum_reference_cost_difference = 0.
    maximum_reference_cost_relative_difference = 0.
    maximum_cost_accounting_error = 0.
    reference_matches = 0
    for run in summary["runs"]:
        rows = np.loadtxt(root / (run["run_id"] + ".csv"), delimiter=",", skiprows=1, ndmin=2)
        if len(rows) != run["steps"]:
            raise AssertionError("The CSV and summary disagree on the step count")
        x, u, z, v = (rows[:, i] for i in (1, 2, 3, 4))
        successor = np.r_[x[1:], run["final_state"]]
        np.testing.assert_allclose(successor, 1.1 * x + u - .2 * np.sin(x), atol=2e-10, rtol=0)
        feedback = -.6 * (x - z)
        np.testing.assert_allclose(u, v + feedback, atol=2e-10, rtol=0)
        np.testing.assert_allclose(u * u, v * v + feedback * feedback + 2 * v * feedback,
                                   atol=2e-10, rtol=0)
        np.testing.assert_allclose(rows[:, 16], feedback, atol=2e-10, rtol=0)
        cost = np.cumsum(x * x + run["input_weight"] * u * u)
        np.testing.assert_allclose(rows[:, 12], cost, atol=2e-9, rtol=0)
        maximum_cost_accounting_error = max(maximum_cost_accounting_error,
                                             abs(float(cost[-1]) - run["actual_cost"]))
        np.testing.assert_allclose(run["actual_cost"], cost[-1], atol=2e-9, rtol=0,
                                   err_msg=f"Summary/CSV cost mismatch: {run['run_id']}")
        prediction = 1.1 * z + v
        if run["controller"] in models:
            kernel, alpha, offset = models[run["controller"]]
            prediction += kernel(z) @ alpha - offset
        np.testing.assert_allclose(rows[:, 6], np.abs(successor - prediction), atol=2e-8, rtol=0)
        excess = max(float(np.max(np.abs(x)) - 1), abs(run["final_state"]) - 1,
                     float(np.max(np.abs(u)) - .3),
                     float(np.max(np.abs(x - z) - rows[:, 14])),
                     float(np.max(rows[:, 6] - rows[:, 15])), 0.)
        maximum_constraint_excess = max(maximum_constraint_excess, excess)
        if excess > 5e-7:
            raise AssertionError(f"Physical/tube constraint check failed: {run['run_id']}")
        np.testing.assert_allclose(run["input_energy"], u @ u, atol=2e-9, rtol=0)
        if reference:
            old = by_reference[run["input_weight"], run["controller"], run["initial_state"]]
            if run["steps"] != old["steps"]:
                raise ValueError("Reference cost comparison requires the original 30-step runs")
            difference = abs(run["actual_cost"] - old["actual_cost"])
            relative = difference / max(abs(old["actual_cost"]), 1e-15)
            maximum_reference_cost_difference = max(maximum_reference_cost_difference, difference)
            maximum_reference_cost_relative_difference = max(maximum_reference_cost_relative_difference, relative)
            np.testing.assert_allclose(run["actual_cost"], old["actual_cost"], rtol=1e-4, atol=5e-7,
                                       err_msg=f"Reference actual-cost mismatch: {run['run_id']}")
            reference_matches += 1
    return {
        "passed": True, "run_count": len(summary["runs"]),
        "recorded_steps": sum(r["steps"] for r in summary["runs"]),
        "maximum_constraint_excess": maximum_constraint_excess,
        "maximum_cost_accounting_error": maximum_cost_accounting_error,
        "reference_cost_comparison": bool(reference), "reference_runs_matched": reference_matches,
        "reference_rtol": 1e-4, "reference_atol": 5e-7,
        "maximum_reference_cost_difference": maximum_reference_cost_difference,
        "maximum_reference_cost_relative_difference": maximum_reference_cost_relative_difference,
        "scope": "Numerical trajectory and accounting checks, not a proof of global optimality or formal interval validation.",
    }
