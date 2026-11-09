"""Compare kernel-geometry certificates without new MPC simulations.

Every method uses an origin seed, the same candidate pool and evaluation mesh,
and the same fixed Matérn kernel and norm budget. Uniform designs are rebuilt
at each odd budget; random designs use nested prefixes of random permutations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from threadpoolctl import threadpool_limits


def kernel(x, y, length_scale):
    t = (math.sqrt(5) / length_scale
         * np.abs(np.asarray(x).reshape(-1, 1) - np.asarray(y).reshape(1, -1)))
    return (1 + t + t*t/3) * np.exp(-t)


def prefix_certificates(centers, mesh, length_scale, norm_budget, correction):
    """One Cholesky factorization gives every nested-prefix power function."""
    gram = kernel(centers, centers, length_scale)
    factor = cholesky(gram, lower=True, check_finite=True)
    features = solve_triangular(factor, kernel(centers, mesh, length_scale),
                                lower=True, check_finite=True)
    variance = 1 - np.cumsum(features*features, axis=0)
    minimum = float(variance.min())
    if minimum < -5e-9:
        raise ArithmeticError("Materially negative computed power variance")
    power = np.sqrt(np.maximum(0, variance)).max(axis=1)
    certificates = norm_budget*power + correction
    if np.any(np.diff(certificates) > 1e-10):
        raise ArithmeticError("Unexpected increase along a nested sequence")
    diagnostics = {
        "gram_condition": float(np.linalg.cond(gram)),
        "minimum_cholesky_diagonal": float(np.diag(factor).min()),
        "minimum_computed_power_variance": minimum,
    }
    return certificates, diagnostics


def first_tested_budget(curve, budgets, target):
    indices = np.flatnonzero(np.asarray(curve) <= target)
    return int(budgets[indices[0]]) if len(indices) else None


def censored_quantiles(values):
    """Include unsuccessful designs as censored beyond the tested budget."""
    ordered = sorted(float("inf") if v is None else float(v) for v in values)
    result = {}
    for name, q in (("q10", .1), ("median", .5), ("q90", .9)):
        value = ordered[max(0, math.ceil(q*len(ordered))-1)]
        result[name] = None if math.isinf(value) else int(value)
    return result


def run_center_comparison(
    data_dir, output_dir, repetitions=200, seed=20261107, max_centers=65
) -> dict:
    """Write and return the common-pool greedy/uniform/random comparison.

    Parameters are directories. The input directory must contain summary.json
    with selected_designs and the full origin-seeded center prefix through at
    least max_centers. No labels or saved closed-loop CSV files are consumed.
    """
    if max_centers < 3 or max_centers % 2 != 1:
        raise ValueError("Use an odd maximum budget of at least three")
    if max_centers > 2049:
        raise ValueError("The candidate pool has only 2049 distinct points")
    if repetitions < 1:
        raise ValueError("At least one random repetition is required")
    source = Path(data_dir).resolve() / "summary.json"
    source_bytes = source.read_bytes()
    study = json.loads(source_bytes)
    pool = np.linspace(-1, 1, 2049)
    mesh = np.linspace(-1, 1, 4097)
    origin = len(pool)//2
    budgets = np.arange(1, max_centers+1, 2)
    length_scale, norm_budget = .5, .5
    feature_lipschitz = math.sqrt(5/3)/length_scale
    delta = float(np.diff(mesh).max()/2)
    correction = norm_budget*feature_lipschitz*delta
    design = max(study["selected_designs"], key=lambda d: d["centers"])
    greedy = np.asarray(design["center_locations"][:max_centers])
    if len(greedy) != max_centers or greedy[0] != 0:
        raise ValueError("Saved origin-seeded greedy sequence is too short")
    if not np.allclose(greedy, pool[np.rint((greedy+1)*1024).astype(int)],
                       atol=1e-14, rtol=0):
        raise ValueError("Greedy centers do not belong to the common pool")

    rng = np.random.default_rng(seed)
    other_indices = np.delete(np.arange(len(pool)), origin)
    diagnostics = []
    uniform_indices, uniform = [], []
    random_indices, random_curves = [], []
    with threadpool_limits(limits=1):
        greedy_full, audit = prefix_certificates(
            greedy, mesh, length_scale, norm_budget, correction)
        diagnostics.append(audit)
        greedy_curve = greedy_full[budgets-1]
        for n in budgets:
            indices = (np.array([origin]) if n == 1 else
                       np.rint(np.linspace(0, len(pool)-1, int(n))).astype(int))
            if len(np.unique(indices)) != n or origin not in indices:
                raise ValueError("Uniform design needs distinct centers and origin")
            uniform_indices.append(indices.tolist())
            curve, audit = prefix_certificates(
                pool[indices], mesh, length_scale, norm_budget, correction)
            uniform.append(float(curve[-1]))
            diagnostics.append(audit)
        for _ in range(repetitions):
            indices = np.r_[origin, rng.permutation(other_indices)[:max_centers-1]]
            curve, audit = prefix_certificates(
                pool[indices], mesh, length_scale, norm_budget, correction)
            random_indices.append(indices.tolist())
            random_curves.append(curve[budgets-1].tolist())
            diagnostics.append(audit)
    random_curves = np.asarray(random_curves)

    # Saved control designs also include their tiny origin-offset allowance.
    design_agreement = {}
    for item in study["selected_designs"]:
        if item["certificate"] == "power" and item["centers"] <= max_centers:
            difference = float(abs(greedy_full[item["centers"]-1]
                                   - item["certified_error"]))
            if difference > 1e-9:
                raise AssertionError("Certificate disagrees with saved power design")
            design_agreement[item["key"]] = difference

    thresholds = []
    for target in (.045, .0225):
        stops = [first_tested_budget(row, budgets, target) for row in random_curves]
        thresholds.append({
            "target_error": target,
            "greedy_first_tested_budget":
                first_tested_budget(greedy_curve, budgets, target),
            "uniform_first_tested_budget":
                first_tested_budget(uniform, budgets, target),
            "random_first_tested_budgets": stops,
            "random_budget_quantiles_including_censoring": censored_quantiles(stops),
            "random_uncertified_count": sum(value is None for value in stops),
        })
    result = {
        "source_summary": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scope": "New center-geometry comparison only. Existing MPC runs are unchanged.",
        "protocol": {
            "domain": [-1, 1], "candidate_count": len(pool),
            "evaluation_mesh_count": len(mesh), "covering_radius": delta,
            "kernel": "unit-variance Matern 5/2", "length_scale": length_scale,
            "rkhs_norm_budget": norm_budget,
            "kernel_section_lipschitz": feature_lipschitz,
            "covering_allowance": correction,
            "origin_seed_for_all_methods": True,
            "random_repetitions": int(repetitions), "rng_seed": int(seed),
            "random_sampling": "without replacement, nested prefixes",
            "uniform_sampling": "symmetric equal spacing snapped to pool, nonnested sets",
            "comparison_budgets": budgets.tolist(),
            "quantity": "R * (mesh maximum power function + C_k * delta)",
            "residual_labels_used": False,
            "interval_arithmetic": False, "regularization_or_jitter_added": False,
        },
        "greedy_centers": greedy.tolist(),
        "uniform_candidate_indices": uniform_indices,
        "random_candidate_indices": random_indices,
        "greedy_certificates": greedy_curve.tolist(),
        "uniform_certificates": uniform,
        "random_certificates": random_curves.tolist(),
        "random_median": np.median(random_curves, axis=0).tolist(),
        "random_q10": np.quantile(random_curves, .1, axis=0).tolist(),
        "random_q90": np.quantile(random_curves, .9, axis=0).tolist(),
        "target_comparisons": thresholds,
        "numerical_diagnostics": {
            "maximum_gram_condition": max(a["gram_condition"] for a in diagnostics),
            "minimum_cholesky_diagonal":
                min(a["minimum_cholesky_diagonal"] for a in diagnostics),
            "minimum_computed_power_variance":
                min(a["minimum_computed_power_variance"] for a in diagnostics),
            "saved_design_absolute_differences": design_agreement,
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"center-selection-comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--max-centers", type=int, default=65)
    args = parser.parse_args()
    result = run_center_comparison(
        args.data, args.output, args.repetitions, args.seed, args.max_centers)
    for row in result["target_comparisons"]:
        print({key: value for key, value in row.items()
               if key != "random_first_tested_budgets"})


if __name__ == "__main__":
    main()
