"""Audit derivative certificates for saved scalar-study interpolants.

No trajectories are regenerated. The only model information used is the saved
center/label pairs, fixed Matérn 5/2 kernel, and supplied RKHS norm budget.
The plant formula and its Lipschitz constant are not used in these bounds.

The formulas are deterministic continuous-domain bounds in exact arithmetic.
This implementation audits floating-point evaluation with a finer mesh and
60-digit Decimal calculations at the reported maximizers; it is not a formal
interval-arithmetic certificate. Run with the reproduction environment's Python.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from threadpoolctl import threadpool_limits


ELL = 0.5
RKHS_BUDGET = 0.5
NOMINAL_CLOSED_LOOP = 0.5  # A + K_fb = 1.1 - 0.6.
EXISTING_Q = 0.7


def kernel(x, centers, a):
    d = np.abs(np.asarray(x).reshape(-1, 1) - centers.reshape(1, -1))
    t = a * d
    return (1.0 + t + t*t/3.0) * np.exp(-t)


def kernel_dx(x, centers, a):
    d = np.asarray(x).reshape(-1, 1) - centers.reshape(1, -1)
    t = a * np.abs(d)
    return -(a*a/3.0) * d * (1.0+t) * np.exp(-t)


def verify_kernel_constants():
    """Audit the two diagonal mixed derivative identities independently."""
    with localcontext() as ctx:
        ctx.prec = 60
        a = Decimal(5).sqrt()/Decimal("0.5")
        h = Decimal("1e-8")

        def k(distance):
            t = a*abs(distance)
            return (1+t+t*t/3)*(-t).exp()

        first_norm_squared_fd = -(k(h)-2*k(0)+k(-h))/(h*h)
        second_norm_squared_fd = (k(2*h)-4*k(h)+6*k(0)-4*k(-h)+k(-2*h))/(h**4)
        first_expected = a*a/3
        second_expected = a**4
        relative_first = float(abs(first_norm_squared_fd/first_expected-1))
        relative_second = float(abs(second_norm_squared_fd/second_expected-1))
        if relative_first > 1e-10 or relative_second > 1e-6:
            raise AssertionError("Diagonal kernel derivative identity check failed")
        return {
            "method": "60-digit centered second/fourth differences of the kernel formula",
            "step": str(h),
            "first_derivative_representer_squared_norm_expected": str(first_expected),
            "second_derivative_representer_squared_norm_expected": str(second_expected),
            "relative_difference_first": relative_first,
            "relative_difference_second": relative_second,
            "analytic_reason": (
                "For k(d)=(1+a|d|+a^2 d^2/3)exp(-a|d|), "
                "-k''(0)=a^2/3 and k''''(0)=a^4. Integration of the "
                "second-derivative representer bounds derivative-feature variation "
                "by a^2 times distance."),
        }


def decimal_audit(centers, labels, locations, float_values):
    """Independent high-precision kernel, factorization and selected queries."""
    with localcontext() as ctx:
        ctx.prec = 60
        dc = [Decimal(str(float(x))) for x in centers]
        dy = [Decimal(str(float(x))) for x in labels]
        da = Decimal(5).sqrt()/Decimal("0.5")
        n = len(dc)

        def entry(x, t):
            q = da*abs(x-t)
            return (1+q+q*q/3)*(-q).exp()

        gram = [[entry(x, t) for t in dc] for x in dc]
        lower = [[Decimal(0) for _ in dc] for _ in dc]
        for i in range(n):
            for j in range(i+1):
                value = gram[i][j]-sum(lower[i][k]*lower[j][k] for k in range(j))
                lower[i][j] = value.sqrt() if i == j else value/lower[j][j]

        def forward(rhs):
            result = []
            for i in range(n):
                result.append((rhs[i]-sum(lower[i][j]*result[j] for j in range(i)))
                              /lower[i][i])
            return result

        theta = forward(dy)
        norm_squared = sum(value*value for value in theta)
        beta = (Decimal("0.25")-norm_squared).sqrt()
        queries = []
        for x_float in sorted(set(locations)):
            x = Decimal(str(float(x_float)))
            derivatives = []
            for center in dc:
                d = x-center
                q = da*abs(d)
                derivatives.append(-(da*da/3)*d*(1+q)*(-q).exp())
            features = forward(derivatives)
            mean_derivative = sum(v*t for v, t in zip(features, theta))
            power_squared = da*da/3-sum(v*v for v in features)
            if power_squared < 0:
                raise ArithmeticError("Negative derivative power at 60-digit precision")
            p1 = power_squared.sqrt()
            reference = float_values[float(x_float)]
            queries.append({
                "x": float(x_float),
                "mean_derivative_60digits": str(mean_derivative),
                "derivative_power_60digits": str(p1),
                "mean_derivative_float64_abs_difference":
                    abs(float(mean_derivative)-reference[0]),
                "derivative_power_float64_abs_difference": abs(float(p1)-reference[1]),
            })
        return {
            "decimal_precision": 60,
            "stored_decimal_float_inputs_treated_as_exact": True,
            "norm_60digits": str(norm_squared.sqrt()),
            "remaining_budget_60digits": str(beta),
            "queries": queries,
            "max_mean_derivative_float64_abs_difference":
                max(q["mean_derivative_float64_abs_difference"] for q in queries),
            "max_derivative_power_float64_abs_difference":
                max(q["derivative_power_float64_abs_difference"] for q in queries),
        }


def certify_one(design, mesh, finer_mesh):
    centers = np.asarray(design["center_locations"], dtype=float)
    labels = np.asarray(design["training_labels"], dtype=float)
    if centers.size != int(design["centers"]) or len(set(centers)) != centers.size:
        raise ValueError("Saved centers are inconsistent or repeated")
    a = math.sqrt(5.0)/ELL
    first_derivative_diagonal = a*a/3.0
    # k''''(0) = a**4, so ||d/dx k(.,x)-d/dx k(.,t)|| <= a**2 |x-t|.
    derivative_feature_lipschitz = a*a
    delta = float(np.max(np.diff(mesh))/2.0)
    gram = kernel(centers, centers, a)
    lower = np.linalg.cholesky(gram)
    theta = solve_triangular(lower, labels, lower=True)
    coefficients = cho_solve((lower, True), labels)
    norm = float(np.linalg.norm(theta))
    if norm > RKHS_BUDGET:
        raise ValueError("The supplied RKHS budget is smaller than the fitted norm")
    beta = math.sqrt(RKHS_BUDGET**2-norm**2)

    def evaluate(points):
        derivatives = kernel_dx(points, centers, a)
        features = solve_triangular(lower, derivatives.T, lower=True)
        power_squared = first_derivative_diagonal-np.einsum("ij,ij->j", features, features)
        # An appreciable negative value indicates failed numerical evaluation.
        if np.min(power_squared) < -1e-8:
            raise ArithmeticError("Numerically negative derivative power")
        derivative_power = np.sqrt(np.maximum(0.0, power_squared))
        model_derivative = derivatives@coefficients
        return model_derivative, derivative_power, power_squared

    fitted, p1, p1_squared = evaluate(mesh)
    fitted_fine, p1_fine, _ = evaluate(finer_mesh)
    correction_per_unit_norm = derivative_feature_lipschitz*delta
    model_correction = norm*correction_per_unit_norm
    uncertainty_correction = beta*correction_per_unit_norm
    joint_correction = (norm+beta)*correction_per_unit_norm
    quantities = {
        "model_lipschitz_upper": float(np.max(np.abs(fitted))+model_correction),
        "derivative_error_upper": float(beta*(np.max(p1)+correction_per_unit_norm)),
        "true_residual_lipschitz_upper":
            float(np.max(np.abs(fitted)+beta*p1)+joint_correction),
        "learned_model_contraction_upper":
            float(np.max(np.abs(NOMINAL_CLOSED_LOOP+fitted))+model_correction),
        "true_plant_contraction_upper":
            float(np.max(np.abs(NOMINAL_CLOSED_LOOP+fitted)+beta*p1)+joint_correction),
    }
    fine_values = {
        "model_lipschitz_upper": float(np.max(np.abs(fitted_fine))),
        "derivative_error_upper": float(beta*np.max(p1_fine)),
        "true_residual_lipschitz_upper": float(np.max(np.abs(fitted_fine)+beta*p1_fine)),
        "learned_model_contraction_upper":
            float(np.max(np.abs(NOMINAL_CLOSED_LOOP+fitted_fine))),
        "true_plant_contraction_upper":
            float(np.max(np.abs(NOMINAL_CLOSED_LOOP+fitted_fine)+beta*p1_fine)),
    }
    if any(fine_values[key] > bound+1e-9 for key, bound in quantities.items()):
        raise AssertionError("Fine mesh violates a coarse-mesh corrected allowance")
    idx = {
        "derivative_power": int(np.argmax(p1)),
        "model_lipschitz": int(np.argmax(np.abs(fitted))),
        "true_residual_lipschitz": int(np.argmax(np.abs(fitted)+beta*p1)),
        "learned_contraction": int(np.argmax(np.abs(NOMINAL_CLOSED_LOOP+fitted))),
        "true_contraction": int(np.argmax(np.abs(NOMINAL_CLOSED_LOOP+fitted)+beta*p1)),
    }
    locations = [float(mesh[i]) for i in idx.values()]
    float_values = {float(mesh[i]): (float(fitted[i]), float(p1[i])) for i in idx.values()}
    high_precision = decimal_audit(centers, labels, locations, float_values)
    if high_precision["max_mean_derivative_float64_abs_difference"] > 1e-7:
        raise ArithmeticError("High-precision derivative audit failed")
    if high_precision["max_derivative_power_float64_abs_difference"] > 1e-7:
        raise ArithmeticError("High-precision power audit failed")
    interpolation_residual = gram@coefficients-labels
    origin_offset = float((kernel([0.0], centers, a)@coefficients).item())
    rho = float(design["requested_radius"])
    value_error = float(design["certified_error"])
    result = {
        "key": design["key"], "centers": int(centers.size),
        "original_value_certificate_kind": design["certificate"],
        "original_requested_radius": rho,
        "original_value_error_certificate": value_error,
        "rkhs_norm": norm, "remaining_norm_budget": beta,
        "saved_norm_abs_difference": abs(norm-float(design["model_norm"])),
        "gram_condition_2": float(np.linalg.cond(gram)),
        "interpolation_residual_inf": float(np.max(np.abs(interpolation_residual))),
        "derivative_power_squared_min_float64": float(np.min(p1_squared)),
        "derivative_power_mesh_max": float(np.max(p1)),
        "derivative_power_uniform_upper": float(np.max(p1)+correction_per_unit_norm),
        "origin_offset": origin_offset,
        "mesh_model_correction": model_correction,
        "mesh_uncertainty_correction": uncertainty_correction,
        "mesh_joint_correction": joint_correction,
        **quantities,
        "maximizer_locations": {key: float(mesh[i]) for key, i in idx.items()},
        "fine_mesh_uncorrected_maxima": fine_values,
        "fine_mesh_below_corrected_allowances": True,
        "high_precision_audit": high_precision,
    }
    for name in ("learned_model", "true_plant"):
        q = quantities[name+"_contraction_upper"]
        result[name+"_recertifies_existing_q"] = q <= EXISTING_Q
        result[name+"_margin_to_existing_q"] = EXISTING_Q-q
        result[name+"_q_below_one"] = q < 1.0
        result[name+"_tube_budget_margin"] = (1.0-q)*rho-value_error
        result[name+"_minimum_radius"] = value_error/(1.0-q) if q < 1.0 else None
    return result


def run_derivative_audit(data_dir, output_dir) -> dict:
    """Write and return derivative bounds from data_dir/summary.json.

    Only saved centers, exact labels and fixed benchmark kernel/prior
    parameters are consumed. The source plant formula is not evaluated.
    Output paths are entirely controlled by output_dir.
    """
    source_path = Path(data_dir).resolve()/"summary.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    designs = sorted(source["selected_designs"], key=lambda d: int(d["centers"]))
    if [int(d["centers"]) for d in designs] != [9, 17, 33, 66]:
        raise ValueError("Expected saved accepted models with N=9,17,33,66")
    mesh = np.linspace(-1.0, 1.0, 4097)
    fine = np.linspace(-1.0, 1.0, 8193)
    with threadpool_limits(limits=1):
        results = [certify_one(design, mesh, fine) for design in designs]
    output = {
        "source_summary": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scope": "Saved-data derivative audit only; no closed-loop trajectories regenerated.",
        "assumptions": [
            "Exact labels and unregularized interpolation at the saved distinct centers.",
            "The true residual belongs to the fixed Matérn RKHS with norm at most 0.5.",
            "No assumed numerical Lipschitz constant of the true residual is used.",
            "Convex scalar learning domain [-1,1].",
        ],
        "kernel": {"family": "Matern 5/2", "variance": 1.0, "length_scale": ELL,
                   "first_derivative_representer_norm_squared": 5.0/(3.0*ELL**2),
                   "derivative_feature_lipschitz": 5.0/ELL**2},
        "kernel_constant_audit": verify_kernel_constants(),
        "rkhs_norm_budget": RKHS_BUDGET,
        "nominal_closed_loop_coefficient": NOMINAL_CLOSED_LOOP,
        "existing_q_to_check": EXISTING_Q,
        "evaluation_mesh_points": len(mesh), "covering_radius": 1.0/4096.0,
        "independent_finer_mesh_points": len(fine),
        "norm_and_offset_scope": (
            "The norm and orthogonal residual budget belong to the unshifted exact "
            "interpolant. The simulation's constant origin correction has zero "
            "derivative, so derivative bounds are unchanged. Function-error bounds "
            "must separately include the absolute origin offset, as in the saved study."),
        "arithmetic_scope": (
            "Exact-arithmetic continuous-domain inequalities evaluated in float64; "
            "60-digit Decimal checks at grid maximizers and an 8193-point mesh audit "
            "support numerical reliability. These are not formal interval enclosures."),
        "is_formal_interval_certificate": False,
        "models": results,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path/"derivative-certification.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = ["% Generated by analysis/derivative_audit.py; floating-point audit.",
            r"\begin{tabular}{rccccc}", r"\hline",
            r"$N$ & $m_N$ & $\eta_N$ & $\overline L_N$ & $\hat q_N$ & $q_N^{\rm true}$\\",
            r"\hline"]
    for item in results:
        rows.append(f"{item['centers']} & {item['rkhs_norm']:.4f} & "
                    f"{item['derivative_error_upper']:.4f} & "
                    f"{item['true_residual_lipschitz_upper']:.4f} & "
                    f"{item['learned_model_contraction_upper']:.4f} & "
                    f"{item['true_plant_contraction_upper']:.4f} "+r"\\")
    rows.extend([r"\hline", r"\end{tabular}", ""])
    (output_path/"derivative-table.tex").write_text("\n".join(rows), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_derivative_audit(args.data, args.output)
    for item in result["models"]:
        print(f"N={item['centers']:2d}: eta={item['derivative_error_upper']:.8f}, "
              f"Lip(r)<={item['true_residual_lipschitz_upper']:.8f}, "
              f"qhat={item['learned_model_contraction_upper']:.8f}, "
              f"qtrue={item['true_plant_contraction_upper']:.8f}, "
              f"q=.7: learned={item['learned_model_recertifies_existing_q']}, "
              f"true={item['true_plant_recertifies_existing_q']}")


if __name__ == "__main__":
    main()
