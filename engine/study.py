"""Public numerical-study API and original CSV/summary accounting.

Run ``run_study(output_dir, weights=..., starts=..., keys=..., steps=...)``.
Geometry and labels are regenerated from the stated kernel and plant. The
engine reads no archived results and performs no plotting at import time.
"""
from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits
from . import experiment as ex
from . import state_dependent_baseline as sd

WEIGHTS = (1., 2., 4., 8., 16.)
RADII = (.15, .075)
STARTS = (-.75, -.5, -.25, .25, .5, .75)
KINDS = ("fill", "power")
KEYS = ("bounded", "state_dependent", "fill_015", "power_015", "fill_0075", "power_0075")
ALL_KEYS = KEYS
PRIMARY_KEYS = ("bounded", "state_dependent", "power_015", "power_0075")
CONFERENCE_KEYS = ("state_dependent", "power_015", "power_0075")
CANDIDATE_POINTS = 2049
POWER_MESH_POINTS = 4097
Q = ex.LEARNED_CONTRACTION_CAP
CSV_HEADER = "step,x,u,z0,v0,plan_cost,next_prediction_deviation,solve_ms,fallback,shift_violation,true_residual,actual_stage_cost,actual_accumulated_cost,model_residual,current_tube_radius,next_tube_radius,feedback_input"

def key_for(kind,rho):
    return f"{kind}_{'015' if rho == .15 else '0075'}"


def fill_certificate(model):
    h = ex.core.fill_distance(model.centers)
    covariance = float(ex.core.kernel([0],[h])[0,0])
    return ex.RKHS_BUDGET*math.sqrt(max(0.,2*(1-covariance)))+abs(model.origin_offset)


def power_certificate(model):
    mesh = np.linspace(-1,1,4097)
    delta = float(np.max(np.diff(mesh))/2)
    sampled_max = float(np.max(model.power(mesh)))
    return ex.RKHS_BUDGET*(sampled_max+ex.core.FEATURE_LIP*delta)+abs(model.origin_offset)


class FixedRadiusController(ex.Controller):
    def __init__(self,name,model,epsilon,rho,input_weight=1.):
        super().__init__(name,model,{"epsilon":epsilon},input_weight)
        self.learned_contraction = learned_contraction_certificate(model)
        if self.learned_contraction["bound"] > self.q:
            raise ValueError("The fitted predictor does not pass the fixed contraction cap")
        if epsilon > (1-self.q)*rho+1e-12:
            raise ValueError("Chosen certificate does not meet the fixed target")
        self.rho = float(rho)
        self.zmax = 1-self.rho
        self.vmax = ex.INPUT_LIMIT-abs(ex.GAIN)*self.rho
        self.reachable = self.backward_intervals()
        self.cache = None


def learned_contraction_certificate(model):
    """Bound |A+K+rhat'(x)| uniformly, without a true-residual slope prior.

    For the unit-variance Matérn 5/2 kernel, ||k''(.,x)||_H=5/ell^2.
    Therefore the fitted RKHS norm supplies the Hessian covering allowance.
    These are float64 evaluations of continuous-domain inequalities, not
    interval-arithmetic enclosures of every rounding error.
    """
    mesh = np.linspace(-1., 1., POWER_MESH_POINTS)
    delta = float(np.max(np.diff(mesh))/2)
    sampled = float(np.max(abs(ex.A+ex.GAIN+model.derivative(mesh))))
    curvature = (5/ex.core.ELL**2)*model.norm
    allowance = float(curvature*delta)
    return {"bound": sampled+allowance, "mesh_maximum": sampled,
            "covering_allowance": allowance, "curvature_bound": float(curvature),
            "mesh_points": POWER_MESH_POINTS, "covering_radius": delta}


def select_models():
    """Regenerate the original nested P-greedy prefixes and four stop tests.

    The candidate pool has 2049 equally spaced points including both endpoints.
    Origin is mandatory. Full Cholesky refits and numpy.argmax preserve the
    original tie behavior; no historical center sequence is read from disk.
    """
    candidates = np.linspace(-1., 1., CANDIDATE_POINTS)
    centers = [0.0]
    observed = {0.0: float(ex.observe(np.array([0.0]))[0])}
    selected, log = {}, []
    for _ in range(len(candidates)):
        center_array = np.array(centers)
        labels = np.array([observed[x] for x in centers])
        model = ex.LearnedModel(center_array, labels)
        if model.norm > ex.RKHS_BUDGET+1e-8:
            raise AssertionError("Interpolant norm contradicts the supplied budget")
        values = {"fill": fill_certificate(model), "power": power_certificate(model)}
        log.append({"centers": len(centers), **values,
                    "fill_distance": ex.core.fill_distance(center_array), "norm": model.norm})
        contraction = None
        for rho in RADII:
            for kind in KINDS:
                key = key_for(kind, rho)
                if key in selected or values[kind] > (1-Q)*rho:
                    continue
                if contraction is None:
                    contraction = learned_contraction_certificate(model)
                if contraction["bound"] > Q:
                    continue
                ctrl = FixedRadiusController(key, model, values[kind], rho)
                timings = []
                function = fill_certificate if kind == "fill" else power_certificate
                for _ in range(7):
                    start = perf_counter()
                    function(model)
                    timings.append(1000*(perf_counter()-start))
                grid = np.linspace(-1., 1., 16001)
                metadata = {
                    "key": key, "certificate": kind, "centers": len(centers),
                    "requested_radius": rho, "used_radius": ctrl.rho, "q": Q,
                    "target_error": (1-Q)*rho, "certified_error": values[kind],
                    "fill_bound": values["fill"], "uniform_power_bound": values["power"],
                    "minimum_supported_radius": values[kind]/(1-Q),
                    "measured_grid_error": float(np.max(abs(ex.residual(grid)-model.predict(grid)))),
                    "model_norm": model.norm, "gram_condition": model.gram_condition,
                    "nominal_state_limit": ctrl.zmax, "nominal_input_limit": ctrl.vmax,
                    "feasible_initial_interval": ctrl.actual_feasible_interval,
                    "certificate_eval_median_ms": float(np.median(timings)),
                    "center_locations": centers.copy(), "training_labels": labels.tolist(),
                    "learned_jacobian_contraction_bound": contraction["bound"],
                    "learned_jacobian_certificate": contraction.copy(),
                }
                selected[key] = (model, metadata)
        if len(selected) == len(RADII)*len(KINDS):
            return selected, log
        if len(centers) == len(candidates):
            break
        powers = model.power(candidates)
        for xi in centers:
            powers[np.argmin(abs(candidates-xi))] = -1
        xi = float(candidates[int(np.argmax(powers))])
        centers.append(xi)
        observed[xi] = float(ex.observe(np.array([xi]))[0])
    raise RuntimeError("The candidate pool cannot certify every requested target")


def controller(key,selected,weight):
    if key == "bounded":
        return ex.Controller("bounded",ex.ZeroModel(),{"epsilon":ex.DISTURBANCE_BOUND},weight)
    if key == "state_dependent":
        return sd.VariableTubeController(weight)
    model,meta = selected[key]
    return FixedRadiusController(key,model,meta["certified_error"],meta["requested_radius"],weight)


def run_one(key,ctrl,x0,weight,output_dir,steps=ex.STEPS):
    if key == "state_dependent":
        data,xf,_ = sd.simulate(ctrl,x0,steps=steps)
        radii0,radii1 = data[:,14],data[:,15]
        full = data[:,:14]
    else:
        data,xf = ctrl.simulate(x0,steps=steps)
        true_r = ex.residual(data[:,1])
        model_r = ctrl.model.predict(data[:,1])
        stage = data[:,1]**2+weight*data[:,2]**2
        full = np.column_stack([data,true_r,stage,np.cumsum(stage),model_r])
        radii0 = np.full(len(data),ctrl.rho)
        radii1 = np.full(len(data),ctrl.rho)
    feedback = ex.GAIN*(full[:,1]-full[:,3])
    nominal = full[:,4]
    np.testing.assert_allclose(full[:,2],nominal+feedback,atol=2e-12)
    state_energy = float(full[:,1]@full[:,1])
    input_energy = float(full[:,2]@full[:,2])
    nominal_energy = float(nominal@nominal)
    feedback_energy = float(feedback@feedback)
    cross = float(2*(nominal@feedback))
    np.testing.assert_allclose(input_energy,nominal_energy+feedback_energy+cross,atol=2e-12)
    np.testing.assert_allclose(full[-1,12],state_energy+weight*input_energy,atol=2e-12)
    nominal_pred = float(np.sum(full[:,3]**2+weight*nominal**2))
    trajectory = np.r_[full[:,1],xf]
    settled = next((i for i in range(len(trajectory)) if np.all(abs(trajectory[i:])<=.01)),None)
    weight_token = str(int(weight)) if float(weight).is_integer() else format(weight, ".12g").replace(".", "p")
    rid = f"w{weight_token}_{key}_x{str(x0).replace('-','m').replace('.','p')}"
    np.savetxt(Path(output_dir)/f"{rid}.csv",np.column_stack([full,radii0,radii1,feedback]),delimiter=",",comments="",
        header=CSV_HEADER)
    return {"run_id":rid,"controller":key,"input_weight":weight,"initial_state":x0,
            "steps":len(full),"final_state":xf,"actual_cost":float(full[-1,12]),
            "state_energy":state_energy,"input_energy":input_energy,
            "applied_nominal_input_energy":nominal_energy,"feedback_input_energy":feedback_energy,
            "input_cross_term":cross,"nominal_stage_sum_along_closed_loop":nominal_pred,
            "max_model_error_on_run":float(np.max(abs(full[:,10]-full[:,13]))),
            "peak_current_deviation":float(np.max(abs(full[:,1]-full[:,3]))),
            "rms_current_deviation":float(np.sqrt(np.mean((full[:,1]-full[:,3])**2))),
            "peak_next_deviation":float(np.max(full[:,6])),"peak_input":float(np.max(abs(full[:,2]))),
            "settled_step":settled,"fallbacks":int(np.sum(full[:,8])),
            "max_shift_violation":float(np.max(full[:,9])),
            "median_solve_ms":float(np.median(full[:,7]))}


def aggregate(runs,weights=WEIGHTS,keys=KEYS):
    result = []
    for weight in weights:
        baseline = [r for r in runs if r["controller"] == "state_dependent" and r["input_weight"] == weight]
        bc = float(np.mean([r["actual_cost"] for r in baseline])) if baseline else None
        be = float(np.mean([r["input_energy"] for r in baseline])) if baseline else None
        for key in keys:
            group = [r for r in runs if r["controller"] == key and r["input_weight"] == weight]
            fields = ("actual_cost","state_energy","input_energy","applied_nominal_input_energy",
                      "feedback_input_energy","input_cross_term","nominal_stage_sum_along_closed_loop")
            means = {f"mean_{field}":float(np.mean([r[field] for r in group])) for field in fields}
            result.append({"controller":key,"input_weight":weight,"cohort_count":len(group),**means,
                "actual_cost_gain_vs_prior_percent":float(100*(1-means["mean_actual_cost"]/bc)) if bc else None,
                "input_energy_gain_vs_prior_percent":float(100*(1-means["mean_input_energy"]/be)) if be else None,
                "peak_current_deviation":max(r["peak_current_deviation"] for r in group),
                "peak_next_deviation":max(r["peak_next_deviation"] for r in group),
                "max_model_error_on_runs":max(r["max_model_error_on_run"] for r in group),
                "total_fallbacks":sum(r["fallbacks"] for r in group),
                "median_final_abs_state":float(np.median([abs(r["final_state"]) for r in group]))})
    return result


def run_study(output_dir, *, weights=WEIGHTS, starts=STARTS,
              keys=PRIMARY_KEYS, steps=ex.STEPS, progress=print):
    """Regenerate models, simulate a matched cohort, and save its raw results.

    All four model-selection designs are retained, including fill diagnostics,
    even when only a subset of controllers is simulated. Reusing fitted models
    across input weights requires no additional noiseless labels. ``steps``
    controls the closed-loop duration; the MPC horizon remains twelve.
    """
    weights = tuple(float(w) for w in weights)
    starts = tuple(float(x) for x in starts)
    keys = tuple(keys)
    if not weights or any(not np.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Weights must be nonempty, finite, and positive")
    if len(set(weights)) != len(weights):
        raise ValueError("Weights must be distinct to avoid overwriting runs")
    if not starts or any(not np.isfinite(x) or abs(x) > 1 for x in starts):
        raise ValueError("Starts must be nonempty finite states in [-1, 1]")
    if len(set(starts)) != len(starts):
        raise ValueError("Starts must be distinct to avoid overwriting runs")
    if not keys or len(set(keys)) != len(keys) or any(key not in KEYS for key in keys):
        raise ValueError("Choose distinct controller keys from engine.study.KEYS")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    notify = progress if progress is not None else lambda message: None
    with threadpool_limits(limits=1):
        selected, selection_log = select_models()
        designs = [selected[key][1] for key in KEYS[2:]]
        notify("Selected models: "+json.dumps([
            {field: design[field] for field in
             ("key", "centers", "requested_radius", "certified_error")}
            for design in designs]))
        runs = []
        for weight in weights:
            for key in keys:
                ctrl = controller(key, selected, weight)
                for x0 in starts:
                    runs.append(run_one(key, ctrl, x0, weight, output_dir, steps))
            notify(f"Completed fresh optimizations for input weight {weight:g}")
        summary = {
            "plant": "x_next=1.1x+u-0.2sin(x)", "weights": weights,
            "target_radii": RADII, "starts": starts, "steps_per_run": int(steps),
            "q_learned": Q, "selected_designs": designs,
            "selection_history": selection_log,
            "aggregate": aggregate(runs, weights, keys), "runs": runs,
            "total_recorded_steps": sum(run["steps"] for run in runs),
            "power_mesh_points": POWER_MESH_POINTS,
            "power_mesh_covering_radius": 1/4096,
            "feature_lipschitz_constant": ex.core.FEATURE_LIP,
            "radius_scope": "Use the exact prescribed target in each certificate pair, not the minimum radius supported by each certificate.",
            "objective_scope": "All controllers penalize nominal z and v; actual u=v+K(x-z) is used for evaluation. Feedback effort is recorded, not added to the optimization objective.",
            "certificate_scope": "Compare an explicit nearest-center fill bound with a uniform mesh-corrected power bound; not a claim against all fill-distance methods.",
            "data_scope": "Shared noiseless nested sample prefixes regenerated locally; no repeated physical samples are needed when reusing a prefix across weight settings.",
            "controllers": keys, "horizon": ex.HORIZON,
            "state_limit": 1.0, "input_limit": ex.INPUT_LIMIT, "gain": ex.GAIN,
            "RKHS_norm_prior": ex.RKHS_BUDGET,
            "RKHS_norm_derivation": ex.core.norm_budget_derivation(),
            "prior_only_residual_slope_bound": ex.RESIDUAL_LIP,
            "learned_contraction_scope": "The fixed cap 0.7 is checked using the fitted derivative plus an RKHS Hessian mesh-covering allowance; a true-residual Lipschitz prior is not used for this learned-model check.",
            "solver_scope": "Local SLSQP solutions are accepted only when feasible and no more costly than a feasible shifted candidate; otherwise that candidate is retained. Global nonlinear optimality is not claimed.",
            "numerical_scope": "Float64 evaluations of continuous-domain inequalities; not interval-arithmetic verification. Timings and solver details depend on the numerical environment.",
            "greedy_geometry": {
                "candidate_points": CANDIDATE_POINTS, "interval": [-1.0, 1.0],
                "mandatory_first_center": 0.0, "selection": "full-Cholesky P-greedy; numpy.argmax first-index tie behavior",
                "selected_prefix_locations": max(designs, key=lambda d: d["centers"])["center_locations"],
                "source": "Regenerated from kernel values and candidate inputs; no saved geometry was loaded.",
            },
        }
        indexed = {(row["input_weight"], row["controller"]): row
                   for row in summary["aggregate"]}
        summary["pair_comparisons"] = []
        for rho in RADII:
            if not all(key_for(kind, rho) in keys for kind in KINDS):
                continue
            changes = [100*(indexed[weight, key_for("power", rho)]["mean_actual_cost"] /
                            indexed[weight, key_for("fill", rho)]["mean_actual_cost"]-1)
                       for weight in weights]
            summary["pair_comparisons"].append({
                "target_radius": rho,
                "cost_change_power_vs_fill_percent_by_weight": changes,
                "max_abs_mean_cost_change_percent": max(abs(value) for value in changes),
            })
        (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
