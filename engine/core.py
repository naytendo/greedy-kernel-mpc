"""Matérn 5/2 interpolation and scalar tube-MPC numerical primitives.

Adapted from the original numerical study. All geometry is regenerated locally;
no archived simulation outputs or machine-specific imports are used.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

ELL = 0.5
A_KERNEL = math.sqrt(5) / ELL
FEATURE_LIP = A_KERNEL / math.sqrt(3)
RKHS_BUDGET = 0.5
STATE_LIMIT = 1.0
HORIZON = 12
TOL = 2e-8


def norm_budget_derivation(amplitude=0.2, budget=0.5):
    """Analytical extension bound for amplitude*sin(x) on [-1, 1].

    A C2 compactly supported cutoff gives an H3 whole-line extension. The
    supplied norm budget is a prior, never inferred from fitted labels.
    """
    p = [Fraction(v) for v in (1, 0, 0, -10, 15, -6)]
    norms_sq = []
    for order in range(4):
        integral = sum((a*b/Fraction(i+j+1)
                        for i, a in enumerate(p) for j, b in enumerate(p)), Fraction(0))
        norms_sq.append(2*integral + (2 if order == 0 else 0))
        p = [Fraction(i)*p[i] for i in range(1, len(p))]
    chi_norms = np.sqrt([float(v) for v in norms_sq])
    derivative_bounds = np.array([
        abs(amplitude)*sum(math.comb(j, k)*chi_norms[k] for k in range(j+1))
        for j in range(4)
    ])
    a = A_KERNEL
    weights = np.array([a**6, 3*a**4, 3*a*a, 1.0])
    upper = math.sqrt(3/(16*a**5)*float(weights @ derivative_bounds**2))
    if upper > budget:
        raise ValueError("The supplied RKHS norm budget is too small for this extension")
    return {
        "cutoff_derivative_squared_norms_exact": [str(v) for v in norms_sq],
        "extension_derivative_L2_upper_bounds": derivative_bounds.tolist(),
        "whole_line_RKHS_norm_upper_bound": upper,
        "supplied_RKHS_budget": budget,
        "fourier_convention": "f_hat(omega)=integral f(x) exp(-i omega x) dx",
        "kernel_spectrum": "16 a^5 / [3 (a^2 + omega^2)^3]",
    }

def kernel(x, centers):
    d = np.abs(np.asarray(x, dtype=float).reshape(-1, 1)
               - np.asarray(centers, dtype=float).reshape(1, -1))
    t = A_KERNEL * d
    return (1 + t + t*t/3) * np.exp(-t)


def kernel_dx(x, centers):
    d = np.asarray(x, dtype=float).reshape(-1, 1) - np.asarray(centers).reshape(1, -1)
    t = A_KERNEL * np.abs(d)
    return -(A_KERNEL**2/3) * d * (1 + t) * np.exp(-t)


def fill_distance(centers):
    c = np.sort(centers)
    return float(max(c[0]+1, 1-c[-1], np.max(np.diff(c))/2 if len(c) > 1 else 0))


@dataclass
class Interpolant:
    centers: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        self.gram = kernel(self.centers, self.centers)
        self.cholesky = np.linalg.cholesky(self.gram)
        self.theta = solve_triangular(self.cholesky, self.labels, lower=True)
        self.alpha = cho_solve((self.cholesky, True), self.labels)
        self.origin_offset = float(self.raw_predict([0])[0])
        self.norm = float(np.linalg.norm(self.theta))
        self.gram_condition = float(np.linalg.cond(self.gram))

    def raw_predict(self, x):
        return kernel(x, self.centers) @ self.alpha

    def predict(self, x):
        # In exact arithmetic origin_offset is zero. Include its magnitude in
        # the approximation allowance while making the numerical equilibrium exact.
        return self.raw_predict(x) - self.origin_offset

    def derivative(self, x):
        return kernel_dx(x, self.centers) @ self.alpha

    def power(self, x):
        features = solve_triangular(self.cholesky, kernel(x, self.centers).T, lower=True)
        return np.sqrt(np.maximum(0, 1 - np.sum(features*features, axis=0)))

    def certificate(self, mesh):
        h = fill_distance(self.centers)
        phi_h = float(kernel([0], [h])[0, 0])
        eps_fill = RKHS_BUDGET * math.sqrt(max(0, 2*(1-phi_h))) + abs(self.origin_offset)
        delta = float(np.max(np.diff(mesh))/2)
        mesh_max = float(np.max(self.power(mesh)))
        eps_power = RKHS_BUDGET*(mesh_max + FEATURE_LIP*delta) + abs(self.origin_offset)
        return {
            "centers": len(self.centers), "fill_distance": h,
            "eps_fill": eps_fill, "eps_uniform_power": eps_power,
            "power_mesh_max": mesh_max, "power_mesh_covering_radius": delta,
            "origin_offset": self.origin_offset, "interpolant_norm": self.norm,
            "gram_condition": self.gram_condition,
        }


class TubeMPC:
    """Shared rollout and feasible-shift acceptance methods (SLSQP local solves)."""

    def drift(self, z):
        return self.A*z + float(self.model.predict([z])[0])

    @property
    def actual_feasible_interval(self):
        lo, hi = self.reachable[-1]
        return max(-STATE_LIMIT, lo-self.rho), min(STATE_LIMIT, hi+self.rho)

    def feasible_initial_plan(self, x):
        lo, hi = self.reachable[HORIZON]
        lo, hi = max(lo, x-self.rho), min(hi, x+self.rho)
        if lo > hi+1e-12:
            raise ValueError("Initial state outside the computed feasible interval")
        z = float(np.clip(0, lo, hi))
        plan = [z]
        for remaining in range(HORIZON, 0, -1):
            flo = self.drift(z)
            lo, hi = self.reachable[remaining-1]
            lo, hi = max(lo, flo-self.vmax), min(hi, flo+self.vmax)
            zn = float(np.clip(0, lo, hi))
            plan.append(zn-flo)
            z = zn
        return np.array(plan)

    def rollout(self, y):
        if self.cache is not None and np.array_equal(y, self.cache[0]):
            return self.cache[1:]
        z = np.zeros(HORIZON+1)
        jac = np.zeros((HORIZON+1, HORIZON+1))
        z[0], jac[0, 0] = y[0], 1
        for i in range(HORIZON):
            z[i+1] = self.drift(z[i])+y[i+1]
            derivative = self.A+float(self.model.derivative([z[i]])[0])
            jac[i+1] = derivative*jac[i]
            jac[i+1, i+1] += 1
        self.cache = (y.copy(), z, jac)
        return z, jac

    def inequalities(self, y):
        z, _ = self.rollout(y)
        return np.r_[self.zmax-z[:-1], self.zmax+z[:-1]]

    def inequality_jacobian(self, y):
        _, jac = self.rollout(y)
        return np.vstack([-jac[:-1], jac[:-1]])

    def terminal(self, y):
        return self.rollout(y)[0][-1:]

    def terminal_jacobian(self, y):
        return self.rollout(y)[1][-1:]

    def violation(self, y, x):
        return float(max(0, -np.min(self.inequalities(y)), abs(self.terminal(y)[0]),
                         abs(y[0]-x)-self.rho, np.max(abs(y[1:]))-self.vmax))

    def shift(self, y):
        z, _ = self.rollout(y)
        return np.r_[z[1], y[2:], 0.0]

    def solve(self, x, warm=None):
        if warm is None:
            warm = self.feasible_initial_plan(x)
        if self.violation(warm, x) > TOL:
            raise RuntimeError("Shifted candidate is infeasible")
        candidate_cost = self.cost(warm)
        lo, hi = self.reachable[HORIZON]
        bounds = [(max(lo, x-self.rho), min(hi, x+self.rho))]
        bounds += [(-self.vmax, self.vmax)]*HORIZON
        sol = minimize(self.cost, warm, jac=self.gradient, method="SLSQP", bounds=bounds,
                       constraints=[
                           {"type": "ineq", "fun": self.inequalities, "jac": self.inequality_jacobian},
                           {"type": "eq", "fun": self.terminal, "jac": self.terminal_jacobian}],
                       options={"ftol": 1e-11, "maxiter": 180})
        accepted = (sol.success and self.violation(sol.x, x) <= TOL
                    and self.cost(sol.x) <= candidate_cost+1e-10)
        return (sol.x if accepted else warm.copy()), not accepted
