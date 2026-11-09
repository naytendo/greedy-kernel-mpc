"""Nonlearning variable-tube baseline using r(0)=0 and |r(x)| <= 0.2 |x|.

The controller has no access to the truth or training labels. The runner uses
truth only for simulation and recorded diagnostics.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from . import experiment as ex

TOL = 2e-7

class VariableTubeController:
    def __init__(self, input_weight=1., horizon=ex.HORIZON):
        self.H = int(horizon)
        self.A, self.K, self.L = ex.A, ex.GAIN, ex.RESIDUAL_LIP
        self.q = abs(self.A+self.K)+self.L
        self.weight = float(input_weight)
        self.terminal_radius = min(1., ex.INPUT_LIMIT/abs(self.K))
        self.size = 2*(self.H+1)
        identity = np.eye(self.size)
        # y = (z0, v0,...,v[H-1], s0,...,sH); all z_i are eliminated.
        self.Z = np.zeros((self.H+1, self.size))
        self.Z[0, 0] = 1.
        self.V = identity[1:self.H+1]
        self.S = identity[self.H+1:]
        for i in range(self.H):
            self.Z[i+1] = self.A*self.Z[i]+self.V[i]
        self.Q = self.Z[:-1].T@self.Z[:-1]+self.weight*self.V.T@self.V
        self.C = np.vstack([self.Z+self.S, -self.Z+self.S,
                            self.V+abs(self.K)*self.S[:-1],
                            -self.V+abs(self.K)*self.S[:-1],
                            self.q*self.S[:-1]+self.L*self.Z[:-1]-self.S[1:],
                            self.q*self.S[:-1]-self.L*self.Z[:-1]-self.S[1:],
                            self.S[-1]])
        self.b = np.r_[np.ones(2*(self.H+1)),
                       ex.INPUT_LIMIT*np.ones(2*self.H),
                       np.zeros(2*self.H), self.terminal_radius]
        self.E = self.Z[-1:]
        self.lower = np.r_[-1., -ex.INPUT_LIMIT*np.ones(self.H), np.zeros(self.H+1)]
        self.upper = np.r_[1., ex.INPUT_LIMIT*np.ones(self.H), np.ones(self.H+1)]
        self.bounds = list(zip(self.lower, self.upper))
        # Projection is an interval: constraints are convex and sign-symmetric.
        direction = self.Z[0]+self.S[0]
        support = linprog(-direction, A_ub=self.C, b_ub=self.b,
                          A_eq=self.E, b_eq=[0.], bounds=self.bounds, method="highs")
        if not support.success:
            raise RuntimeError(f"Cannot calculate initial set: {support.message}")
        self.initial_halfwidth = min(1., float(direction@support.x))
        self.actual_feasible_interval = (-self.initial_halfwidth, self.initial_halfwidth)

    def cost(self, y):
        return float(y@self.Q@y)

    def gradient(self, y):
        return 2*self.Q@y

    def initial_constraints(self, x):
        rows = np.vstack([self.Z[0]-self.S[0], -self.Z[0]-self.S[0]])
        return np.vstack([self.C, rows]), np.r_[self.b, x, -x]

    def violation(self, y, x):
        c, b = self.initial_constraints(x)
        return float(max(0., np.max(c@y-b), np.max(abs(self.E@y)),
                         np.max(self.lower-y), np.max(y-self.upper)))

    def initial_plan(self, x):
        c, b = self.initial_constraints(x)
        # This LP only supplies the first feasible warm start. Subsequent calls
        # use the shifted plan; no second online robust-feedback OCP is solved.
        seed = linprog(np.sum(self.S, axis=0), A_ub=c, b_ub=b,
                       A_eq=self.E, b_eq=[0.], bounds=self.bounds, method="highs")
        if not seed.success:
            raise ValueError(f"Initial state {x} not feasible: {seed.message}")
        return seed.x

    def shift(self, y):
        z, s = self.Z@y, self.S@y
        return np.r_[z[1], y[2:self.H+1], 0., s[1:], self.q*s[-1]]

    def minimal_radii(self, y, x):
        # Radii have no cost, so the QP may leave arbitrary slack. Replace
        # them by the smallest recursively valid radii for the SAME z,v plan.
        # This only relaxes state/input constraints and leaves cost unchanged.
        result = y.copy()
        z = self.Z@result
        s = np.empty(self.H+1)
        s[0] = abs(x-z[0])
        for i in range(self.H):
            s[i+1] = self.q*s[i]+self.L*abs(z[i])
        result[self.H+1:] = s
        return result

    def solve(self, x, warm=None):
        if warm is None:
            warm = self.initial_plan(x)
        if self.violation(warm, x) > TOL:
            raise AssertionError("Variable-tube shifted plan is infeasible")
        c, b = self.initial_constraints(x)
        result = minimize(self.cost, warm, jac=self.gradient, method="SLSQP",
                          bounds=Bounds(self.lower, self.upper),
                          constraints=[LinearConstraint(c, -np.inf, b),
                                       LinearConstraint(self.E, 0., 0.)],
                          options={"ftol": 1e-11, "maxiter": 300})
        accepted = (result.success and self.violation(result.x, x) <= TOL
                    and self.cost(result.x) <= self.cost(warm)+1e-9)
        selected = self.minimal_radii(result.x if accepted else warm.copy(),x)
        if self.violation(selected,x) > TOL:
            raise AssertionError("Minimal-radius postprocessing violated feasibility")
        return selected, not accepted


def simulate(controller, x0, plant_step=ex.plant, steps=ex.STEPS):
    x, warm, previous, rows, plans = float(x0), None, None, [], []
    for k in range(steps):
        start = perf_counter()
        y, fallback = controller.solve(x, warm)
        elapsed = 1000*(perf_counter()-start)
        z, v, s = controller.Z@y, controller.V@y, controller.S@y
        u = float(v[0]+controller.K*(x-z[0]))
        xn = float(plant_step(x, u))
        shifted = controller.shift(y)
        defect = abs(xn-z[1])
        cost = controller.cost(y)
        stage_nom = z[0]**2+controller.weight*v[0]**2
        remaining = controller.violation(shifted, xn)
        checks = [abs(x-z[0])-s[0], defect-s[1], remaining,
                  abs(u)-ex.INPUT_LIMIT, abs(x)-1, abs(xn)-1,
                  defect-(controller.q*abs(x-z[0])+controller.L*abs(z[0])),
                  controller.cost(shifted)-cost+stage_nom]
        if previous is not None:
            checks.append(cost-previous[0]+previous[1])
        if max(checks) > TOL:
            raise AssertionError(f"Variable-tube closed-loop check failed: {checks}")
        actual_r = xn-controller.A*x-u
        actual_stage = x*x+controller.weight*u*u
        cumulative = actual_stage+(rows[-1][12] if rows else 0.)
        # First 14 columns match the original experiment CSV schema.
        rows.append([k,x,u,z[0],v[0],cost,defect,elapsed,int(fallback),remaining,
                     actual_r,actual_stage,cumulative,0.,s[0],s[1],np.max(s)])
        plans.append({"step": k, "nominal_states": z.tolist(),
                      "nominal_inputs": v.tolist(), "tube_radii": s.tolist()})
        previous, x, warm = (cost,stage_nom), xn, shifted
    return np.array(rows), x, plans
