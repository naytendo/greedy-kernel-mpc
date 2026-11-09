"""Plant, interpolation wrappers, and constant-tube controllers.

The nonlinear optimizers retain a feasible shifted plan unless a successful,
feasible local SLSQP solution has no greater nominal objective. Global nonlinear
optimality is not claimed.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from . import core

A = 1.1
AMPLITUDE = 0.2
DISTURBANCE_BOUND = 0.2
RKHS_BUDGET = 0.5
RESIDUAL_LIP = 0.2  # Prior-only baseline information; not a learned slope estimate.
INPUT_LIMIT = 0.3
GAIN = -0.6
HORIZON = core.HORIZON
STEPS = 30
COST_WEIGHTS = (1.0, 4.0)
LEARNED_CONTRACTION_CAP = 0.7

def residual(x):
    return -AMPLITUDE*np.sin(np.asarray(x))


def plant(x, u):
    return A*x+u+float(residual([x])[0])


def observe(x):
    # Reset experiments. Under r(0)=0 and L_r<=0.2 this admissible input gives
    # |x_next| <= |x|, even before the sign of the residual is learned.
    x = np.asarray(x)
    u = -.3*x
    successor = A*x+u+residual(x)
    return successor-A*x-u


class ZeroModel:
    norm = 0.0
    origin_offset = 0.0
    gram_condition = 1.0

    def __init__(self):
        self.centers = np.array([])
        self.points_evaluated = 0

    def predict(self, x):
        x = np.asarray(x).reshape(-1)
        self.points_evaluated += len(x)
        return np.zeros(len(x))

    derivative = predict


class LearnedModel(core.Interpolant):
    def __post_init__(self):
        self.points_evaluated = 0
        super().__post_init__()

    def predict(self, x):
        self.points_evaluated += np.asarray(x).size
        return super().predict(x)

    def derivative(self, x):
        self.points_evaluated += np.asarray(x).size
        return super().derivative(x)


class Controller(core.TubeMPC):
    def __init__(self, name, model, bound, input_weight=1.0):
        self.name, self.model, self.bound, self.input_weight = name, model, bound, input_weight
        self.gain = GAIN
        self.A = A
        # Give the baseline its own tighter nominal contraction, rather than
        # incorrectly assigning it the learned model's 0.7 factor.
        self.q = abs(A+GAIN) if name == "bounded" else LEARNED_CONTRACTION_CAP
        self.rho = bound["epsilon"]/(1-self.q)
        self.cert = {"eps_fill": bound["epsilon"]}
        self.zmax = 1-self.rho
        self.vmax = INPUT_LIMIT-abs(GAIN)*self.rho
        if self.q >= 1 or self.zmax <= 0 or self.vmax <= 0:
            raise ValueError("No usable tube/constraint margin for this design")
        self.cache = None
        self.reachable = self.backward_intervals()

    def backward_intervals(self):
        # The fitted norm is known. Its derivative representer gives a valid
        # monotonicity bound for this specific frozen predictor.
        lower_derivative = A-core.FEATURE_LIP*self.model.norm
        if lower_derivative <= 0:
            raise AssertionError("The predictor needs a sharper monotonicity certificate")
        intervals = [(0.0, 0.0)]
        for _ in range(HORIZON):
            low, high = intervals[-1]
            target_low, target_high = low-self.vmax, high+self.vmax
            fl, fr = self.drift(-self.zmax), self.drift(self.zmax)
            left = -self.zmax if fl >= target_low else brentq(lambda z:self.drift(z)-target_low, -self.zmax, self.zmax, xtol=1e-13)
            right = self.zmax if fr <= target_high else brentq(lambda z:self.drift(z)-target_high, -self.zmax, self.zmax, xtol=1e-13)
            intervals.append((float(left), float(right)))
        return intervals

    def cost(self, y):
        z, _ = self.rollout(y)
        return float(z[:-1]@z[:-1]+self.input_weight*(y[1:]@y[1:]))

    def gradient(self, y):
        z, jac = self.rollout(y)
        return 2*jac[:-1].T@z[:-1]+np.r_[0, 2*self.input_weight*y[1:]]

    def simulate(self, x0, steps=STEPS):
        x, warm, previous, records = float(x0), None, None, []
        self.model.points_evaluated = 0
        for k in range(steps):
            start = perf_counter()
            y, fallback = self.solve(x, warm)
            solve_ms = 1000*(perf_counter()-start)
            z, _ = self.rollout(y)
            u = float(y[1]+GAIN*(x-z[0]))
            xn = plant(x, u)
            j, stage_nominal = self.cost(y), z[0]**2+self.input_weight*y[1]**2
            shifted = self.shift(y)
            violation = self.violation(shifted, xn)
            defect = abs(xn-z[1])
            checks = [self.q*abs(x-z[0])+self.bound["epsilon"]-self.rho,
                      defect-self.rho, abs(x)-1, abs(xn)-1, abs(u)-INPUT_LIMIT,
                      violation, self.cost(shifted)-j+stage_nominal]
            if previous is not None:
                checks.append(j-previous[0]+previous[1])
            if max(checks) > 1e-7:
                raise AssertionError(f"Closed-loop check failed for {self.name}: {checks}")
            records.append([k, x, u, z[0], y[1], j, defect, solve_ms, int(fallback), violation])
            previous, x, warm = (j, stage_nominal), xn, shifted
        return np.array(records), x
