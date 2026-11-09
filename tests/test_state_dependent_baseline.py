"""Baseline uncertainty, feasibility, and shift tests adapted from the original study."""
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from engine import state_dependent_baseline as sd


def setUpModule():
    global _threads
    _threads = threadpool_limits(limits=1)


def tearDownModule():
    _threads.restore_original_limits()


class VariableTubeTests(unittest.TestCase):
    def test_controller_needs_no_truth_or_training_labels(self):
        with patch.object(sd.ex, "residual", side_effect=AssertionError("truth leaked")), \
             patch.object(sd.ex, "plant", side_effect=AssertionError("truth leaked")), \
             patch.object(sd.ex, "observe", side_effect=AssertionError("labels leaked")):
            c = sd.VariableTubeController(4.)
            plan, _ = c.solve(.9)
            self.assertLess(c.violation(plan,.9), sd.TOL)

    def test_matrix_constraints_match_scalar_tube_conditions(self):
        c = sd.VariableTubeController(4.)
        rng = np.random.default_rng(101)
        for _ in range(10):
            y = rng.normal(size=c.size)
            z,v,s = c.Z@y,c.V@y,c.S@y
            explicit = np.r_[z+s-1,-z+s-1,
                v+.6*s[:-1]-.3,-v+.6*s[:-1]-.3,
                .7*s[:-1]+.2*z[:-1]-s[1:],
                .7*s[:-1]-.2*z[:-1]-s[1:],s[-1]-.5]
            np.testing.assert_allclose(c.C@y-c.b,explicit,atol=2e-12)

    def test_feasible_interval_and_outside_rejection(self):
        c = sd.VariableTubeController()
        self.assertGreater(c.initial_halfwidth,.96)
        self.assertLess(c.initial_halfwidth,.97)
        for sign in (-1,1):
            x = sign*(c.initial_halfwidth-1e-7)
            y = c.initial_plan(x)
            self.assertLess(c.violation(y,x),sd.TOL)
            with self.assertRaises(ValueError):
                c.initial_plan(sign*(c.initial_halfwidth+1e-5))

    def test_shift_is_feasible_for_extreme_pointwise_uncertainty(self):
        c = sd.VariableTubeController(4.)
        y,_ = c.solve(.9)
        z,v,s = c.Z@y,c.V@y,c.S@y
        shifted = c.shift(y)
        for x in np.linspace(z[0]-s[0],z[0]+s[0],31):
            u = v[0]+c.K*(x-z[0])
            for error in (-c.L*abs(x),c.L*abs(x)):
                xn = c.A*x+u+error
                self.assertLessEqual(abs(xn-z[1]),s[1]+sd.TOL)
                self.assertLessEqual(abs(u),sd.ex.INPUT_LIMIT+sd.TOL)
                self.assertLess(c.violation(shifted,xn),sd.TOL)

    def test_terminal_set_invariance_for_both_uncertainty_signs(self):
        c = sd.VariableTubeController()
        for x in np.linspace(-c.terminal_radius,c.terminal_radius,101):
            self.assertLessEqual(abs(c.K*x),sd.ex.INPUT_LIMIT+1e-12)
            for error in (-c.L*abs(x),c.L*abs(x)):
                xn = (c.A+c.K)*x+error
                self.assertLessEqual(abs(xn),c.q*c.terminal_radius+1e-12)

    def test_adversarial_slope_simulations(self):
        # +0.2*x is destabilizing, unlike the report's true residual.
        for slope in (-.2,.2):
            for x in (-.9,.9):
                c = sd.VariableTubeController(4.)
                d,xf,_ = sd.simulate(c,x,lambda state,u:(1.1+slope)*state+u,steps=30)
                self.assertLess(abs(xf),.001)
                self.assertLessEqual(float(np.max(abs(d[:,2]))),.3+sd.TOL)
                self.assertLessEqual(float(np.max(d[:,6]-d[:,15])),sd.TOL)
