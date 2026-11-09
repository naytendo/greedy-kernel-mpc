"""Checks adapted from the original study, independent of archived results."""
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from engine import core, experiment as ex, study


def setUpModule():
    global _threads
    _threads = threadpool_limits(limits=1)


def tearDownModule():
    _threads.restore_original_limits()


@lru_cache(maxsize=1)
def selection():
    with threadpool_limits(limits=1):
        return study.select_models()


class KernelAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected, cls.history = selection()

    def test_supplied_norm_budget_and_reset_observations(self):
        derivation = core.norm_budget_derivation()
        self.assertLess(derivation["whole_line_RKHS_norm_upper_bound"], ex.RKHS_BUDGET)
        self.assertEqual(derivation["supplied_RKHS_budget"], .5)
        self.assertEqual(derivation["cutoff_derivative_squared_norms_exact"],
                         ["643/231", "20/7", "240/7", "1440"])
        xx = np.linspace(-1, 1, 503)
        np.testing.assert_allclose(ex.observe(xx), ex.residual(xx), atol=1e-15)
        self.assertTrue(np.all(abs(.8*xx+ex.residual(xx)) <= abs(xx)+1e-12))

    def test_kernel_derivative_matches_finite_differences(self):
        xx = np.linspace(-.95, .95, 19)
        centers = np.array([-.8, 0., .625])
        h = 1e-6
        numerical = (core.kernel(xx+h, centers)-core.kernel(xx-h, centers))/(2*h)
        np.testing.assert_allclose(core.kernel_dx(xx, centers), numerical,
                                   atol=2e-9, rtol=2e-7)

    def test_exact_interpolation_origin_and_norms(self):
        for model, meta in self.selected.values():
            np.testing.assert_allclose(model.predict(model.centers),
                                       meta["training_labels"], atol=2e-11)
            self.assertEqual(float(model.predict([0.])[0]), 0.)
            self.assertLess(model.norm, ex.RKHS_BUDGET)
            self.assertLess(float(np.max(model.power(model.centers))), 2e-7)

    def test_first_sufficient_prefixes_are_regenerated(self):
        expected = {"fill_015": 33, "power_015": 9,
                    "fill_0075": 66, "power_0075": 17}
        full = self.selected["fill_0075"][0].centers
        self.assertEqual(len(self.history), 66)
        self.assertEqual(len(set(full)), len(full))
        self.assertEqual(full[0], 0.)
        np.testing.assert_allclose(full*1024, np.round(full*1024), atol=0.)
        for key, (model, meta) in self.selected.items():
            self.assertEqual(len(model.centers), expected[key])
            np.testing.assert_array_equal(model.centers, full[:expected[key]])
            self.assertLessEqual(meta["certified_error"], meta["target_error"])
            self.assertTrue(all(row[meta["certificate"]] > meta["target_error"]
                                for row in self.history[:expected[key]-1]))

    def test_greedy_geometry_does_not_use_training_labels(self):
        candidates = np.linspace(-1., 1., 2049)
        centers = [0.]
        for _ in range(8):
            model = core.Interpolant(np.array(centers), np.zeros(len(centers)))
            powers = model.power(candidates)
            for xi in centers:
                powers[np.argmin(abs(candidates-xi))] = -1
            centers.append(float(candidates[int(np.argmax(powers))]))
        np.testing.assert_array_equal(centers, self.selected["power_015"][0].centers)

    def test_uniform_certificate_covers_separate_diagnostic_grid(self):
        grid = np.linspace(-1., 1., 16001)
        for model, meta in self.selected.values():
            function = study.fill_certificate if meta["certificate"] == "fill" else study.power_certificate
            bound = function(model)
            self.assertAlmostEqual(bound, meta["certified_error"], places=12)
            self.assertLessEqual(float(np.max(abs(ex.residual(grid)-model.predict(grid)))), bound)

    def test_learned_contraction_uses_no_truth_slope_or_labels(self):
        grid = np.linspace(-1., 1., 16385)
        for model, meta in self.selected.values():
            with patch.object(ex, "residual", side_effect=AssertionError("truth leaked")), \
                 patch.object(ex, "observe", side_effect=AssertionError("labels leaked")), \
                 patch.object(ex, "RESIDUAL_LIP", 100.):
                bound = study.learned_contraction_certificate(model)
                ctrl = study.FixedRadiusController(meta["key"], model,
                                                   meta["certified_error"], meta["requested_radius"])
            self.assertEqual(ctrl.q, .7)
            self.assertLessEqual(float(np.max(abs(ex.A+ex.GAIN+model.derivative(grid)))),
                                 bound["bound"])
            self.assertLess(bound["bound"], .7)
            self.assertGreater(bound["covering_allowance"], 0.)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected, _ = selection()

    def test_constant_baseline_radius_and_reachable_interval(self):
        ctrl = study.controller("bounded", self.selected, 1.)
        self.assertAlmostEqual(ctrl.q, .5)
        self.assertAlmostEqual(ctrl.rho, .4)
        self.assertAlmostEqual(ctrl.vmax, .06)
        nominal = min(ctrl.zmax, ctrl.vmax/(ex.A-1)*(1-ex.A**(-ex.HORIZON)))
        expected = min(1., nominal+ctrl.rho)
        np.testing.assert_allclose(ctrl.actual_feasible_interval, [-expected, expected], atol=2e-12)

    def test_matched_certificates_keep_identical_requested_tubes(self):
        for radius in study.RADII:
            pair = [study.controller(study.key_for(kind, radius), self.selected, 16.)
                    for kind in study.KINDS]
            self.assertEqual(pair[0].rho, radius)
            self.assertEqual(pair[0].rho, pair[1].rho)
            self.assertEqual(pair[0].zmax, pair[1].zmax)
            self.assertEqual(pair[0].vmax, pair[1].vmax)

    def test_weighted_cost_gradients(self):
        for weight in (1., 16.):
            for key in ("bounded", "power_015", "power_0075"):
                ctrl = study.controller(key, self.selected, weight)
                y = ctrl.feasible_initial_plan(.5)
                directions = np.eye(len(y))*1e-6
                difference = np.array([(ctrl.cost(y+d)-ctrl.cost(y-d))/2e-6
                                       for d in directions])
                np.testing.assert_allclose(ctrl.gradient(y), difference,
                                           atol=2e-6, rtol=2e-5)

    def test_worst_case_additive_errors_fit_fixed_tube(self):
        for key in ("bounded", "power_015", "power_0075"):
            ctrl = study.controller(key, self.selected, 1.)
            for error in (-ctrl.rho, 0., ctrl.rho):
                for uncertainty in (-ctrl.bound["epsilon"], ctrl.bound["epsilon"]):
                    self.assertLessEqual(abs(ctrl.q*error+uncertainty), ctrl.rho+1e-12)

    def test_failed_or_infeasible_local_solution_retains_feasible_candidate(self):
        ctrl = study.controller("power_015", self.selected, 4.)
        warm = ctrl.feasible_initial_plan(.75)
        for success in (False, True):
            invalid = warm.copy()
            invalid[-1] = 100.
            with patch("engine.core.minimize", return_value=SimpleNamespace(success=success, x=invalid)):
                plan, fallback = ctrl.solve(.75, warm)
            self.assertTrue(fallback)
            np.testing.assert_array_equal(plan, warm)

    def test_prior_only_zero_nominal_plan_exposes_feedback_accounting(self):
        for weight in study.WEIGHTS:
            ctrl = study.sd.VariableTubeController(weight)
            for x in (-.5, -.25, .25, .5):
                plan = np.r_[np.zeros(ctrl.H+1), abs(x)*ctrl.q**np.arange(ctrl.H+1)]
                self.assertLess(ctrl.violation(plan, x), 2e-12)
                self.assertEqual(ctrl.cost(plan), 0.)
                self.assertGreater(weight*(ctrl.K*x)**2, 0.)

    def test_short_cohort_csv_dynamics_costs_tubes_and_input_decomposition(self):
        # Geometry has its own tests; reuse it here to keep the integration
        # check short while exercising the public runner and all three methods.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary, \
             patch.object(study, "select_models", return_value=selection()):
            result = study.run_study(temporary, weights=(16.,), starts=(.75,),
                                     keys=study.CONFERENCE_KEYS, steps=3, progress=None)
            self.assertEqual(result["total_recorded_steps"], 9)
            self.assertEqual(len(result["runs"]), 3)
            self.assertTrue((Path(temporary)/"summary.json").is_file())
            for run in result["runs"]:
                data = np.loadtxt(Path(temporary)/f"{run['run_id']}.csv", delimiter=",", skiprows=1)
                self.assertEqual(data.shape, (3, 17))
                successor = np.r_[data[1:, 1], run["final_state"]]
                np.testing.assert_allclose(successor, ex.A*data[:, 1]+data[:, 2]+ex.residual(data[:, 1]), atol=2e-12)
                np.testing.assert_allclose(data[:, 2], data[:, 4]+data[:, 16], atol=2e-12)
                np.testing.assert_allclose(data[:, 2]**2, data[:, 4]**2+data[:, 16]**2+2*data[:, 4]*data[:, 16], atol=2e-12)
                np.testing.assert_allclose(data[:, 12], np.cumsum(data[:, 1]**2+16*data[:, 2]**2), atol=2e-12)
                self.assertLessEqual(float(np.max(abs(data[:, 1]-data[:, 3])-data[:, 14])), 2e-7)
                self.assertLessEqual(float(np.max(data[:, 6]-data[:, 15])), 2e-7)
                self.assertLessEqual(float(np.max(abs(data[:, 2]))), .3+2e-7)


if __name__ == "__main__":
    unittest.main()
