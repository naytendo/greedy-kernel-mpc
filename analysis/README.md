# Portable analysis helpers

Both helpers read summary.json from the supplied data directory, write their
results to the supplied output directory, and return the output dictionary.
They use no original-workspace paths, manuscript files, or controller imports.

    from analysis import run_center_comparison, run_derivative_audit

    centers = run_center_comparison(
        data_dir, output_dir,
        repetitions=200, seed=20261107, max_centers=65,
    )
    derivatives = run_derivative_audit(data_dir, output_dir)

Dependencies: NumPy, SciPy, and threadpoolctl. The derivative diagnostic's
60-digit checks use Python's standard-library decimal module.

The input summary must contain selected_designs with saved center_locations,
training_labels, centers, key, certificate, certified_error, requested_radius,
and model_norm fields. The derivative audit expects the four accepted models
at N=9,17,33,66. The center comparison uses the longest saved center prefix and
requires at least max_centers points.

Outputs:

- center-selection-comparison.json: origin-seeded greedy/uniform/random
  kernel-power comparison, pointwise random median and 10th/90th percentiles,
  threshold-crossing budgets with right-censored quantiles, and diagnostics.
- derivative-certification.json: continuous-domain mesh-corrected derivative
  allowances and learned/true contraction bounds, with high-precision audits.
- derivative-table.tex: compact table of the four derivative diagnostics.

For command-line use from the reproduction directory:

    python analysis/center_selection.py --data data/reference --output outputs/analysis
    python analysis/derivative_audit.py --data data/reference --output outputs/analysis

The default center comparison retains 200 repetitions, seed 20261107, and odd
budgets through 65. It compares fixed benchmark parameters: unit-variance
Matérn 5/2, length scale 0.5, RKHS norm budget 0.5, domain [-1,1], 2,049
candidate points, and a 4,097-point evaluation mesh. It uses only kernel
geometry, with no new closed-loop simulation or labels.

The derivative audit also uses the saved labels. It does not assume the
numerical true-residual Lipschitz bound 0.2. The RKHS norm budget and noiseless
interpolation assumptions remain necessary. It uses 4,097 corrected mesh
points, an 8,193-point check, and 60-digit calculations at maximizers.
These diagnostics evaluate exact-arithmetic inequalities numerically; they
are not formal interval-arithmetic enclosures.
