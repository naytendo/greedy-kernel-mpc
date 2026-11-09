# Validation record

Validated on 2026-11-09 with Python 3.12.4 and the versions pinned in
requirements.txt on Windows. All commands were run from this standalone
folder without importing modules from the original experiment directories.

- All 22 unit/integration checks passed. These include fresh short closed-loop
  runs, retained-candidate fallback, exact interpolation, certificate targets,
  constraint and tube checks, and actual input/cost accounting.
- Reference mode verified 183 data-file checksums and regenerated the
  200-permutation center comparison, derivative audits, tables and figure.
- Extended mode independently regenerated the greedy centers, acquired the
  simulated labels, and ran all 180 trajectories (5,400 steps). It compared
  their actual costs with the supplied archive using relative tolerance
  1e-4 and absolute tolerance 5e-7. All comparisons passed, with zero
  observed actual-cost difference in this tested environment.
- The ZIP was extracted into a separate folder and its command-line smoke
  mode ran successfully from a different working directory. It regenerated
  the models, simulated three controllers for eight steps each, recomputed
  both analyses, and rendered its outputs.
- Selected center counts remained 9 and 17 for the power certificate and
  33 and 66 for the nearest-center comparison. The four fitted-model
  contraction checks passed the fixed cap 0.7.

Each completed run writes its own verification.json and run-manifest.json.
The latter records dependency versions, parameters and elapsed time. Solver
timings and PDF metadata need not be byte-identical between machines. These
are computational checks of the stated experiment, not global-optimality
proofs or formal interval-arithmetic certificates.
