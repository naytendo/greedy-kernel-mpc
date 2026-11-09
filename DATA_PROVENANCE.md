# Data provenance and numerical scope

## Reference experiment

The included reference trajectories come from the original `outputs/larger_residual_budget/results/certificate_weight_study` experiment in the research workspace. They are an archived numerical dataset, not measurements from a physical plant. This package includes copies so that redrawing the figures does not require access to the original workspace.

The included files live in `data/reference/`: `summary.json`, the 180 per-run CSV files, `center-selection-comparison.json`, `derivative-certification.json`, and `SHA256SUMS.json`. The hash manifest identifies the copied reference files. Generated files go to `outputs/` or the explicitly requested output directory.

The original summary has SHA-256:

```text
976e46f09d357fe3b05ead957da7c828aff79bb6104a15b08bb713f2e517d1fc
```

The archive contains 180 runs, each with 30 applied decisions, for 5,400 recorded steps. Its six controllers are:

| Archive key | Model and certificate | Prescribed tube radius | Centers |
|---|---|---:|---:|
| `bounded` | Zero residual predictor; constant magnitude bound | 0.4 | 0 |
| `state_dependent` | Zero residual predictor; state-dependent magnitude bound | Variable | 0 |
| `power_015` | Learned residual; mesh-corrected power certificate | 0.15 | 9 |
| `power_0075` | Learned residual; mesh-corrected power certificate | 0.075 | 17 |
| `fill_015` | Learned residual; explicit nearest-center certificate | 0.15 | 33 |
| `fill_0075` | Learned residual; explicit nearest-center certificate | 0.075 | 66 |

Each controller is run from `{-0.75,-0.5,-0.25,0.25,0.5,0.75}` at input weights `{1,2,4,8,16}`. The conference comparison selects `state_dependent`, `power_015`, and `power_0075`: 90 runs and 2,700 steps. The primary four-controller comparison adds `bounded`: 120 runs and 3,600 steps. The historical `fill` name identifies the implemented nearest-center bound; it does not mean that every possible Sobolev fill-distance bound was compared.

The source scripts were `certificate_weight_study.py`, `experiment.py`, `state_dependent_baseline.py`, and their inherited `greedy_tube_targets/example.py` controller implementation. The portable engine in this project brings those numerical ingredients into one standalone package. Reference mode checks the archive and regenerates the certificate analyses and plots without new closed-loop simulations; simulate mode computes new trajectories.

## Data acquisition and learning budget

The residual is input independent and is learned on `[-1,1]`. P-greedy selection uses 2,049 equally spaced eligible states and starts at the origin. At a selected state `xi`, the synthetic experiment resets the plant to `x=xi`, applies `u=-0.3*xi`, and forms the label `x_next-1.1*xi-u`.

The source selection routine caches each distinct queried label and reuses nested prefixes. A 9-center model uses 9 labels and a 17-center model uses 17; their union contains 17 distinct locations. The complete four-design archive extends the common prefix to 66 locations. These are per-model and shared-data counts, not 2,049 measured transitions. Preparing all four models can require the larger shared prefix even when a simulation profile plots only the two power-selected models.

Reset access is a sampling assumption. The one-step experiments are admissible for the synthetic plant, but the example does not design a trajectory that safely visits arbitrary selected states.

The unit-variance Matérn 5/2 kernel has length scale 0.5. The supplied RKHS norm budget is 0.5, supported by an independently constructed extension bound of approximately 0.443034 for the synthetic residual. A fitted interpolant norm is not itself an upper certificate for the unknown function's norm.

The uniform function-error certificate evaluates the power function on an independent 4,097-point mesh, then adds a covering correction. The covering radius is `1/4096`, the kernel-section Lipschitz constant is `sqrt(5/3)/0.5`, and the resulting norm-weighted correction is approximately `0.0003151841915858901`. Mesh evaluation uses kernel calculations, not new residual observations. The tiny origin offset removed from the numerical predictor is included in the function-error allowance.

The saved power certificates are approximately 0.0407830553 and 0.00840503253 for the 9- and 17-center models. These meet the respective benchmark budgets 0.045 and 0.0225. Their counts are first sufficient prefixes under the selected rule and certificate, not minimum possible sample requirements.

## Center-placement comparison

The included geometry comparison was generated separately from the controller trajectories. It uses the same kernel, candidate pool, norm budget, origin seed, and 4,097-point evaluation mesh. It requires no residual labels and runs no MPC controller for the alternative placements.

- P-greedy uses the saved nested sequence.
- Uniform designs are equally spaced, snapped to the candidate pool, and rebuilt at each tested odd budget. They are not generally nested.
- Random designs use 200 independent permutations of the non-origin candidates, sampled without replacement with seed `20261107`. Each permutation supplies nested prefixes.

The displayed random line is the pointwise median of the 200 certificates; its band spans the 10th–90th percentiles. It is not a probabilistic confidence band for the unknown residual. At the two target errors, first sufficient tested budgets are 9/17 for P-greedy and 9/13 for uniform placement; the random median curve reaches them at 27/37. Of the random sequences, 5% and 15.5% remain uncertified at 65 centers. P-greedy therefore does not uniformly dominate structured placement in this scalar example.

The center-selection helper records float64 Gram-conditioning, Cholesky, and computed power-variance diagnostics. These diagnostics do not constitute formal interval-arithmetic enclosures. The higher-precision derivative checks described below are a separate analysis.

## Historical contraction and the later derivative audit

The original source engine that generated the archive used gain `K=-0.6` and a supplied analytic residual-slope bound 0.2. For the learned benchmark controllers, this gave the direct plant–predictor comparison factor `abs(1.1+K)+0.2=0.7`. The constant-bound baseline instead had nominal factor 0.5. The variable-tube baseline used the additional magnitude bound `abs(r(x)) <= 0.2*abs(x)`.

The portable implementation retains `K=-0.6` and 0.7 as the prescribed contraction cap, but checks the fitted learned predictor directly. Its mesh-corrected derivative bound must establish `abs(0.5+rhat_prime(x)) <= 0.7` over the interval before accepting a learned model. That learned-model route uses the known fitted kernel function and does not assume the true residual has Lipschitz constant 0.2. The magnitude information for the nonlearning comparators remains separate. This additional acceptance check preserves the reference policies when the check passes; it does not imply a new optimal choice of gain or contraction cap.

The included saved-data derivative audit is a later analysis of the already selected interpolants. It uses their center/label pairs, the kernel, and the justified norm budget; it does not assume the numeric residual-slope bound or use the sine formula. Its learned-model contraction upper bounds are approximately 0.469228, 0.421392, 0.400846, and 0.394839 for 9, 17, 33, and 66 centers. These support the existing factor 0.7 by the learned-model route.

The alternative true-plant derivative-uncertainty route gives approximately 1.072298, 0.705465, 0.508767, and 0.430193. It therefore does not recertify the same factor 0.7 for the 9- or 17-center model. Failure of a sufficient test is not a claim of instability. This later audit neither reruns the controllers nor establishes new first-stopping counts for a joint derivative/function-error refinement rule.

The derivative audit uses analytical mesh corrections, float64 calculations, an independent finer mesh, and 60-digit numerical checks at reported maximizers. These remain numerical checks rather than formal interval-arithmetic enclosures.

## Reported costs and solver behavior

Per-run CSV files record actual state/input, the chosen nominal tube center/input, accepted plan cost, prediction deviation, timing, fallback indicator, shifted feasibility violation, true/model residuals, actual stage/accumulated costs, tube radii, and feedback input. There is one row per applied decision. The summary also records the final successor state.

The evaluation cost is `C_weight = sum(x**2 + weight*u**2)` and input energy is `E_u = sum(u**2)`. The nominal optimization objective uses `z` and `v`. Because `u=v+K*(x-z)`, actual energy equals nominal-input energy plus feedback-input energy plus their cross term. Cost-reduction percentages compare mean costs at the same weight, rather than averaging per-run percentage reductions.

At weight 16, the saved mean costs are 1.81984144 for the prior-only controller, 1.20011321 for 9 centers, and 1.04051227 for 17 centers. The reductions accompany slower regulation and depend on the model, tube, baseline, and nominal objective. They do not isolate learning as the sole cause or establish a universal improvement guarantee.

SLSQP computes local nonlinear plans. A returned solution is used only if it passes feasibility and shifted-candidate cost checks; otherwise the feasible candidate is retained. Learned-controller feasibility tolerance is `2e-8`, with accepted-cost slack `1e-10`; the variable-tube baseline uses `2e-7` and `1e-9`. An infeasible initial or shifted candidate stops the computation. The terminal equality/zero terminal cost and this accepted-suboptimal implementation do not, by themselves, verify the generic theorem's terminal-neighborhood and global-minimizer hypotheses.

The saved timings depend on the original machine. Fresh runs may have small numerical differences and different timings. No new trajectories are generated by reference mode, and no exact infinite-time guarantee follows solely from passing the finite-precision implementation checks.
