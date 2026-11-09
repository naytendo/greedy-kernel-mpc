# Kernel MPC numerical study

This standalone Python project reproduces the scalar kernel-learning and tube-MPC numerical study. Start by regenerating the figures and certificate analyses from the included reference data, then run a small closed-loop simulation before attempting a complete sweep. No LaTeX installation is required.

The project compares a prior-only controller with learned residual models selected to meet prescribed tube-error budgets. It also compares P-greedy, uniform, and random center placement. The saved results and their interpretation are documented in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Windows setup

Install Python 3.12, open this folder in VS Code, and choose **Terminal > New Terminal**. In the PowerShell terminal, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_study.py --mode reference
```

The first command creates a Python environment named `.venv` inside this folder. The remaining commands use its Python directly, so activating the environment is optional. In VS Code, **Python: Select Interpreter** lets you select `.venv\Scripts\python.exe` for editing and debugging.

If the `py` command is unavailable but `python --version` reports Python 3.12, use `python -m venv .venv` for the first command.

## macOS and Linux setup

With Python 3.12 installed, open a terminal in this folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_study.py --mode reference
```

Use `requirements.txt` for the pinned reproduction environment. The optional `requirements-compatible.txt` permits a broader set of package versions; results obtained with those versions need not match the reference as closely.

## Running the study

The examples below use Windows paths. On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

| Run | Command arguments after `run_study.py` | Purpose | Default output folder |
|---|---|---|---|
| Reference | `--mode reference` | Verify reference checksums, recompute the center-placement and derivative analyses, and redraw figures/tables; no new MPC simulations. | `outputs/reference/` |
| Smoke | `--mode smoke` | Simulate three controllers from `x0=0.75`, with input weight 16, for eight steps each. | `outputs/smoke/` |
| Conference | `--mode simulate --profile conference` | Simulate the prior-only baseline and the 9- and 17-center learned controllers: 90 runs. | `outputs/conference/` |
| Primary | `--mode simulate --profile primary` | Add the constant-bound baseline: 120 runs. | `outputs/primary/` |
| Extended | `--mode simulate --profile extended` | Include all six archived controller designs: 180 runs. | `outputs/extended/` |

For example:

```powershell
.\.venv\Scripts\python.exe run_study.py --mode smoke
.\.venv\Scripts\python.exe run_study.py --mode simulate --profile conference
```

All complete profiles use six initial states, five input weights, and 30 control steps per run. The smoke run is an installation and implementation check; its short trajectories are not the paper's numerical results. The engine prepares all four learned-model designs for its certificate analysis even when a profile simulates only a subset of them.

Use `--output "path/to/results"` to select a different output folder, and `--help` to inspect the command-line options. Generated results are separate from the included reference data. Full simulations repeatedly solve optimization problems; runtime depends on the machine and numerical libraries. No fixed runtime estimate is assumed.

The output folder contains `study-summary.pdf` and `.png`, `study-table.tex`, `metrics.csv`, `summary.json`, the center-selection and derivative-analysis JSON files, `derivative-table.tex`, `run-manifest.json`, and `verification.json`. Every mode writes verification checks; complete fresh runs also compare with the reference results. The included archive is under `data/reference/`; leave it unchanged when comparing a new run with the reference.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Reading the results

The five-panel summary connects learning to control:

1. **Center placement:** a uniform error certificate versus the number of centers. The random curve is the median of 200 draws; shading gives the 10th–90th percentiles of certificate values at each budget. It describes placement variability, not measurement noise or a confidence interval for the residual.
2. **Actual state:** a representative closed-loop trajectory at `x0=0.75`, input weight 16.
3. **Applied input:** the complete input, including the feedback correction, for the same run.
4. **Actual input effort:** the mean of `sum(u[k]**2)` over six starts, versus the input weight.
5. **Actual cost reduction:** reduction in mean actual cost relative to the prior-only controller at the same weight.

The evaluated cost is `sum(x[k]**2 + weight*u[k]**2)`. The optimizer instead penalizes nominal states and inputs. Since `u = v + K*(x-z)`, actual input energy includes the nominal–feedback cross term. Component energies cannot simply be added without that term.

In the saved study, the 9- and 17-center designs reduce mean actual cost by 34.05% and 42.82% at input weight 16. They also regulate more slowly: state-energy sums increase by 31.45% and 51.63%. These are tradeoffs for the chosen model, tube, objective, and baseline; they are not a general cost-improvement guarantee.

## What is assumed

- The plant is `x_next = 1.1*x + u - 0.2*sin(x)`, with `abs(x) <= 1` and `abs(u) <= 0.3`.
- Residual labels are exact. Selected states can be reached by an idealized reset, followed by a one-step experiment using `u=-0.3*x`. Safe travel to an arbitrary selected state is not modeled.
- There are 2,049 eligible candidate locations. A model with 9 or 17 centers uses 9 or 17 distinct labels, including the origin; the models share nested labels. Candidate locations and certificate-mesh points are not all measured transitions.
- The kernel is unit-variance Matérn 5/2 with length scale 0.5. The RKHS norm budget 0.5 is independently justified for this synthetic residual, rather than estimated from the fitted norm.
- The portable engine keeps gain `K=-0.6` and uses `q=0.7` as a design cap. Each accepted learned model must pass a fitted-derivative contraction check with a mesh correction; no numeric Lipschitz bound on the true residual is needed for that learned-model check. The nonlearning baselines receive a separate analytic residual-magnitude bound.
- The learned model and tube radius remain fixed online. SLSQP supplies local nonlinear solutions, checked against feasible shifted candidates. No global nonlinear optimum is claimed. Numerical tolerances and higher-precision cross-checks do not constitute formal interval-arithmetic certification.

Fresh numerical trajectories may differ slightly from the reference because of solver and library differences. Timing fields are machine dependent and are not expected to reproduce exactly.

## Sharing the project

Create a portable ZIP from this folder with:

```powershell
.\.venv\Scripts\python.exe make_archive.py
```

This creates `numerical-study-reproduction.zip` in the parent folder. It includes the source, documentation, dependencies, and reference data, while excluding generated outputs, virtual environments, and caches. Use `--output "path/to/package.zip"` to choose another archive location. On macOS or Linux, use `.venv/bin/python make_archive.py`.

Before public release, confirm the authorship and attribution of the code and data, choose their licenses, and archive a versioned release if a DOI is desired. No copyright license is assigned by this package, and the project has not been published automatically.
