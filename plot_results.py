"""Render the five-panel paper figure and tables without manuscript dependencies."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "state_dependent": ("#398347", "--", "Prior-only"),
    "power_015": ("#2580bf", "-", "N=9"),
    "power_0075": ("#cf688b", "-.", "N=17"),
}


def render_results(data_dir, output_dir):
    source, out = Path(data_dir), Path(output_dir)
    study = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((out / "center-selection-comparison.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.1, 3.65), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, i:i + 2]) for i in (0, 2, 4)]
    axes.extend([fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])])
    ns = comparison["protocol"]["comparison_budgets"]
    axes[0].fill_between(ns, comparison["random_q10"], comparison["random_q90"],
                         color="#777777", alpha=.18, linewidth=0)
    axes[0].semilogy(ns, comparison["random_median"], color="#777777",
                     linestyle="--", label="Random median")
    axes[0].semilogy(ns, comparison["uniform_certificates"], color="#56559a",
                     linestyle="-.", label="Uniform")
    axes[0].semilogy(ns, comparison["greedy_certificates"], color="#b96822", label="P-greedy")
    designs = {d["key"]: d for d in study["selected_designs"] if d["certificate"] == "power"}
    correction = .5 * study["feature_lipschitz_constant"] * study["power_mesh_covering_radius"]
    axes[0].axhline(correction, color="#777777", linestyle="--", linewidth=.75)
    trajectory_steps = []
    for key, (color, line, label) in STYLES.items():
        if key in designs:
            d = designs[key]
            axes[0].axhline(d["target_error"], color=color, linestyle=":", linewidth=.8)
            axes[0].scatter(d["centers"], d["certified_error"], s=14, color=color, zorder=5)
            axes[0].annotate(f"N={d['centers']}", (d["centers"], d["certified_error"]),
                             xytext=(3, -12), textcoords="offset points", fontsize=6.5)
        run = next(r for r in study["runs"] if r["controller"] == key
                   and r["input_weight"] == 16 and r["initial_state"] == .75)
        rows = np.loadtxt(source / (run["run_id"] + ".csv"), delimiter=",", skiprows=1,
                          ndmin=2)[:13]
        trajectory_steps.append(len(rows))
        for ax, col in zip(axes[1:3], (1, 2)):
            ax.plot(rows[:, 0], rows[:, col], color=color, linestyle=line, label=label)
    axes[0].set(title="(a) Center placement", xlabel="Centers N",
                ylabel=r"$\bar\varepsilon_N^{\rm mesh}$", ylim=(2e-4, .8))
    axes[0].legend(loc="upper right", fontsize=6.2, handlelength=1.7,
                   borderpad=.25, labelspacing=.25)
    axes[1].set(title="(b) Actual state", xlabel="Control step k", ylabel="x", ylim=(-.01, .8))
    axes[2].set(title="(c) Applied input", xlabel="Control step k", ylabel="u", ylim=(-.32, .01))
    axes[2].axhline(-.3, color="#777777", linestyle=":", linewidth=.75)
    axes[1].legend(loc="upper right", handlelength=1.8, borderpad=.3)
    aggregate = {(r["input_weight"], r["controller"]): r for r in study["aggregate"]}
    weights = sorted({w for w, key in aggregate if key in STYLES})
    for key, (color, line, label) in STYLES.items():
        rows = [aggregate[w, key] for w in weights]
        reductions = [100 * (1 - r["mean_actual_cost"]
                            / aggregate[w, "state_dependent"]["mean_actual_cost"])
                      for w, r in zip(weights, rows)]
        axes[3].plot(weights, [r["mean_input_energy"] for r in rows], color=color,
                     linestyle=line, marker="o", markersize=3)
        axes[4].plot(weights, reductions, color=color, linestyle=line, marker="o", markersize=3)
    axes[3].set(title="(d) Actual input effort", ylabel=r"Mean $E_u$", ylim=(0, .105))
    axes[4].set(title="(e) Actual cost reduction", ylabel="Reduction vs. prior-only (%)", ylim=(-2, 48))
    for ax in axes[3:]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(weights, labels=[f"{w:g}" for w in weights])
        ax.set_xlabel(r"Input weight $\omega$")
    for ax in axes:
        ax.grid(alpha=.15)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"study-summary.{suffix}", bbox_inches="tight", pad_inches=.02, dpi=260)
    plt.close(fig)

    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Design & $\rho_{\rm tar}$ & $N$ & $\bar\varepsilon_N^{\rm mesh}$ & $\bar v$ & Mean $C_{16}$ & Mean $E_u$\\",
             r"\hline"]
    for key in STYLES:
        row = aggregate[16, key]
        if key in designs:
            d = designs[key]
            fields = ("Learned", f"{d['requested_radius']:.3f}", str(d["centers"]),
                      f"{d['certified_error']:.5f}", f"{d['nominal_input_limit']:.3f}")
        else:
            fields = ("Prior-only", r"\text{variable}", "0", "--", r"\text{variable}")
        lines.append(" & ".join((*fields, f"{row['mean_actual_cost']:.5f}",
                                 f"{row['mean_input_energy']:.5f}")) + r"\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    (out / "study-table.tex").write_text("\n".join(lines), encoding="utf-8")
    rows = [r for r in study["aggregate"] if r["controller"] in STYLES]
    fields = ["input_weight", "controller", "cohort_count", "mean_actual_cost",
              "mean_state_energy", "mean_input_energy", "actual_cost_gain_vs_prior_percent",
              "input_energy_gain_vs_prior_percent", "total_fallbacks"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"trajectory_steps_shown": min(trajectory_steps), "input_weights": weights,
            "controllers": list(STYLES), "steps_per_run": study["steps_per_run"]}
