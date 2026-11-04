"""Ablation stage for Drawn Apart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.plot_helpers import annotate_bar_values

from src.baseline import (
    _build_variant_context,
    _run_variant_result_row,
    check_baseline_dependencies,
)


VARIANTS = [
    {
        "key": "domain_balanced_sampling",
        "name": "Domain-balanced Sampling",
        "overrides": {
            "model_arch": "resnet34",
            "sampling_mode": "domain_balanced",
            "loss_mode": "cross_entropy",
            "unfreeze_strategy": "baseline_frozen",
        },
    },
    {
        "key": "class_imbalance_mitigation",
        "name": "Class-imbalance Mitigation",
        "overrides": {
            "model_arch": "resnet34",
            "sampling_mode": "shuffle",
            "loss_mode": "class_weighted_cross_entropy",
            "unfreeze_strategy": "baseline_frozen",
        },
    },
    {
        "key": "partial_unfreezing_fine_tune",
        "name": "Partial Unfreezing Fine-tune",
        "overrides": {
            "model_arch": "resnet34",
            "sampling_mode": "shuffle",
            "loss_mode": "cross_entropy",
            "unfreeze_strategy": "partial_unfreeze_layer4",
            "backbone_lr_scale": 0.1,
        },
    },
]



def _selected_variants(config: dict) -> list[dict]:
    """Select configured ablation variants while keeping stable order."""
    names = config.get("ablation_variant_names", [])
    if not names:
        return VARIANTS
    selected = {str(name).strip() for name in names}
    return [item for item in VARIANTS if item["name"] in selected]



def _epochs(config: dict, name: str) -> int:
    """Resolve epoch count for each ablation variant."""
    default_epochs = int(config.get("ablation_num_epochs", 1))
    by_name = config.get("ablation_epochs_by_name", {})
    if isinstance(by_name, dict) and name in by_name:
        return int(by_name[name])
    return default_epochs



def _plot_ablation_weighted_f1(project_root: Path, rows: list[dict]) -> str:
    """Plot ablation weighted F1 values in a dedicated figure."""

    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "ablation_weighted_f1.png"

    names = [row["name"] for row in rows]
    values = [float(row.get("weighted_f1", 0.0)) for row in rows]

    plt.figure(figsize=(7.2, 4.8))
    bar_patches = plt.bar(names, values, color="#1f77b4")
    plt.title("Ablation Weighted F1")
    plt.ylabel("Weighted F1")
    plt.xticks(rotation=16, ha="right")
    annotate_bar_values(bar_patches, values)
    plt.tight_layout()
    plt.savefig(output_file, dpi=165)
    plt.close()
    return str(output_file)



def _plot_ablation_delta(project_root: Path, rows: list[dict]) -> str:
    """Plot ablation delta versus baseline in a dedicated figure."""

    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "ablation_delta_vs_baseline.png"

    names = [row["name"] for row in rows]
    deltas = [float(row.get("delta_vs_anchor", 0.0)) for row in rows]
    colors = ["#1f77b4" if d >= 0 else "#d62728" for d in deltas]

    plt.figure(figsize=(7.2, 4.8))
    bar_patches = plt.bar(names, deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Ablation Delta vs ResNet34 Baseline")
    plt.ylabel("Delta Weighted F1")
    plt.xticks(rotation=16, ha="right")
    for bar_patch, delta in zip(bar_patches, deltas):
        y = delta + 0.002 if delta >= 0 else delta - 0.002
        plt.text(
            bar_patch.get_x() + bar_patch.get_width() / 2,
            y,
            f"{delta:+.4f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_file, dpi=165)
    plt.close()
    return str(output_file)



def write_results_summary(
    project_root: Path,
    baseline_f1: float,
    baseline_compare_rows: list[dict],
    ablation_rows: list[dict],
    seed: int,
) -> str:
    """Write one merged markdown summary for baseline comparison and ablation."""
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / "results_summary.md"

    lines = [
        "# Experiment Results Summary",
        "",
        "Metric: weighted F1 (single run)",
        "",
        "## Baseline Comparison",
        "",
        "| Method | Seed | Weighted F1 | Delta vs ResNet34 |",
        "|---|---:|---:|---:|",
        f"| ResNet34 Baseline | {seed} | {baseline_f1:.4f} | +0.0000 |",
    ]

    # Keep one merged table for consistent report references
    for row in baseline_compare_rows:
        lines.append(
            f"| {row['name']} | {row['seed']} | "
            f"{row['weighted_f1']:.4f} | {row['delta_vs_anchor']:+.4f} |"
        )

    lines.extend([
        "",
        "## Ablation",
        "",
        "| Method | Seed | Weighted F1 | Delta vs ResNet34 Baseline |",
        "|---|---:|---:|---:|",
    ])

    for row in ablation_rows:
        lines.append(
            f"| {row['name']} | {row['seed']} | "
            f"{row['weighted_f1']:.4f} | {row['delta_vs_anchor']:+.4f} |"
        )

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return str(output_file)



def run_experiment_stage(project_config: dict, data_state: dict, baseline_state: dict) -> dict:
    """Run ablation-only variants with checkpoint caching for fast reruns."""
    print("[START][ablation] Ablation stage started")

    deps_ok, missing = check_baseline_dependencies()
    if not deps_ok:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")

    variants = _selected_variants(project_config)
    context = _build_variant_context(
        project_config,
        baseline_state,
        scope="ablation",
        delta_label="delta_vs_anchor",
        epoch_resolver=_epochs,
    )
    rows = [
        _run_variant_result_row(project_config, data_state, variant, context)
        for variant in variants
    ]

    project_root = Path(project_config["project_root"])
    f1_plot = _plot_ablation_weighted_f1(project_root, rows)
    delta_plot = _plot_ablation_delta(project_root, rows)

    return {
        "status": "ready",
        "experiments": rows,
        "plots": {
            "ablation_weighted_f1": f1_plot,
            "ablation_delta": delta_plot,
        },
    }
