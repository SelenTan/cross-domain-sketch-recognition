"""Main workflow entry for cross-domain sketch recognition."""

from pathlib import Path

from src.baseline import (
    resolve_runtime_device,
    run_baseline_compare_stage,
    run_baseline_stage,
)
from src.data_pipeline import run_data_pipeline_stage
from src.evaluation import run_evaluation_stage
from src.experiments import run_experiment_stage, write_results_summary


PROJECT_CONFIG = {
    # Kaggle auth should come from env vars or ~/.kaggle/access_token
    "baseline_seed": 42,
    "baseline_batch_size": 128,
    "baseline_num_workers": 0,
    "baseline_learning_rate": 0.001,
    "baseline_num_epochs": 3,
    "baseline_pretrained": True,
    "runtime_device": "auto",
    "baseline_compare_num_epochs": 2,
    "baseline_compare_epochs_by_name": {
        "MobileNetV3 Frozen Baseline": 2,
    },
    "baseline_compare_variant_names": [
        "MobileNetV3 Frozen Baseline",
    ],
    "ablation_num_epochs": 1,
    "ablation_epochs_by_name": {
        "Domain-balanced Sampling": 1,
        "Class-imbalance Mitigation": 1,
        "Partial Unfreezing Fine-tune": 2,
    },
    "ablation_variant_names": [
        "Domain-balanced Sampling",
        "Class-imbalance Mitigation",
        "Partial Unfreezing Fine-tune",
    ],
}


FALLBACK_MESSAGES = {
    "deps_missing": "dependencies are missing; install required packages and rerun",
    "data_access": "dataset access failed; check Kaggle auth/CLI and rerun",
    "data_integrity": "dataset files are incomplete or invalid; fix dataset structure and rerun",
    "artifacts_missing": (
        "required model/evaluation artifacts are missing; "
        "run prerequisite stages first"
    ),
    "config_invalid": "runtime configuration is invalid for current pipeline settings",
}


def _classify_fallback(error: RuntimeError) -> str:
    """Map runtime errors to one of five pipeline error categories."""
    message = str(error)
    if "Missing dependencies:" in message:
        return "deps_missing"
    if any(
        token in message
        for token in (
            "Kaggle auth token required",
            "Kaggle CLI not found",
            "Kaggle download failed",
        )
    ):
        return "data_access"
    if any(
        token in message
        for token in (
            "no zip file was found",
            "expected folder structure",
            "Dataset incomplete:",
        )
    ):
        return "data_integrity"
    if any(
        token in message
        for token in (
            "Baseline checkpoint missing",
            "class_names missing",
            "Missing model artifact",
            "No matched sketch_val labels",
            "sketch_val evaluation produced zero matched samples",
        )
    ):
        return "artifacts_missing"
    return "config_invalid"


def main() -> None:
    """Run the full non-interactive workflow."""
    project_root = Path(__file__).resolve().parent
    workflow_config = dict(PROJECT_CONFIG)
    workflow_config["project_root"] = str(project_root)

    resolved_device, resolved_gpu_name = resolve_runtime_device(workflow_config)
    workflow_config["resolved_device"] = resolved_device
    workflow_config["resolved_gpu_name"] = resolved_gpu_name

    try:
        data_state = run_data_pipeline_stage(project_root)

        baseline_state = run_baseline_stage(workflow_config, data_state)
        baseline_compare_state = run_baseline_compare_stage(
            workflow_config,
            data_state,
            baseline_state,
        )
        ablation_state = run_experiment_stage(workflow_config, data_state, baseline_state)
        run_evaluation_stage(workflow_config, data_state, baseline_state)

        summary_file = write_results_summary(
            project_root=project_root,
            baseline_f1=float(baseline_state["weighted_f1"]),
            baseline_compare_rows=list(baseline_compare_state.get("comparisons", [])),
            ablation_rows=list(ablation_state.get("experiments", [])),
            seed=int(workflow_config["baseline_seed"]),
        )
        print(f"[RESULT][main] Workflow finished, summary={summary_file}")
    except RuntimeError as error:
        code = _classify_fallback(error)
        print(f"[PROGRESS][fallback] detail={error}")
        print(f"[RESULT][fallback] code={code}, message={FALLBACK_MESSAGES[code]}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
