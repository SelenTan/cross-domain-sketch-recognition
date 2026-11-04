"""Evaluation stage for Drawn Apart."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import models, transforms
from src.baseline import _load_val_labels
from src.plot_helpers import annotate_bar_values

TOP1_METHODS = [
    ("ResNet34 Frozen baseline", "baseline"),
    ("MobileNetV3 Frozen baseline", "mobilenetv3_frozen_baseline"),
    ("Domain-balanced Sampling", "domain_balanced_sampling"),
    ("Class-imbalance Mitigation", "class_imbalance_mitigation"),
    ("Partial Unfreezing Fine-tune", "partial_unfreezing_fine_tune"),
]



def _check_dependencies() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for name in ("torch", "torchvision", "sklearn", "PIL", "matplotlib", "numpy"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return len(missing) == 0, missing



def _eval_transform():

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )



def _load_model(baseline_state: dict):
    """Load baseline checkpoint into ResNet34 with runtime-device fallback."""

    ckpt_path = Path(baseline_state.get("model_file", ""))
    if not ckpt_path.exists():
        raise RuntimeError("Baseline checkpoint missing; run baseline stage first")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt.get("class_names", [])
    if not class_names:
        raise RuntimeError("class_names missing in baseline checkpoint")

    device = str(baseline_state.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, device



def _extract_resnet34_feature(model, tensor):
    """Extract penultimate feature vector from ResNet34 for PCA plotting."""

    with torch.no_grad():
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x



def _predict_dir(model, device: str, image_dir: Path, class_names: list[str]) -> list[dict]:
    """Run directory inference and return per-image prediction rows."""

    tf = _eval_transform()
    rows: list[dict] = []
    with torch.no_grad():
        # Keep export order stable across repeated runs
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            image = Image.open(image_path).convert("RGB")
            logits = model(tf(image).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            rows.append(
                {
                    "filename": image_path.name,
                    "pred_idx": pred_idx,
                    "class_name": class_names[pred_idx],
                    "confidence": float(probs[0, pred_idx].item()),
                }
            )
    return rows



def _build_filename_label_index(
    labels_map: dict[str, str],
    class_to_idx: dict[str, int],
) -> dict[str, int]:
    """Map each filename to a numeric class index when label is valid."""
    filename_to_idx: dict[str, int] = {}
    for filename, label_name in labels_map.items():
        if label_name in class_to_idx:
            filename_to_idx[filename] = class_to_idx[label_name]
    return filename_to_idx



def _collect_sketch_val_features(
    model,
    device: str,
    sketch_val_dir: Path,
    filename_to_idx: dict[str, int],
    max_points: int = 1000,
):
    """Collect sketch_val features and labels for PCA scatter plotting."""
    tf = _eval_transform()
    features: list[list[float]] = []
    labels: list[int] = []

    for image_path in sorted(sketch_val_dir.iterdir()):
        if len(features) >= max_points:
            break
        if not image_path.is_file():
            continue
        if image_path.name not in filename_to_idx:
            continue

        image = Image.open(image_path).convert("RGB")
        tensor = tf(image).unsqueeze(0).to(device)
        feature = _extract_resnet34_feature(model, tensor).cpu().numpy().reshape(-1)
        features.append(feature.tolist())
        labels.append(int(filename_to_idx[image_path.name]))

    if not features:
        return None, None
    return np.array(features), np.array(labels)
def _plot_confusion_matrix(project_root: Path, cm) -> str:
    """Plot confusion matrix as a dedicated figure."""

    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_file = plot_dir / "confusion_matrix.png"

    plt.figure(figsize=(7.0, 6.2))
    plt.imshow(cm, cmap="Blues")
    plt.title("Sketch-Val Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=165)
    plt.close()
    return str(out_file)



def _plot_pca_scatter(project_root: Path, pca_points, pca_labels) -> str:
    """Plot PCA scatter as a dedicated figure."""

    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_file = plot_dir / "pca_scatter.png"

    plt.figure(figsize=(7.0, 6.2))
    if pca_points is not None and pca_labels is not None and len(pca_points) > 0:
        sc = plt.scatter(
            pca_points[:, 0],
            pca_points[:, 1],
            c=pca_labels,
            cmap="tab20",
            s=10,
            alpha=0.8,
        )
        plt.colorbar(sc, fraction=0.046, pad=0.04, label="class index")
        plt.title("Sketch-Val Feature PCA (ResNet34 Baseline Features)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    else:
        plt.title("Sketch-Val Feature PCA (ResNet34 Baseline Features, no data)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=165)
    plt.close()
    return str(out_file)



def _collect_true_and_pred(
    preds: list[dict],
    filename_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Collect aligned true and predicted indices from prediction rows."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for row in preds:
        filename = row["filename"]
        if filename in filename_to_idx:
            y_true.append(filename_to_idx[filename])
            y_pred.append(int(row["pred_idx"]))
    return y_true, y_pred




def _count_files_in_class_dir(class_dir: Path) -> int:
    """Count image files inside one class directory."""
    return sum(1 for file_path in class_dir.iterdir() if file_path.is_file())


def _build_source_class_counts(
    dataset_paths: dict[str, str],
    class_names: list[str],
) -> dict[str, int]:
    """Build per-class sample counts from cartoon and photograph training data."""
    counts = {name: 0 for name in class_names}
    for domain_key in ("cartoon", "photograph"):
        domain_dir = Path(dataset_paths[domain_key])
        if not domain_dir.exists():
            continue
        for class_dir in domain_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name in counts:
                counts[class_name] += _count_files_in_class_dir(class_dir)
    return counts


def _plot_data_imbalance_chart(
    project_root: Path,
    source_class_counts: dict[str, int],
) -> str:
    """Plot sorted long-tail class distribution for source training data."""
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_file = plot_dir / "data_imbalance_long_tail.png"

    values = sorted(source_class_counts.values(), reverse=True)
    class_indices = np.arange(len(values))
    mean_value = float(np.mean(values)) if values else 0.0
    max_value = int(max(values)) if values else 0
    min_value = int(min(values)) if values else 0
    max_idx = int(np.argmax(values)) if values else 0
    min_idx = int(np.argmin(values)) if values else 0

    plt.figure(figsize=(12.0, 2.5))
    plt.bar(class_indices, values, color="#1f77b4", width=0.9)
    plt.axhline(
        mean_value,
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label=f"Mean: {mean_value:.1f}",
    )

    if values:
        plt.annotate(
            f"Max: {max_value}",
            xy=(max_idx, max_value),
            xytext=(max_idx + 4, max_value * 0.95),
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=9,
        )
        plt.annotate(
            f"Min: {min_value}",
            xy=(min_idx, min_value),
            xytext=(max(min_idx - 12, 0), min_value + max_value * 0.12),
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=9,
        )
        plt.text(
            0.99,
            0.95,
            f"Max: {max_value}, Min: {min_value}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    plt.xlabel("Class Index")
    plt.ylabel("Number of Samples")
    plt.title("Long-tail Distribution of Source Training Data")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(out_file, dpi=165, bbox_inches="tight")
    plt.close()
    return str(out_file)

def _build_per_class_rows(class_names: list[str], class_f1, y_true: list[int]) -> list[dict]:
    """Build per-class f1 rows with support counts."""
    support_by_class: dict[int, int] = {}
    for label in y_true:
        support_by_class[label] = support_by_class.get(label, 0) + 1

    rows: list[dict] = []
    for idx, score in enumerate(class_f1):
        rows.append(
            {
                "class_idx": idx,
                "class_name": class_names[idx],
                "f1": float(score),
                "support": int(support_by_class.get(idx, 0)),
            }
        )
    return rows



def _summarize_sketch_val_metrics(
    class_names: list[str],
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """Compute confusion matrix and class-level summary metrics."""
    label_ids = list(range(len(class_names)))
    class_f1 = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    per_class_rows = _build_per_class_rows(class_names, class_f1, y_true)
    worst_classes = sorted(per_class_rows, key=lambda row: (row["f1"], row["support"]))[:5]
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": per_class_rows,
        "worst_classes": worst_classes,
    }



def _build_pca_points(
    model,
    device: str,
    sketch_val_dir: Path,
    filename_to_idx: dict[str, int],
):
    """Project extracted sketch_val features to 2D when enough samples exist."""
    features, labels = _collect_sketch_val_features(
        model,
        device,
        sketch_val_dir,
        filename_to_idx,
        max_points=1000,
    )
    if features is None or labels is None or len(features) < 2:
        return None, labels
    return PCA(n_components=2, random_state=42).fit_transform(features), labels


def _evaluate_sketch_val(
    data_state: dict,
    model,
    class_names: list[str],
    device: str,
    project_root: Path,
) -> dict:
    """Evaluate sketch_val and export confusion matrix plus PCA figures."""
    paths = data_state["dataset_paths"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels_map = _load_val_labels(Path(paths["val.csv"]))
    filename_to_idx = _build_filename_label_index(labels_map, class_to_idx)
    preds = _predict_dir(model, device, Path(paths["sketch_val"]), class_names)

    y_true, y_pred = _collect_true_and_pred(preds, filename_to_idx)
    if not y_true:
        raise RuntimeError(
            "No matched sketch_val labels were found. "
            "Please check that `val.csv` columns are `filename,class_name` "
            "and filenames exist under sketch_val."
        )

    metric_state = _summarize_sketch_val_metrics(class_names, y_true, y_pred)
    pca_points, pca_labels = _build_pca_points(
        model,
        device,
        Path(paths["sketch_val"]),
        filename_to_idx,
    )

    return {
        "weighted_f1": metric_state["weighted_f1"],
        "val_samples": len(y_true),
        "device": device,
        "confusion_matrix_plot": _plot_confusion_matrix(
            project_root,
            metric_state["confusion_matrix"],
        ),
        "pca_plot": _plot_pca_scatter(project_root, pca_points, pca_labels),
        "per_class_f1": metric_state["per_class_f1"],
        "worst_classes": metric_state["worst_classes"],
        "pred_rows": preds,
        "filename_to_idx": filename_to_idx,
        "class_names": class_names,
    }



def _select_confident_error_rows(
    pred_rows: list[dict],
    filename_to_idx: dict[str, int],
    class_names: list[str],
    sketch_val_dir: Path,
    top_k: int = 6,
) -> list[dict]:
    """Select highest-confidence misclassified sketch_val samples."""
    error_rows: list[dict] = []
    for row in pred_rows:
        filename = row["filename"]
        if filename not in filename_to_idx:
            continue
        true_idx = int(filename_to_idx[filename])
        pred_idx = int(row["pred_idx"])
        if true_idx == pred_idx:
            continue
        error_rows.append(
            {
                "image_path": str(sketch_val_dir / filename),
                "true_name": class_names[true_idx],
                "pred_name": class_names[pred_idx],
                "confidence": float(row["confidence"]),
            }
        )

    error_rows.sort(key=lambda item: item["confidence"], reverse=True)

    selected: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in error_rows:
        pair = (item["true_name"], item["pred_name"])
        if pair in seen_pairs:
            continue
        selected.append(item)
        seen_pairs.add(pair)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in error_rows:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected


def _plot_error_analysis_grid(project_root: Path, error_rows: list[dict]) -> str:
    """Plot confident misclassifications as a compact image grid."""
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_file = plot_dir / "error_analysis_grid.png"

    if not error_rows:
        plt.figure(figsize=(12.5, 2.9))
        plt.text(0.5, 0.5, "No misclassified sketch_val samples found", ha="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_file, dpi=165, bbox_inches="tight")
        plt.close()
        return str(out_file)

    n = min(len(error_rows), 5)
    nrows, ncols = 1, 5

    _, axes = plt.subplots(nrows, ncols, figsize=(19.5, 4.3))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for idx, axis in enumerate(axes_array):
        if idx >= n:
            axis.axis("off")
            continue

        row = error_rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"True: {row['true_name']}\n"
            f"Pred: {row['pred_name']} (Conf: {row['confidence']:.2f})",
            fontsize=13,
            pad=10,
        )
    plt.tight_layout()
    plt.savefig(out_file, dpi=165, bbox_inches="tight")
    plt.close()
    return str(out_file)

def _write_error_analysis_note(project_root: Path, val_state: dict) -> str:
    """Write per-class F1 worst-class analysis for report section 6.2."""
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_file = plot_dir / "error_analysis.md"

    rows = list(val_state.get("worst_classes", []))
    lines = [
        "# 6.2 Error Analysis",
        "",
        "Per-class F1 worst categories (from sketch_val, ResNet34 baseline checkpoint):",
        "",
        "| Rank | Class | Support | F1 |",
        "|---:|---|---:|---:|",
    ]

    for i, row in enumerate(rows, start=1):
        lines.append(f"| {i} | {row['class_name']} | {row['support']} | {row['f1']:.4f} |")

    if not rows:
        lines.append("| 1 | N/A | 0 | 0.0000 |")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return str(out_file)



def _write_sketch_test_submission(rows: list[dict], output_dir: Path) -> Path:
    """Write sketch test predictions csv in submission format."""
    submission_csv = output_dir / "sketch_test_predictions.csv"
    with submission_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["filename", "class_name"])
        for row in rows:
            writer.writerow([row["filename"], row["class_name"]])
    return submission_csv



def _count_predictions(rows: list[dict]) -> tuple[dict[str, int], list[float]]:
    """Count predicted labels and collect confidence scores."""
    counts: dict[str, int] = {}
    confidences: list[float] = []
    for row in rows:
        label = row["class_name"]
        counts[label] = counts.get(label, 0) + 1
        confidences.append(float(row["confidence"]))
    return counts, confidences



def _plot_sketch_test_outputs(
    plot_dir: Path,
    counts: dict[str, int],
    confidences: list[float],
) -> tuple[Path, Path]:
    """Plot sketch test class distribution and confidence histogram."""
    top20 = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    dist_plot = plot_dir / "sketch_test_distribution.png"
    conf_plot = plot_dir / "sketch_test_confidence_hist.png"

    plt.figure(figsize=(10.5, 4.8))
    if top20:
        plt.bar([name for name, _ in top20], [value for _, value in top20], color="#1f77b4")
        plt.xticks(rotation=70)
    plt.title("Sketch-Test Predicted Class Distribution (Top-20)")
    plt.tight_layout()
    plt.savefig(dist_plot, dpi=165)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    if confidences:
        plt.hist(confidences, bins=20, color="#1f77b4", edgecolor="white")
    plt.title("Sketch-Test Prediction Confidence Histogram")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.savefig(conf_plot, dpi=165)
    plt.close()
    return dist_plot, conf_plot



def _run_sketch_test(
    data_state: dict,
    model,
    class_names: list[str],
    device: str,
    project_root: Path,
) -> dict:
    """Run sketch_test inference and export separate distribution and confidence figures."""
    rows = _predict_dir(
        model,
        device,
        Path(data_state["dataset_paths"]["sketch_test"]),
        class_names,
    )

    output_dir = project_root / "outputs"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    submission_csv = _write_sketch_test_submission(rows, output_dir)
    counts, confidences = _count_predictions(rows)
    dist_plot, conf_plot = _plot_sketch_test_outputs(plot_dir, counts, confidences)

    return {
        "device": device,
        "prediction_count": len(rows),
        "submission_csv": str(submission_csv),
        "distribution_plot": str(dist_plot),
        "confidence_plot": str(conf_plot),
    }
def _build_model_for_arch(model_arch: str, class_count: int):
    """Build architecture-compatible classifier head for checkpoint loading."""

    arch = str(model_arch).strip().lower()
    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, class_count)
        return model
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feat, class_count)
        return model
    raise RuntimeError(f"Unsupported model_arch for top1 export: {model_arch}")



def _eval_top1_for_checkpoint(
    model,
    class_names: list[str],
    device: str,
    sketch_val_dir: Path,
    labels_map: dict[str, str],
) -> tuple[float, int]:
    """Compute top-1 on sketch_val using val.csv filename->class labels."""
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    preds = _predict_dir(model, device, sketch_val_dir, class_names)

    y_true: list[int] = []
    y_pred: list[int] = []
    for row in preds:
        truth = labels_map.get(row["filename"])
        if truth in class_to_idx:
            y_true.append(class_to_idx[truth])
            y_pred.append(int(row["pred_idx"]))

    if not y_true:
        raise RuntimeError("No matched sketch_val labels when computing top1 accuracy")
    top1 = float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true))
    return top1, len(y_true)



def _load_top1_method_assets(method_spec: tuple[str, str], model_dir: Path) -> dict:
    """Load persisted artifacts for one top1 method."""
    method_name, run_name = method_spec
    metrics_file = model_dir / f"{run_name}_metrics.json"
    model_file = model_dir / f"{run_name}_best.pt"
    if not metrics_file.exists() or not model_file.exists():
        raise RuntimeError(f"Missing model artifact for {run_name}")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    model_arch = str(metrics.get("model_arch", "resnet34"))
    ckpt = torch.load(model_file, map_location="cpu")
    class_names = ckpt.get("class_names", [])
    if not class_names:
        raise RuntimeError(f"class_names missing in checkpoint: {run_name}")
    return {
        "method_name": method_name,
        "run_name": run_name,
        "metrics": metrics,
        "metrics_file": metrics_file,
        "class_names": class_names,
        "model_arch": model_arch,
        "state_dict": ckpt["model_state_dict"],
    }



def _evaluate_top1_method(
    method_spec: tuple[str, str],
    model_dir: Path,
    sketch_val_dir: Path,
    labels_map: dict[str, str],
    device: str,
) -> dict:
    """Evaluate one stored checkpoint on sketch_val and update its metrics file."""
    assets = _load_top1_method_assets(method_spec, model_dir)
    model = _build_model_for_arch(assets["model_arch"], len(assets["class_names"]))
    model.load_state_dict(assets["state_dict"])
    model = model.to(device)
    model.eval()

    top1, samples = _eval_top1_for_checkpoint(
        model,
        assets["class_names"],
        device,
        sketch_val_dir,
        labels_map,
    )
    assets["metrics"]["top1_accuracy"] = top1
    assets["metrics_file"].write_text(json.dumps(assets["metrics"], indent=2), encoding="utf-8")
    return {
        "method": assets["method_name"],
        "run_name": assets["run_name"],
        "top1_accuracy": top1,
        "val_samples": samples,
    }


def _plot_top1_accuracy_summary(plot_dir: Path, rows: list[dict]) -> str:
    """Plot top1 accuracy summary and return plot path."""
    plot_file = plot_dir / "top1_accuracy_summary.png"
    names = [row["method"] for row in rows]
    values = [float(row["top1_accuracy"]) for row in rows]

    plt.figure(figsize=(10.0, 5.2))
    bar_patches = plt.bar(names, values, color="#1f77b4")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Accuracy on sketch_val")
    plt.xticks(rotation=18, ha="right")
    annotate_bar_values(bar_patches, values)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=165)
    plt.close()
    return str(plot_file)



def _export_top1_accuracy_summary(
    project_root: Path,
    data_state: dict,
    preferred_device: str,
) -> dict:
    """Evaluate five configured methods on sketch_val and export one summary plot."""
    device = preferred_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_dir = project_root / "outputs" / "models"
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels_map = _load_val_labels(Path(data_state["dataset_paths"]["val.csv"]))
    sketch_val_dir = Path(data_state["dataset_paths"]["sketch_val"])
    rows = [
        _evaluate_top1_method(method, model_dir, sketch_val_dir, labels_map, device)
        for method in TOP1_METHODS
    ]
    return {"rows": rows, "plot": _plot_top1_accuracy_summary(plot_dir, rows)}


def _prepare_evaluation_outputs(
    project_root: Path,
    data_state: dict,
    baseline_state: dict,
) -> dict:
    """Run all evaluation substeps and return consolidated outputs."""
    model, class_names, device = _load_model(baseline_state)

    val_state = _evaluate_sketch_val(data_state, model, class_names, device, project_root)
    print("[PROGRESS][evaluation] sketch_val complete")

    source_counts = _build_source_class_counts(data_state["dataset_paths"], class_names)
    imbalance_plot = _plot_data_imbalance_chart(project_root, source_counts)

    confident_errors = _select_confident_error_rows(
        val_state["pred_rows"],
        val_state["filename_to_idx"],
        class_names,
        Path(data_state["dataset_paths"]["sketch_val"]),
        top_k=5,
    )
    error_grid_plot = _plot_error_analysis_grid(project_root, confident_errors)

    test_state = _run_sketch_test(data_state, model, class_names, device, project_root)
    print("[PROGRESS][evaluation] sketch_test complete")

    sketch_val_output = dict(val_state)
    sketch_val_output.pop("pred_rows", None)
    sketch_val_output.pop("filename_to_idx", None)
    sketch_val_output.pop("class_names", None)

    return {
        "sketch_val": sketch_val_output,
        "sketch_test": test_state,
        "error_analysis_note": _write_error_analysis_note(project_root, val_state),
        "top1_summary": _export_top1_accuracy_summary(project_root, data_state, device),
        "data_imbalance_plot": imbalance_plot,
        "confident_error_grid": error_grid_plot,
        "confident_error_examples": confident_errors,
    }


def run_evaluation_stage(project_config: dict, data_state: dict, baseline_state: dict) -> dict:
    """Run evaluation stage and export report figures with separated responsibilities."""
    print("[START][evaluation] Evaluation stage started")

    deps_ok, missing = _check_dependencies()
    if not deps_ok:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")

    outputs = _prepare_evaluation_outputs(
        Path(project_config["project_root"]),
        data_state,
        baseline_state,
    )
    sketch_val = outputs["sketch_val"]
    sketch_test = outputs["sketch_test"]

    print(
        f"[RESULT][evaluation] sketch_val_weighted_f1={sketch_val['weighted_f1']:.4f}, "
        f"val_samples={sketch_val['val_samples']}, "
        f"sketch_test_predictions={sketch_test['prediction_count']}"
    )
    return {"status": "ready", **outputs}
