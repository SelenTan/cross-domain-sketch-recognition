"""Baseline training and baseline comparison stages for Drawn Apart."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights, ResNet34_Weights
from tqdm.auto import tqdm



BASELINE_COMPARE_VARIANTS = [
    {
        "key": "mobilenetv3_frozen_baseline",
        "name": "MobileNetV3 Frozen Baseline",
        "overrides": {
            "model_arch": "mobilenet_v3_small",
            "sampling_mode": "shuffle",
            "loss_mode": "cross_entropy",
            "unfreeze_strategy": "baseline_frozen",
        },
    }
]



def check_baseline_dependencies() -> tuple[bool, list[str]]:
    """Check dependencies used by training and plotting."""
    missing: list[str] = []
    for module_name in ("torch", "torchvision", "sklearn", "PIL", "matplotlib", "tqdm", "numpy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return len(missing) == 0, missing



def resolve_runtime_device(config: dict) -> tuple[str, str]:
    """Resolve runtime device once with automatic CPU fallback."""

    mode = str(config.get("runtime_device", "auto")).strip().lower()
    if mode == "cpu":
        return "cpu", "N/A"
    if mode == "cuda" and not torch.cuda.is_available():
        return "cpu", "N/A"
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "N/A"



def _set_seed(seed: int) -> None:
    """Set random seeds so repeated runs are controlled and comparable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _load_val_labels(val_csv: Path) -> dict[str, str]:
    """Load sketch_val labels from val.csv."""
    mapping: dict[str, str] = {}
    # Handle csv files that include a BOM mark
    with val_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            normalized = {str(k).strip().replace("\ufeff", ""): v for k, v in row.items()}
            fn = (normalized.get("filename") or "").strip()
            cls = (normalized.get("class_name") or "").strip()
            if fn and cls:
                mapping[fn] = cls
    return mapping


class _TransformDataset(Dataset):
    """Apply one transform to each sample in the merged dataset."""

    def __init__(self, dataset, tf):
        self.dataset = dataset
        self.tf = tf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.tf(image), label



def _build_train_loader(dataset_paths: dict[str, str], config: dict):
    """Build cartoon+photograph train loader and class statistics."""
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    cartoon = ImageFolder(root=dataset_paths["cartoon"], transform=None)
    photo = ImageFolder(root=dataset_paths["photograph"], transform=None)
    if cartoon.classes != photo.classes:
        raise RuntimeError("Class folders mismatch between cartoon and photograph")

    train_ds, class_counts, sample_domains = _merge_domain_datasets(cartoon, photo, train_tf)
    batch = int(config.get("baseline_batch_size", 128))
    workers = int(config.get("baseline_num_workers", 0))
    sampling_mode = str(config.get("sampling_mode", "shuffle")).strip().lower()
    loader = _build_train_data_loader(train_ds, sample_domains, sampling_mode, batch, workers)
    return loader, class_counts, sorted(cartoon.classes), sampling_mode



def _merge_domain_datasets(cartoon, photo, train_tf):
    """Merge two source domains and collect shared sample statistics."""
    merged = ConcatDataset([cartoon, photo])
    train_ds = _TransformDataset(merged, train_tf)

    class_counts = [0 for _ in cartoon.classes]
    sample_domains: list[str] = []

    for _, label in cartoon.samples:
        class_counts[label] += 1
        sample_domains.append("cartoon")

    for _, label in photo.samples:
        class_counts[label] += 1
        sample_domains.append("photograph")

    return train_ds, class_counts, sample_domains



def _build_train_data_loader(train_ds, sample_domains, sampling_mode, batch, workers):
    """Build train loader using either shuffled or domain-balanced sampling."""
    if sampling_mode == "shuffle":
        return DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)

    if sampling_mode == "domain_balanced":
        domain_counts = {
            "cartoon": max(sample_domains.count("cartoon"), 1),
            "photograph": max(sample_domains.count("photograph"), 1),
        }
        c_weight = 1.0 / domain_counts["cartoon"]
        p_weight = 1.0 / domain_counts["photograph"]
        weights = [c_weight if d == "cartoon" else p_weight for d in sample_domains]
        sampler = WeightedRandomSampler(weights, len(weights), True)
        return DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers)

    raise RuntimeError(f"Unsupported sampling_mode: {sampling_mode}")


def _build_sketch_val_loader(
    dataset_paths: dict[str, str],
    class_to_idx: dict[str, int],
    config: dict,
):
    """Build sketch_val loader from folder + val.csv labels."""

    class SketchValDataset(Dataset):
        """Load sketch val files that can be matched with labels from val csv."""

        def __init__(self, root: Path, mapping: dict[str, str], index: dict[str, int], tf):
            self.tf = tf
            self.samples: list[tuple[Path, int]] = []
            # Evaluate only files that have labels in val csv
            for path in sorted(root.iterdir()):
                if not path.is_file():
                    continue
                cls = mapping.get(path.name)
                if cls in index:
                    self.samples.append((path, index[cls]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert("RGB")
            return self.tf(image), label

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    mapping = _load_val_labels(Path(dataset_paths["val.csv"]))
    ds = SketchValDataset(Path(dataset_paths["sketch_val"]), mapping, class_to_idx, eval_tf)
    batch = int(config.get("baseline_batch_size", 128))
    workers = int(config.get("baseline_num_workers", 0))
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)



def _build_model(config: dict, class_count: int):
    """Build model for selected architecture and return (model, arch_name)."""

    arch = str(config.get("model_arch", "resnet34")).strip().lower()
    pretrained = bool(config.get("baseline_pretrained", True))

    if arch == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, class_count)
        return model, arch

    if arch == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feat, class_count)
        return model, arch

    raise RuntimeError(f"Unsupported model_arch: {arch}")



def _set_trainable_partial_unfreeze_layer4(model, arch: str) -> str:
    """Unfreeze layer4 and head for partial fine-tuning."""
    if arch != "resnet34":
        raise RuntimeError("partial_unfreeze_layer4 is only supported for resnet34")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return "partial_unfreeze_layer4"



def _set_trainable_baseline_frozen(model, arch: str) -> str:
    """Freeze backbone and train only the classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    if arch == "resnet34":
        head_params = model.fc.parameters()
    else:
        head_params = model.classifier[-1].parameters()

    for param in head_params:
        param.requires_grad = True

    return "baseline_frozen"



def _set_trainable_full_finetune(model) -> str:
    """Train all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return "full_finetune"



def _apply_trainable_strategy(model, config: dict, arch: str) -> str:
    """Apply trainable-parameter strategy and return resolved mode name."""
    strategy = str(config.get("unfreeze_strategy", "baseline_frozen")).strip().lower()
    strategy_handlers = {
        "partial_unfreeze_layer4": lambda: _set_trainable_partial_unfreeze_layer4(model, arch),
        "baseline_frozen": lambda: _set_trainable_baseline_frozen(model, arch),
        "full_finetune": lambda: _set_trainable_full_finetune(model),
    }

    if strategy not in strategy_handlers:
        raise RuntimeError(f"Unsupported unfreeze_strategy: {strategy}")

    return strategy_handlers[strategy]()


def _build_optimizer(model, config: dict, arch: str, strategy: str):
    """Build optimizer and return optimizer with mode label."""

    lr = float(config.get("baseline_learning_rate", 0.001))
    if strategy == "partial_unfreeze_layer4" and arch == "resnet34":
        scale = float(config.get("backbone_lr_scale", 0.1))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        p for p in model.layer4.parameters() if p.requires_grad
                    ],
                    "lr": lr * scale,
                },
                {"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": lr},
            ]
        )
        return optimizer, "differential_lr"

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    return optimizer, "single_lr"



def _build_loss(config: dict, class_counts: list[int], device: str):
    """Build requested loss and return (criterion, mode_name)."""

    loss_mode = str(config.get("loss_mode", "cross_entropy")).strip().lower()
    if loss_mode == "class_weighted_cross_entropy":
        total = float(sum(class_counts))
        cnum = len(class_counts)
        weights = [0.0 if c <= 0 else total / (cnum * float(c)) for c in class_counts]
        tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=tensor), loss_mode
    if loss_mode == "cross_entropy":
        return nn.CrossEntropyLoss(), loss_mode
    raise RuntimeError(f"Unsupported loss_mode: {loss_mode}")



def _run_signature(config: dict) -> str:
    """Build deterministic run signature for cache reuse."""
    payload = {
        "seed": int(config.get("baseline_seed", 42)),
        "model_arch": str(config.get("model_arch", "resnet34")),
        "batch": int(config.get("baseline_batch_size", 128)),
        "lr": float(config.get("baseline_learning_rate", 0.001)),
        "epochs": int(config.get("baseline_num_epochs", 4)),
        "pretrained": bool(config.get("baseline_pretrained", True)),
        "sampling_mode": str(config.get("sampling_mode", "shuffle")),
        "loss_mode": str(config.get("loss_mode", "cross_entropy")),
        "unfreeze_strategy": str(config.get("unfreeze_strategy", "baseline_frozen")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()



def _load_cached_result(
    project_root: Path,
    run_name: str,
    signature: str,
    require_model: bool,
) -> dict | None:
    """Load cached metrics/checkpoint when signature matches."""
    model_file = project_root / "outputs" / "models" / f"{run_name}_best.pt"
    metrics_file = project_root / "outputs" / "models" / f"{run_name}_metrics.json"
    if not metrics_file.exists():
        return None
    if require_model and not model_file.exists():
        return None

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    # Reject cache entries when the run signature does not match
    if metrics.get("run_signature") != signature:
        return None

    return {
        "status": "ready",
        "run_name": run_name,
        "run_title": metrics.get("run_title", run_name),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
        "top1_accuracy": float(metrics.get("top1_accuracy", 0.0) or 0.0),
        "val_samples": int(metrics.get("val_samples", 0) or 0),
        "device": metrics.get("device", "cpu"),
        "gpu_name": metrics.get("gpu_name", "N/A"),
        "model_file": str(model_file) if model_file.exists() else "",
        "metrics_file": str(metrics_file),
        "val_loss_history": metrics.get("val_loss_history", []),
        "cached": True,
    }



def _prepare_run_settings(config: dict) -> dict:
    """Resolve stable run settings used by training and cache lookup."""
    run_name = str(config.get("run_name", "baseline")).strip().lower().replace(" ", "_")
    return {
        "run_name": run_name,
        "run_title": str(config.get("run_title", run_name)).strip(),
        "log_scope": str(config.get("log_scope", "baseline")).strip().lower(),
        "seed": int(config.get("baseline_seed", 42)),
        "epochs": int(config.get("baseline_num_epochs", 4)),
        "save_checkpoint": bool(config.get("save_checkpoint", True)),
        "allow_cache": bool(config.get("allow_cache", True)),
        "log_device": bool(config.get("log_device", False)),
    }



def _resolve_cached_run(project_root: Path, config: dict, settings: dict) -> dict | None:
    """Return cached run payload when signature and artifacts are valid."""
    if not settings["allow_cache"]:
        return None
    signature = _run_signature(config)
    cached = _load_cached_result(
        project_root,
        settings["run_name"],
        signature,
        require_model=settings["save_checkpoint"],
    )
    if cached is not None:
        print(f"[RESULT][{settings['log_scope']}] run={settings['run_title']} reused cached model")
    return cached



def _prepare_training_artifacts(project_root: Path) -> None:
    """Prepare runtime cache paths and environment for torch artifacts."""
    torch_cache = project_root / "outputs" / "torch_cache"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)



def _run_train_epoch(loop_state: dict, desc: str) -> None:
    """Run one training epoch."""
    model = loop_state["model"]
    train_loader = loop_state["train_loader"]
    optimizer = loop_state["optimizer"]
    criterion = loop_state["criterion"]
    device = loop_state["device"]

    model.train()
    for images, labels in tqdm(train_loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()



def _collect_eval_outputs(
    model,
    sketch_loader,
    criterion,
    device: str,
    desc: str,
) -> tuple[list[int], list[int], float, int]:
    """Collect sketch val predictions and aggregated loss statistics."""
    y_true: list[int] = []
    y_pred: list[int] = []
    val_sum = 0.0
    val_n = 0

    with torch.no_grad():
        for images, labels in tqdm(sketch_loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            val_sum += float(loss.item()) * labels.size(0)
            val_n += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred, val_sum, val_n



def _run_eval_epoch(loop_state: dict, desc: str) -> dict:
    """Run one evaluation epoch on sketch_val."""
    model = loop_state["model"]
    sketch_loader = loop_state["sketch_loader"]
    criterion = loop_state["criterion"]
    device = loop_state["device"]

    model.eval()
    y_true, y_pred, val_sum, val_n = _collect_eval_outputs(
        model,
        sketch_loader,
        criterion,
        device,
        desc,
    )

    if not y_true:
        raise RuntimeError("sketch_val evaluation produced zero matched samples")

    val_loss = val_sum / max(val_n, 1)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    top1 = float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true))
    return {
        "val_loss": val_loss,
        "weighted_f1": weighted_f1,
        "top1_accuracy": top1,
        "samples": len(y_true),
    }


def _fit_model(loop_state: dict, settings: dict) -> tuple[dict, list[float]]:
    """Train model for all epochs and keep the best checkpoint by weighted f1."""
    best = {"weighted_f1": -1.0, "top1_accuracy": 0.0, "samples": 0, "state_dict": None}
    val_loss_history: list[float] = []

    for epoch in range(settings["epochs"]):
        train_desc = (
            f"[{settings['log_scope']}] "
            f"{settings['run_title']} train {epoch + 1}/{settings['epochs']}"
        )
        eval_desc = (
            f"[{settings['log_scope']}] "
            f"{settings['run_title']} sketch_val {epoch + 1}/{settings['epochs']}"
        )
        _run_train_epoch(loop_state, train_desc)
        eval_state = _run_eval_epoch(loop_state, eval_desc)
        val_loss_history.append(eval_state["val_loss"])

        if eval_state["weighted_f1"] > best["weighted_f1"]:
            best["weighted_f1"] = eval_state["weighted_f1"]
            best["top1_accuracy"] = eval_state["top1_accuracy"]
            best["samples"] = eval_state["samples"]
            best["state_dict"] = loop_state["model"].state_dict()

    if best["state_dict"] is not None:
        loop_state["model"].load_state_dict(best["state_dict"])
    return best, val_loss_history



def _build_data_state(dataset_paths: dict[str, str], config: dict) -> dict:
    """Build loaders and class metadata for one run."""
    train_loader, class_counts, class_names, sampling_mode = _build_train_loader(
        dataset_paths,
        config,
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    sketch_loader = _build_sketch_val_loader(dataset_paths, class_to_idx, config)
    return {
        "train_loader": train_loader,
        "sketch_loader": sketch_loader,
        "class_counts": class_counts,
        "class_names": class_names,
        "sampling_mode": sampling_mode,
    }



def _build_model_state(config: dict, device: str, data_state: dict) -> dict:
    """Build model and optimizer state for one run."""
    model, arch = _build_model(config, len(data_state["class_names"]))
    strategy = _apply_trainable_strategy(model, config, arch)
    model = model.to(device)
    criterion, loss_mode = _build_loss(config, data_state["class_counts"], device)
    optimizer, lr_mode = _build_optimizer(model, config, arch, strategy)
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
        "model_arch": arch,
        "loss_mode": loss_mode,
        "unfreeze_mode": strategy,
        "lr_mode": lr_mode,
    }



def _build_loop_state(dataset_paths: dict[str, str], config: dict, device: str) -> dict:
    """Build loaders, model, optimizer, and metadata for one run."""
    data_state = _build_data_state(dataset_paths, config)
    model_state = _build_model_state(config, device, data_state)
    return {
        **data_state,
        **model_state,
    }


def _save_run_artifacts(
    project_root: Path,
    settings: dict,
    run_state: dict,
    summary: dict,
) -> dict:
    """Persist checkpoint and metrics, then return stage payload."""
    model_dir = project_root / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{settings['run_name']}_best.pt"
    metrics_file = model_dir / f"{settings['run_name']}_metrics.json"

    model = run_state["model"]
    class_names = run_state["class_names"]
    if settings["save_checkpoint"]:
        torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, model_file)

    metrics = {
        "run_name": settings["run_name"],
        "run_title": settings["run_title"],
        "seed": settings["seed"],
        "model_arch": run_state["model_arch"],
        "weighted_f1": summary["best"]["weighted_f1"],
        "top1_accuracy": summary["best"]["top1_accuracy"],
        "val_samples": summary["best"]["samples"],
        "device": summary["device"],
        "gpu_name": summary["gpu_name"],
        "num_epochs": settings["epochs"],
        "sampling_mode": run_state["sampling_mode"],
        "loss_mode": run_state["loss_mode"],
        "unfreeze_mode": run_state["unfreeze_mode"],
        "lr_mode": run_state["lr_mode"],
        "val_loss_history": [round(value, 6) for value in summary["val_loss_history"]],
        "run_signature": _run_signature(summary["config"]),
    }
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "status": "ready",
        "run_name": settings["run_name"],
        "run_title": settings["run_title"],
        "weighted_f1": summary["best"]["weighted_f1"],
        "top1_accuracy": summary["best"]["top1_accuracy"],
        "val_samples": summary["best"]["samples"],
        "device": summary["device"],
        "gpu_name": summary["gpu_name"],
        "model_file": str(model_file) if settings["save_checkpoint"] else "",
        "metrics_file": str(metrics_file),
        "val_loss_history": [round(value, 6) for value in summary["val_loss_history"]],
        "cached": False,
    }



def _train_one_run(dataset_paths: dict[str, str], project_root: Path, config: dict) -> dict:
    """Train one run, evaluate each epoch, and save only necessary artifacts."""
    settings = _prepare_run_settings(config)
    _set_seed(settings["seed"])

    cached = _resolve_cached_run(project_root, config, settings)
    if cached is not None:
        return cached

    device = str(config.get("resolved_device", "cpu"))
    gpu_name = str(config.get("resolved_gpu_name", "N/A"))
    if settings["log_device"]:
        print(
            f"[PROGRESS][{settings['log_scope']}] "
            f"run={settings['run_title']}, device={device}, gpu_name={gpu_name}"
        )

    _prepare_training_artifacts(project_root)
    run_state = _build_loop_state(dataset_paths, config, device)
    best, val_loss_history = _fit_model(run_state, settings)
    summary = {
        "best": best,
        "val_loss_history": val_loss_history,
        "device": device,
        "gpu_name": gpu_name,
        "config": config,
    }
    return _save_run_artifacts(project_root, settings, run_state, summary)

def run_baseline_variant(
    project_config: dict,
    data_state: dict,
    variant_id: str,
    variant_overrides: dict,
) -> dict:
    """Run one variant by reusing the baseline training pipeline."""
    cfg = dict(project_config)
    cfg.update(variant_overrides)
    cfg["run_name"] = variant_id
    cfg.setdefault("allow_cache", True)
    return _train_one_run(data_state["dataset_paths"], Path(project_config["project_root"]), cfg)



def run_baseline_stage(project_config: dict, data_state: dict) -> dict:
    """Run baseline stage with essential logs only."""
    print("[START][baseline] Baseline stage started")

    deps_ok, missing = check_baseline_dependencies()
    if not deps_ok:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")

    cfg = dict(project_config)
    cfg.update(
        {
            "run_name": "baseline",
            "run_title": "baseline_frozen",
            "log_scope": "baseline",
            "log_device": True,
            "allow_cache": True,
            "save_checkpoint": True,
            "model_arch": "resnet34",
            "sampling_mode": "shuffle",
            "loss_mode": "cross_entropy",
            "unfreeze_strategy": "baseline_frozen",
        }
    )
    result = _train_one_run(data_state["dataset_paths"], Path(project_config["project_root"]), cfg)
    print(
        f"[RESULT][baseline] weighted_f1={result['weighted_f1']:.4f}, "
        f"val_samples={result['val_samples']}, device={result['device']}"
    )
    return result



def _selected_baseline_compare_variants(config: dict) -> list[dict]:
    """Select configured baseline comparison variants while keeping stable order."""
    names = config.get("baseline_compare_variant_names", [])
    if not names:
        return BASELINE_COMPARE_VARIANTS
    selected = {str(name).strip() for name in names}
    return [item for item in BASELINE_COMPARE_VARIANTS if item["name"] in selected]



def _baseline_compare_epochs(config: dict, name: str) -> int:
    """Resolve epoch count for each baseline comparison variant."""
    default_epochs = int(config.get("baseline_compare_num_epochs", 1))
    by_name = config.get("baseline_compare_epochs_by_name", {})
    if isinstance(by_name, dict) and name in by_name:
        return int(by_name[name])
    return default_epochs




def _build_variant_context(
    project_config: dict,
    baseline_state: dict,
    scope: str,
    delta_label: str,
    epoch_resolver,
) -> dict:
    """Build shared context for variant runs in comparison and ablation stages."""
    return {
        "seed": int(project_config.get("baseline_seed", 42)),
        "anchor_f1": float(baseline_state.get("weighted_f1") or 0.0),
        "scope": scope,
        "delta_label": delta_label,
        "epoch_resolver": epoch_resolver,
    }

def _run_variant_result_row(
    project_config: dict,
    data_state: dict,
    variant: dict,
    context: dict,
) -> dict:
    """Run one variant and return a comparable result row."""
    name = variant["name"]
    seed = int(context["seed"])
    anchor_f1 = float(context["anchor_f1"])
    scope = str(context["scope"])
    delta_label = str(context["delta_label"])
    epochs = context["epoch_resolver"](project_config, name)

    overrides = dict(variant["overrides"])
    overrides.update(
        {
            "baseline_seed": seed,
            "baseline_num_epochs": epochs,
            "log_scope": scope,
            "run_title": name,
            "run_name": variant["key"],
            "log_device": False,
            "allow_cache": True,
            "save_checkpoint": True,
        }
    )

    result = run_baseline_variant(project_config, data_state, variant["key"], overrides)
    weighted_f1 = float(result.get("weighted_f1", 0.0))
    delta_vs_anchor = weighted_f1 - anchor_f1
    print(
        f"[RESULT][{scope}] {name} weighted_f1={weighted_f1:.4f}, "
        f"{delta_label}={delta_vs_anchor:+.4f}"
    )
    return {
        "name": name,
        "seed": seed,
        "weighted_f1": weighted_f1,
        "delta_vs_anchor": delta_vs_anchor,
    }



def run_baseline_compare_stage(
    project_config: dict,
    data_state: dict,
    baseline_state: dict,
) -> dict:
    """Run baseline comparison models separately from ablation."""
    print("[START][baseline_compare] Baseline comparison stage started")

    deps_ok, missing = check_baseline_dependencies()
    if not deps_ok:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")

    variants = _selected_baseline_compare_variants(project_config)
    context = _build_variant_context(
        project_config,
        baseline_state,
        scope="baseline_compare",
        delta_label="delta_vs_resnet34",
        epoch_resolver=_baseline_compare_epochs,
    )
    rows = [
        _run_variant_result_row(project_config, data_state, variant, context)
        for variant in variants
    ]
    return {"status": "ready", "comparisons": rows}
