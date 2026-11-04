"""Dataset setup stage for Drawn Apart."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION_SLUG = "drawn-apart-aicc-round-3"
REQUIRED_PARTS = [
    "cartoon",
    "photograph",
    "sketch/unlabeled",
    "sketch_val",
    "sketch_test",
    "val.csv",
]


def _required_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in REQUIRED_PARTS}


def _missing_parts(path_map: dict[str, Path]) -> list[str]:
    return [name for name, path in path_map.items() if not path.exists()]


def _find_dataset_root_under(base_root: Path) -> Path | None:
    """Find a valid dataset root at base root or one nested level below."""
    if not base_root.exists():
        return None

    if len(_missing_parts(_required_paths(base_root))) == 0:
        return base_root

    for child in base_root.iterdir():
        if not child.is_dir():
            continue
        if len(_missing_parts(_required_paths(child))) == 0:
            return child
        for grandchild in child.iterdir():
            if not grandchild.is_dir():
                continue
            if len(_missing_parts(_required_paths(grandchild))) == 0:
                return grandchild

    return None


def _find_ready_root(project_root: Path) -> Path | None:
    """Find local dataset root from supported layouts."""
    roots = [
        (project_root / "DATA").resolve(),
        (project_root / "data").resolve(),
    ]

    for root in roots:
        found = _find_dataset_root_under(root)
        if found is not None:
            return found
    return None


def _resolve_kaggle_cli() -> str:
    """Resolve Kaggle CLI path from current environment first."""
    scripts_dir = Path(sys.executable).resolve().parent
    candidate = scripts_dir / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if candidate.exists():
        return str(candidate)

    fallback = shutil.which("kaggle")
    if fallback:
        return fallback

    raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")


def _download_dataset(project_root: Path) -> Path:
    """Download dataset zip via Kaggle CLI and extract into project data."""
    download_dir = project_root / "downloads" / COMPETITION_SLUG
    extract_dir = project_root / "data"
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    kaggle_cli = _resolve_kaggle_cli()
    cmd = [
        kaggle_cli,
        "competitions",
        "download",
        "-c",
        COMPETITION_SLUG,
        "-p",
        str(download_dir),
        "--force",
    ]

    try:
        subprocess.run(cmd, env=os.environ.copy(), check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle") from error
    except subprocess.CalledProcessError as error:
        msg = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"Kaggle download failed: {msg}") from error

    archives = sorted(download_dir.glob("*.zip"))
    if not archives:
        raise RuntimeError("Kaggle download finished but no zip file was found")

    for archive in archives:
        with zipfile.ZipFile(archive, "r") as zip_obj:
            zip_obj.extractall(extract_dir)

    ready_root = _find_ready_root(project_root)
    if ready_root is None:
        raise RuntimeError("Downloaded dataset does not match expected folder structure")
    return ready_root


def run_data_pipeline_stage(project_root: Path) -> dict:
    """Prepare dataset paths with local-first lookup and Kaggle fallback."""
    print("[START][data] Data pipeline stage started")
    print("[PROGRESS][data] Kaggle auth source checked")

    root = _find_ready_root(project_root)
    source = "local"

    if root is None:
        print("[PROGRESS][data] Local dataset incomplete, trying Kaggle download")
        root = _download_dataset(project_root)
        source = "downloaded"

    path_map = _required_paths(root)
    missing = _missing_parts(path_map)
    if missing:
        raise RuntimeError(f"Dataset incomplete: {', '.join(missing)}")

    print(f"[RESULT][data] Dataset ready: {root}")
    return {
        "status": "ready",
        "dataset_source": source,
        "dataset_root": str(root),
        "dataset_paths": {k: str(v) for k, v in path_map.items()},
        "missing_parts": [],
    }
