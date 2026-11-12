#!/usr/bin/env python3
"""Read-only validation for the documented project layout and saved artifacts."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
import os

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML is required. Run: python -m pip install -r requirements.txt\n")
    raise SystemExit(2)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SOURCE_ROOT.parent


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
    def error(self, message: str) -> None:
        self.errors.append(message); print(f"ERROR: {message}")
    def warning(self, message: str) -> None:
        self.warnings.append(message); print(f"WARNING: {message}")
    def ok(self, message: str) -> None:
        print(f"OK: {message}")


def load_yaml(path: Path, reporter: Reporter) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        reporter.error(f"Cannot read {path}: {exc}"); return {}
    if not isinstance(value, dict):
        reporter.error(f"{path} must contain a YAML mapping."); return {}
    return value


def workspace_path(value: str) -> Path:
    return WORKSPACE_ROOT / Path(value)


def validate_splits(manifest: dict, reporter: Reporter) -> None:
    splits = manifest.get("splits", {})
    # Base directory for split CSV files (relative to workspace root)
    splits_dir = splits.get("sub_dir", "")
    required_columns = splits.get("required_columns", [])
    allowed_labels = {str(label) for label in splits.get("labels", [])}
    for split_name, expected in splits.get("files", {}).items():
        # Expected entry now uses 'file' instead of full path
        csv_rel_path = os.path.join(splits_dir, expected.get("file", ""))
        csv_path = workspace_path(csv_rel_path)
        if not csv_path.is_file():
            reporter.error(f"{split_name}: missing CSV at {csv_path}. Prepare the dataset and place the split there.")
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames or []) != required_columns:
                reporter.error(f"{split_name}: expected columns {required_columns}, found {reader.fieldnames} in {csv_path}.")
                continue
            rows, labels = 0, Counter()
            for row in reader:
                rows += 1
                labels[str(row.get("label", ""))] += 1
        expected_counts = {str(key): value for key, value in expected.get("label_counts", {}).items()}
        if rows != expected.get("rows"):
            reporter.error(f"{split_name}: expected {expected.get('rows')} rows, found {rows}.")
        elif set(labels) - allowed_labels:
            reporter.error(f"{split_name}: unsupported labels {sorted(set(labels) - allowed_labels)}.")
        elif dict(labels) != expected_counts:
            reporter.error(f"{split_name}: expected label counts {expected_counts}, found {dict(labels)}.")
        else:
            reporter.ok(f"{split_name} split ({rows} rows; labels {dict(labels)}).")


def validate_artifacts(manifest: dict, reporter: Reporter) -> None:
    info = manifest.get("historical_artifacts", {})
    root = workspace_path(info.get("root", "Outputs"))
    for experiment in info.get("experiments", []):
        for relative in info.get("expected_files", []):
            candidate = root / experiment / relative
            if not candidate.is_file():
                reporter.warning(f"{experiment}: historical artifact not found: {candidate}. This does not affect config validation.")
        if (root / experiment).is_dir(): reporter.ok(f"Historical artifact directory found for {experiment}.")


def validate_hardware(profile: dict, strict: bool, reporter: Reporter) -> None:
    accelerator = profile.get("hardware", {}).get("accelerator", "any")
    if accelerator != "cuda":
        reporter.ok(f"Profile accepts {accelerator} execution."); return
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if available:
        reporter.ok("CUDA is available for the gpu profile.")
    else:
        message = "CUDA is unavailable; use --profile cpu or install a CUDA-compatible PyTorch build."
        (reporter.error if strict else reporter.warning)(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate project configuration without writing files.")
    parser.add_argument("--profile", choices=("cpu", "gpu", "experiment"), default="experiment")
    parser.add_argument("--strict-hardware", action="store_true")
    args = parser.parse_args()
    reporter = Reporter()
    project = load_yaml(SOURCE_ROOT / "configs" / "project.yaml", reporter)
    manifest = load_yaml(SOURCE_ROOT / project.get("paths", {}).get("dataset_manifest", "configs/dataset_manifest.yaml"), reporter)
    profile = project.get("profiles", {}).get(args.profile)
    if profile is None: reporter.error(f"Profile '{args.profile}' is missing from project.yaml.")
    else:
        reporter.ok(f"Profile '{args.profile}' loaded."); validate_hardware(profile, args.strict_hardware, reporter)
    if manifest:
        validate_splits(manifest, reporter); validate_artifacts(manifest, reporter)
    print(f"\nValidation finished: {len(reporter.errors)} error(s), {len(reporter.warnings)} warning(s). No files were changed.")
    return 1 if reporter.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
