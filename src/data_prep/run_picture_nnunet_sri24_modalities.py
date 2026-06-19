#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def normalize_patient_id(raw_value: str) -> str:
    value = raw_value.strip().upper()
    if value.startswith("P"):
        value = value[1:]
    patient_num = int(value)
    if patient_num < 1 or patient_num > 99:
        raise ValueError("Patient number must be between 1 and 99")
    return f"P{patient_num:02d}"


def windows_to_wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    drive = resolved[0].lower()
    rest = resolved[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def get_modalities(patient_dir: Path, patient_id: str) -> dict[str, Path]:
    modalities = {
        "t1ce": patient_dir / f"{patient_id}T1G_SRI24.nii.gz",
        "t1": patient_dir / f"{patient_id}T01_SRI24.nii.gz",
        "t2": patient_dir / f"{patient_id}T02_SRI24.nii.gz",
        "flair": patient_dir / f"{patient_id}FLR_SRI24.nii.gz",
    }
    missing = [name for name, path in modalities.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required modality files for {patient_id}: {missing_text}")
    return modalities


def build_wsl_command(args: argparse.Namespace, patient_id: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    patient_dir = args.base_dir / patient_id
    modalities = get_modalities(patient_dir, patient_id)

    runner = repo_root / "scripts" / "run_picture_nnunet_sri.py"
    wsl_runner = windows_to_wsl_path(runner)

    cmd_parts = [
        "python3",
        f"'{wsl_runner}'",
        "--t1ce",
        f"'{windows_to_wsl_path(modalities['t1ce'])}'",
        "--t1",
        f"'{windows_to_wsl_path(modalities['t1'])}'",
        "--t2",
        f"'{windows_to_wsl_path(modalities['t2'])}'",
        "--flair",
        f"'{windows_to_wsl_path(modalities['flair'])}'",
        "--sessiontype",
        args.sessiontype,
        "--wdir-postfix",
        f"sri24_{patient_id.lower()}",
    ]

    if args.mni:
        cmd_parts.append("--mni")
    if args.remove_intermediate_files:
        cmd_parts.append("--remove-intermediate-files")
    if args.run_skullstrip:
        cmd_parts.append("--run-skullstrip")

    bash_cmd = " ".join(cmd_parts)
    return ["wsl", "bash", "-lc", bash_cmd]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run picture-nnunet on SRI files in Virtual_biopsy/SRI24_modalities."
    )
    parser.add_argument(
        "patient",
        nargs="?",
        default=None,
        help="Patient number/id (e.g. 1, 01, P01). Omit only when using --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run all patient folders found in base-dir.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "SRI24_modalities",
        help="Path to Virtual_biopsy/SRI24_modalities",
    )
    parser.add_argument(
        "--sessiontype",
        choices=["preop", "postop", "postop_beta"],
        default="preop",
        help="Model session type (default: preop).",
    )
    parser.add_argument("--mni", action="store_true", help="Use MNI atlas instead of SRI atlas.")
    parser.add_argument(
        "--remove-intermediate-files",
        action="store_true",
        help="Delete temporary wdir folder after each run.",
    )
    parser.add_argument(
        "--run-skullstrip",
        action="store_true",
        help="Enable HD-BET skullstripping.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    if args.all:
        patients = sorted([p.name for p in args.base_dir.glob("P*") if p.is_dir()])
    else:
        if args.patient is None:
            raise ValueError("Provide a patient id or use --all")
        patients = [normalize_patient_id(args.patient)]

    for patient_id in patients:
        command = build_wsl_command(args, patient_id)
        print(f"\n[{patient_id}]")
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command)
            if result.returncode != 0:
                raise RuntimeError(f"Inference failed for {patient_id} (exit {result.returncode})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)