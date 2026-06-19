#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run picture-nnunet glioma segmentation on SRI-space MRI inputs."
    )
    parser.add_argument("--t1ce", type=Path, required=True, help="Path to T1+contrast (required).")
    parser.add_argument("--t1", type=Path, default=None, help="Path to T1w (optional).")
    parser.add_argument("--t2", type=Path, default=None, help="Path to T2w (optional).")
    parser.add_argument("--flair", type=Path, default=None, help="Path to FLAIR (optional).")
    parser.add_argument(
        "--sessiontype",
        type=str,
        default="preop",
        choices=["preop", "postop", "postop_beta"],
        help="Model variant to run.",
    )
    parser.add_argument("--mni", action="store_true", help="Use MNI atlas instead of SRI atlas.")
    parser.add_argument(
        "--wdir-postfix",
        type=str,
        default="sri_run",
        help="Working directory postfix under input folder /wdir/<postfix>.",
    )
    parser.add_argument(
        "--remove-intermediate-files",
        action="store_true",
        help="Delete /wdir/<postfix> after successful run.",
    )
    parser.add_argument(
        "--run-skullstrip",
        action="store_true",
        help="Run HD-BET skullstripping (off by default to avoid HD-BET packaging issues).",
    )
    return parser.parse_args()


def ensure_exists(path: Path | None, name: str, required: bool = False) -> Path | None:
    if path is None:
        if required:
            raise FileNotFoundError(f"Missing required argument: {name}")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    package_repo = repo_root / "picture-nnunet-package"
    hd_bet_repo = repo_root / "HD-BET"

    if package_repo.is_dir():
        sys.path.insert(0, str(package_repo))
    if hd_bet_repo.is_dir():
        sys.path.insert(0, str(hd_bet_repo))

    if not args.run_skullstrip:
        hd_bet_module = types.ModuleType("HD_BET")
        hd_bet_run_module = types.ModuleType("HD_BET.run")

        def _run_hd_bet_not_available(*_args, **_kwargs):
            raise RuntimeError(
                "HD-BET skullstripping is disabled in this run. Use --run-skullstrip to enable it."
            )

        hd_bet_run_module.run_hd_bet = _run_hd_bet_not_available
        sys.modules["HD_BET"] = hd_bet_module
        sys.modules["HD_BET.run"] = hd_bet_run_module

    from picture_nnunet_package.doInference import do_segmentation

    t1ce = ensure_exists(args.t1ce, "--t1ce", required=True)
    t1 = ensure_exists(args.t1, "--t1")
    t2 = ensure_exists(args.t2, "--t2")
    flair = ensure_exists(args.flair, "--flair")

    do_segmentation(
        t1ce=str(t1ce),
        t1=str(t1) if t1 else None,
        t2=str(t2) if t2 else None,
        flair=str(flair) if flair else None,
        sessionType=args.sessiontype,
        remove_intermediate_files=args.remove_intermediate_files,
        mni=args.mni,
        wdir_postfix=args.wdir_postfix,
        skip_skullstrip=(not args.run_skullstrip),
    )


if __name__ == "__main__":
    main()