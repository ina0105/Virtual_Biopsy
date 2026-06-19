

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "datasets"
BUILD_SCRIPT = BASE_DIR / "build_cnn_dataset.py"
LABEL_SCRIPT = BASE_DIR / "make_purity_with_absneg_and_innertumor.py"
TUMORSYNTH_DIR = BASE_DIR / "TumorSynth_Outputs"
PICTURE_DIR = BASE_DIR / "picture_outputs"

DATASET_GROUPS: dict[str, dict] = {
    "CNN_dataset_standard_MRI": {
        "modalities": ["T01", "T1G", "T02", "FLR"],
        "description": "Standard clinical MRI (T1, T1+Gd, T2, FLAIR)  —  4 channels",
    },
    "CNN_dataset_nonPET": {
        "modalities": ["T01", "T1G", "T02", "FLR", "ADC", "BFa", "BFd", "BVd", "dFA"],
        "description": "Full MRI without PET  —  9 channels",
    },
    "CNN_dataset_all_modalities": {
        "modalities": [
            "T01", "T1G", "T02", "FLR",
            "ADC", "BFa", "BFd", "BVd", "C18",
            "dFA", "FET", "S24", "SR9",
        ],
        "description": "All 13 modalities including C18, FET, S24, SR9 PET",
    },
    "CNN_dataset_ADC_FET": {
        "modalities": ["ADC", "FET"],
        "description": "ADC diffusion + FET PET  —  2 channels",
    },
    "CNN_dataset_diffusion": {
        "modalities": ["ADC", "dFA"],
        "description": "Diffusion only: ADC + dFA  —  2 channels",
    },
    "CNN_dataset_perfusion": {
        "modalities": ["BFa", "BFd", "BVd"],
        "description": "Perfusion only: Blood Flow (asc/desc) + Blood Volume  —  3 channels",
    },
    "CNN_dataset_PET": {
        "modalities": ["C18", "FET", "S24", "SR9"],
        "description": "PET only: C18, FET, S24, SR9  —  4 channels",
    },
    "CNN_dataset_standard_MRI_PET": {
        "modalities": ["T01", "T1G", "T02", "FLR", "C18", "FET", "S24", "SR9"],
        "description": "Standard clinical MRI + PET tracers  —  8 channels",
    },
}

ALL_PATIENTS = [f"P{i:02d}" for i in range(1, 21)]

# TumorSynth / FreeSurfer labels treated as absolute negatives
ABS_NEG_LABELS = [3, 4, 5, 6, 11, 12]

# PICTURE label treated as absolute positive (enhancing tumour)
ABS_POS_LABEL = 3


def run(cmd: list[str], dry_run: bool) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  [WARNING] command exited with code {result.returncode}")


def picture_seg_path(pid: str) -> Path:
    return PICTURE_DIR / f"segmentation_native_{pid}T1G_SRI24.nii.gz"


def tumorsynth_path(pid: str) -> Path:
    return TUMORSYNTH_DIR / f"{pid}_tumor_mask.nii.gz"


def build_images(dataset_name: str, modalities: list[str],
                 dataset_dir: Path, biopsy_roi_mm: float, dry_run: bool) -> None:
    cmd = [
        sys.executable, str(BUILD_SCRIPT),
        "--dataset-dir", str(dataset_dir),
        "--modalities", *modalities,
    ]
    if biopsy_roi_mm > 0:
        cmd += ["--biopsy-roi-mm", str(biopsy_roi_mm)]
    run(cmd, dry_run)


def build_combined_labels(dataset_dir: Path, dry_run: bool) -> None:
    for pid in ALL_PATIENTS:
        seg_path = tumorsynth_path(pid)
        inner_seg = picture_seg_path(pid)

        if not dry_run:
            if not seg_path.exists():
                print(f"  [SKIP] {pid}: TumorSynth mask not found: {seg_path}")
                continue
            if not inner_seg.exists():
                print(f"  [SKIP] {pid}: PICTURE segmentation not found: {inner_seg}")
                continue

        print(f"  {pid}: absneg={seg_path.name}  abspos={inner_seg.name}")

        cmd = [
            sys.executable, str(LABEL_SCRIPT),
            "--dataset-dir", str(dataset_dir),
            "--patient-id", pid,
            "--seg-path", str(seg_path),
            "--abs-neg-labels", *[str(v) for v in ABS_NEG_LABELS],
            "--inner-type", "enhancing",
            "--inner-seg-path", str(inner_seg),
            "--inner-pos-labels", str(ABS_POS_LABEL),
            "--allow-missing-purity",
        ]
        run(cmd, dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_GROUPS),
        default=list(DATASET_GROUPS),
        metavar="NAME",
        help="Which dataset groups to build (default: all).",
    )
    p.add_argument("--biopsy-roi-mm", type=float, default=10.0,
                   help="Side length in mm for expanded biopsy ROI labels (default 10).")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip the image + base-label build step.")
    p.add_argument("--skip-labels", action="store_true",
                   help="Skip the combined abs-neg/abs-pos label step.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("BUILD ALL SPECIALISED DATASETS")
    print("=" * 70)
    if args.dry_run:
        print("  *** DRY RUN — no files will be written ***")
    print()

    print("NOTE: Using native PICTURE segmentations for all patients.")
    print("      (Atlas segs are 3-5x oversegmented; native segs have correct affine.)")
    print()

    DATASET_DIR.mkdir(exist_ok=True)
    for name in args.datasets:
        cfg = DATASET_GROUPS[name]
        dataset_dir = DATASET_DIR / name

        print(f"{'=' * 70}")
        print(f"Dataset : {name}")
        print(f"Desc    : {cfg['description']}")
        print(f"Channels: {cfg['modalities']}")
        print(f"Out dir : {dataset_dir}")
        print()

        if not args.skip_build:
            print("[1/2] Building images + base biopsy labels …")
            build_images(name, cfg["modalities"], dataset_dir, args.biopsy_roi_mm, args.dry_run)
            print()

        if not args.skip_labels:
            print("[2/2] Applying abs-neg (TumorSynth) + abs-pos (PICTURE) labels …")
            build_combined_labels(dataset_dir, args.dry_run)
            print()

    print("=" * 70)
    print("All datasets complete.")
    if not args.dry_run:
        print()
        print("Datasets created:")
        for name in args.datasets:
            d = DATASET_DIR / name
            if d.exists():
                imgs = list((d / "images").glob("*.nii.gz")) if (d / "images").exists() else []
                lbls = list((d / "labels").glob("*.nii.gz")) if (d / "labels").exists() else []
                print(f"  {name}/  ({len(imgs)} images, {len(lbls)} label files)")


if __name__ == "__main__":
    main()
