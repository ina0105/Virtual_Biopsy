#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combine biopsy purity with absolute-negative and inner-tumor "
            "labels into one continuous purity volume for a single patient."
        )
    )
    p.add_argument("--dataset-dir", type=Path, default=Path("CNN_dataset"))
    p.add_argument("--patient-id", type=str, required=True, help="Patient ID, e.g. P01")
    p.add_argument(
        "--seg-path",
        type=Path,
        required=True,
        help="Path to the whole-tumor (or multi-class) segmentation NIfTI for this patient.",
    )
    p.add_argument(
        "--inner-seg-path",
        type=Path,
        default=None,
        help=(
            "Optional second segmentation NIfTI used only for inner-tumor "
            "labels (absolute positives). If provided, --inner-pos-labels are "
            "looked up in this volume instead of --seg-path."
        ),
    )
    p.add_argument(
        "--abs-neg-labels",
        type=int,
        nargs="+",
        required=True,
        help="One or more integer segmentation labels to treat as absolute negatives (purity=0).",
    )
    p.add_argument(
        "--inner-type",
        type=str,
        required=True,
        help=(
            "Name of the inner-tumor class (e.g. 'enhancing'); used only in the "
            "output filename to document which region was set to purity=1."
        ),
    )
    p.add_argument(
        "--inner-mask",
        type=Path,
        default=None,
        help=(
            "Optional NIfTI mask for the chosen inner-tumor type. If provided, "
            "voxels with values > 0 in this mask are treated as inner-tumor and "
            "--inner-pos-labels is ignored."
        ),
    )
    p.add_argument(
        "--inner-pos-labels",
        type=int,
        nargs="+",
        required=False,
        help=(
            "Segmentation labels corresponding to the chosen inner-tumor type. "
            "These voxels are treated as absolute positives (purity=1)."
        ),
    )
    p.add_argument(
        "--purity-basename",
        type=str,
        default="biopsy_purity",
        help=(
            "Basename of the biopsy purity NIfTI (default 'biopsy_purity' for "
            "<patient>_biopsy_purity.nii.gz)."
        ),
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="absneg_innertumor",
        help=(
            "Base suffix for the new purity volume, which will become "
            "<patient>_biopsy_purity_<suffix>_<inner-type>.nii.gz."
        ),
    )
    p.add_argument(
        "--allow-missing-purity",
        action="store_true",
        help=(
            "Allow patients without a biopsy purity file: start from a "
            "zero-filled purity volume and apply only abs-neg and inner "
            "constraints (no biopsy information)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    labels_dir = dataset_dir / "labels"

    purity_path = labels_dir / f"{args.patient_id}_{args.purity_basename}.nii.gz"
    if not args.seg_path.exists():
        raise FileNotFoundError(f"Segmentation NIfTI not found: {args.seg_path}")

    print(f"Patient:              {args.patient_id}")
    print(f"Dataset dir:          {dataset_dir}")
    print(f"Input purity volume:  {purity_path}")
    print(f"Segmentation (absneg): {args.seg_path}")
    print(f"Abs-neg labels:       {args.abs_neg_labels}")
    print(f"Inner-tumor type:     {args.inner_type}")
    if args.inner_seg_path is not None:
        print(f"Inner-tumor seg:      {args.inner_seg_path}")
    if args.inner_mask is not None:
        print(f"Inner-tumor mask:     {args.inner_mask}")
    else:
        print(f"Inner-tumor labels:   {args.inner_pos_labels}")

    seg_img = nib.load(str(args.seg_path))
    seg_data = seg_img.get_fdata(dtype=np.float32)

    if seg_data.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {seg_data.shape}")

    if purity_path.exists():
        purity_img = nib.load(str(purity_path))
        purity_data = purity_img.get_fdata(dtype=np.float32)
        if purity_data.shape != seg_data.shape:
            raise ValueError(
                f"Shape mismatch between purity {purity_data.shape} and segmentation {seg_data.shape}. "
                "They must be in the same space and resolution."
            )
    else:
        if not args.allow_missing_purity:
            raise FileNotFoundError(f"Purity label not found: {purity_path}")
        purity_data = np.zeros(seg_data.shape, dtype=np.float32)
        purity_img = seg_img

    seg_int = seg_data.astype(np.int32)

    inner_seg_int = None
    if args.inner_seg_path is not None:
        if not args.inner_seg_path.exists():
            raise FileNotFoundError(f"Inner-tumor segmentation NIfTI not found: {args.inner_seg_path}")
        inner_seg_img = nib.load(str(args.inner_seg_path))
        inner_seg_data = inner_seg_img.get_fdata(dtype=np.float32)
        if inner_seg_data.shape != purity_data.shape:
            raise ValueError(
                f"Shape mismatch between purity {purity_data.shape} and inner segmentation {inner_seg_data.shape}. "
                "They must be in the same space and resolution."
            )
        inner_seg_int = inner_seg_data.astype(np.int32)

    abs_neg_set = set(int(v) for v in args.abs_neg_labels)
    absneg_mask = np.isin(seg_int, list(abs_neg_set))

    if args.inner_mask is not None:
        if not args.inner_mask.exists():
            raise FileNotFoundError(f"Inner-tumor mask not found: {args.inner_mask}")
        inner_img = nib.load(str(args.inner_mask))
        inner_data = inner_img.get_fdata(dtype=np.float32)
        if inner_data.shape != purity_data.shape:
            raise ValueError(
                f"Shape mismatch between purity {purity_data.shape} and inner mask {inner_data.shape}. "
                "They must be in the same space and resolution."
            )
        inner_mask = inner_data > 0
    else:
        if not args.inner_pos_labels:
            raise ValueError(
                "Either --inner-mask must be provided, or --inner-pos-labels must be set "
                "to one or more label IDs inside a segmentation."
            )
        inner_pos_set = set(int(v) for v in args.inner_pos_labels)
        if inner_seg_int is not None:
            inner_mask = np.isin(inner_seg_int, list(inner_pos_set))
        else:
            inner_mask = np.isin(seg_int, list(inner_pos_set))


    biopsy_mask = purity_data > 0.0

    absneg_effective = absneg_mask & ~biopsy_mask
    inner_effective = inner_mask & ~biopsy_mask

    print(f"  Absolute-negative voxels (all):     {int(absneg_mask.sum())}")
    print(f"  Absolute-negative voxels (applied): {int(absneg_effective.sum())}")
    print(f"  Inner-tumor voxels (all):           {int(inner_mask.sum())}")
    print(f"  Inner-tumor voxels (applied):       {int(inner_effective.sum())}")
    print(f"  Biopsy voxels (ground truth):       {int(biopsy_mask.sum())}")

    new_purity = purity_data.copy()

    new_purity[absneg_effective] = 0.0

    new_purity[inner_effective] = 1.0

    suffix = f"{args.output_suffix}_{args.inner_type}"
    out_purity_name = f"{args.patient_id}_{args.purity_basename}_{suffix}.nii.gz"
    out_purity_path = labels_dir / out_purity_name

    out_purity_img = nib.Nifti1Image(new_purity.astype(np.float32), purity_img.affine, purity_img.header)
    nib.save(out_purity_img, str(out_purity_path))

    print("Done.")
    print(f"  Saved combined purity volume: {out_purity_path}")


if __name__ == "__main__":
    main()
