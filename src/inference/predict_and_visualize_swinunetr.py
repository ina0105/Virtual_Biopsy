
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_semisupervised_regression_swinunetr import (
    build_swinunetr,
    _normalize_channel,
    load_metadata,
)

DEFAULT_WORK    = Path(__file__).resolve().parents[2] / "swinunetr_work"
DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset_final"
DEFAULT_OUT     = Path(__file__).resolve().parents[2] / "visualizations"
DEFAULT_SRI24   = Path(__file__).resolve().parents[2] / "SRI24_modalities"
BIOPSY_CSV      = None  
PATCH_SIZE      = (128, 128, 96)   
FEATURE_SIZE    = 24  
BRAIN_THRESHOLD = 100.0



def load_brain_mask(patient_id: str, sri24_dir: Path) -> np.ndarray | None:
    t1g_path = sri24_dir / patient_id / f"{patient_id}T1G_SRI24.nii.gz"
    if not t1g_path.exists():
        return None
    t1g = nib.load(str(t1g_path)).get_fdata(dtype=np.float32)
    return t1g > BRAIN_THRESHOLD


def apply_brain_mask_uint8(pred: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pred_clipped = np.clip(pred, 0.0, 1.0)
    if mask is not None:
        pred_clipped = np.where(mask, pred_clipped, 0.0)
    return (pred_clipped * 255).astype(np.uint8)



def _pad_to_patch(vol: np.ndarray, patch: tuple):
    pads = [(0, 0)]
    for s, p in zip(vol.shape[1:], patch):
        diff = max(0, p - s)
        pads.append((diff // 2, diff - diff // 2))
    return np.pad(vol, pads, mode="constant"), pads


def sliding_window_inference(
    model: nn.Module,
    image: np.ndarray,
    patch_size: tuple,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval()
    padded, pads = _pad_to_patch(image, patch_size)
    C, X, Y, Z = padded.shape
    px, py, pz = patch_size
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    pred_sum   = np.zeros((X, Y, Z), dtype=np.float32)
    weight_sum = np.zeros((X, Y, Z), dtype=np.float32)

    gx = np.exp(-((np.arange(px) - px / 2) ** 2) / (2 * (px / 4) ** 2))
    gy = np.exp(-((np.arange(py) - py / 2) ** 2) / (2 * (py / 4) ** 2))
    gz = np.exp(-((np.arange(pz) - pz / 2) ** 2) / (2 * (pz / 4) ** 2))
    gauss = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

    xs = sorted(set(list(range(0, max(1, X - px + 1), stride[0])) + [max(0, X - px)]))
    ys = sorted(set(list(range(0, max(1, Y - py + 1), stride[1])) + [max(0, Y - py)]))
    zs = sorted(set(list(range(0, max(1, Z - pz + 1), stride[2])) + [max(0, Z - pz)]))

    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = padded[:, x0:x0+px, y0:y0+py, z0:z0+pz]
                    t = torch.from_numpy(patch[None]).to(device)
                    out = model(t).squeeze().cpu().numpy()
                    pred_sum[x0:x0+px, y0:y0+py, z0:z0+pz]   += out * gauss
                    weight_sum[x0:x0+px, y0:y0+py, z0:z0+pz] += gauss

    pred = pred_sum / np.clip(weight_sum, 1e-8, None)
    orig = image.shape[1:]
    sx, sy, sz = pads[1][0], pads[2][0], pads[3][0]
    return pred[sx:sx+orig[0], sy:sy+orig[1], sz:sz+orig[2]]



def _best_slices(biopsy_df, vol_shape, plane, n=6):
    ax_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    size = vol_shape[ax_map[plane]]
    if biopsy_df is not None and not biopsy_df.empty:
        col = {"axial": "sri24_vox_z", "coronal": "sri24_vox_y", "sagittal": "sri24_vox_x"}[plane]
        if col in biopsy_df.columns:
            center = int(biopsy_df[col].median())
            half = (n // 2) * max(1, size // (n * 2))
            return list(np.linspace(max(0, center - half), min(size - 1, center + half), n, dtype=int))
    return list(np.linspace(size // 8, 7 * size // 8, n, dtype=int))


def _get_slice(vol, idx, plane):
    if plane == "axial":    return vol[:, :, idx]
    if plane == "coronal":  return vol[:, idx, :]
    if plane == "sagittal": return vol[idx, :, :]


def visualize(patient_id, t1_vol, pred_vol, biopsy_df, seg_vol, out_dir, planes, n_slices, suffix="swinunetr"):
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_map = {
        "axial":    ("sri24_vox_x", "sri24_vox_y", "sri24_vox_z"),
        "coronal":  ("sri24_vox_x", "sri24_vox_z", "sri24_vox_y"),
        "sagittal": ("sri24_vox_y", "sri24_vox_z", "sri24_vox_x"),
    }

    for plane in planes:
        slices = _best_slices(biopsy_df, t1_vol.shape, plane, n_slices)
        fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 7))
        fig.suptitle(f"{patient_id} — {plane} | {suffix} purity", fontsize=13)

        bx_h, bx_v, bx_sl = coord_map[plane]
        im = None

        for col, sl_idx in enumerate(slices):
            t1_sl   = _get_slice(t1_vol,   sl_idx, plane)
            pred_sl = _get_slice(pred_vol, sl_idx, plane)

            ax = axes[0, col]
            t1_norm = np.clip(t1_sl, 0, None)
            t1_norm = t1_norm / (t1_norm.max() + 1e-8)
            ax.imshow(t1_norm.T, cmap="gray", origin="lower", aspect="equal")
            brain_mask = t1_norm > 0.02
            pred_masked = np.where(brain_mask, pred_sl, np.nan)
            im = ax.imshow(pred_masked.T, cmap="jet", vmin=0, vmax=1,
                           alpha=0.55, origin="lower", aspect="equal")
            if biopsy_df is not None and not biopsy_df.empty:
                if all(c in biopsy_df.columns for c in [bx_h, bx_v, bx_sl]):
                    tol = max(2, int(t1_vol.shape[0] / 40))
                    nearby = biopsy_df[np.abs(biopsy_df[bx_sl] - sl_idx) <= tol]
                    if not nearby.empty:
                        ax.scatter(nearby[bx_h], nearby[bx_v],
                                   c=nearby["PAMES_purity"], cmap="RdYlGn",
                                   vmin=0, vmax=1, s=60, edgecolors="white",
                                   linewidths=0.8, zorder=5)
            ax.set_title(f"slice {sl_idx}", fontsize=8)
            ax.axis("off")

            # Row 1: pred only + seg contour
            ax2 = axes[1, col]
            ax2.imshow(pred_sl.T, cmap="jet", vmin=0, vmax=1, origin="lower", aspect="equal")
            if seg_vol is not None:
                seg_sl = _get_slice(seg_vol, sl_idx, plane)
                ax2.contour((seg_sl > 0).astype(float).T, levels=[0.5],
                            colors=["cyan"], linewidths=0.8, alpha=0.8)
            ax2.set_title("pred only", fontsize=8)
            ax2.axis("off")

        if im is not None:
            plt.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6, label="Predicted purity")

        handles = [mpatches.Patch(facecolor="cyan", label="tumor seg (contour)")]
        if biopsy_df is not None and not biopsy_df.empty:
            handles += [
                mpatches.Patch(facecolor="red",   label="GT purity high (biopsy)"),
                mpatches.Patch(facecolor="green", label="GT purity low (biopsy)"),
            ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)

        save_path = out_dir / f"{patient_id}_{plane}_{suffix}_prediction.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_biopsy_locations(patient_id, t1_vol, pred_vol, biopsy_df, out_dir, suffix="swinunetr", crop=30):
    if biopsy_df is None or biopsy_df.empty:
        return

    needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z", "PAMES_purity"}
    if not needed.issubset(biopsy_df.columns):
        return

    df = biopsy_df.copy().reset_index(drop=True)
    X, Y, Z = pred_vol.shape
    n = len(df)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"{patient_id} — biopsy locations ({suffix})", fontsize=18)

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        cx = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
        cy = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
        cz = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))

        t1_sl   = t1_vol[:, :, cz]
        pred_sl = pred_vol[:, :, cz]

        x0 = max(0, cx - crop); x1 = min(X, cx + crop)
        y0 = max(0, cy - crop); y1 = min(Y, cy + crop)
        t1_crop   = t1_sl[x0:x1, y0:y1]
        pred_crop = pred_sl[x0:x1, y0:y1]

        t1_norm = np.clip(t1_crop, 0, None)
        t1_norm = t1_norm / (t1_norm.max() + 1e-8)

        ax.imshow(t1_norm.T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(pred_crop.T, cmap="jet", vmin=0, vmax=1,
                  alpha=0.55, origin="lower", aspect="equal")

        # Mark the biopsy centre
        mx = cx - x0
        my = cy - y0
        ax.scatter([mx], [my], s=80, c="cyan", edgecolors="white",
                   linewidths=1.0, zorder=5, marker="+")

        true_val = row["PAMES_purity"]
        pred_val = float(pred_vol[cx, cy, cz])
        lbl = row.get("biopsy_id", str(i))
        color = "#d73027" if true_val > 0.5 else "#4575b4"
        ax.set_title(f"{lbl}  true={true_val:.2f}  pred={pred_val:.2f}",
                     fontsize=13, color=color)
        ax.axis("off")

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = out_dir / f"{patient_id}_{suffix}_biopsy_locations.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_biopsy_predictions(patient_id, pred_vol, biopsy_df, out_dir, suffix="swinunetr"):
    if biopsy_df is None or biopsy_df.empty:
        print("  [skip biopsy plot] no biopsy data")
        return

    needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z", "PAMES_purity"}
    if not needed.issubset(biopsy_df.columns):
        print("  [skip biopsy plot] missing required columns")
        return

    df = biopsy_df.copy().reset_index(drop=True)

    X, Y, Z = pred_vol.shape
    preds = []
    for _, row in df.iterrows():
        x = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
        y = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
        z = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))
        preds.append(float(pred_vol[x, y, z]))
    df["pred_purity"] = preds

    true_vals = df["PAMES_purity"].values
    pred_vals = df["pred_purity"].values
    labels    = df["biopsy_id"].values if "biopsy_id" in df.columns else [str(i) for i in range(len(df))]
    colors    = ["#d73027" if v > 0.5 else "#4575b4" for v in true_vals]  # red = high purity, blue = low

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{patient_id} — biopsy predictions ({suffix})", fontsize=13)

    ax1.scatter(true_vals, pred_vals, c=colors, s=80, edgecolors="k", linewidths=0.6, zorder=3)
    for i, lbl in enumerate(labels):
        ax1.annotate(lbl, (true_vals[i], pred_vals[i]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = (-0.05, 1.05)
    ax1.plot(lim, lim, "--", color="gray", linewidth=1, label="perfect")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("True purity (PAMES)"); ax1.set_ylabel("Predicted purity")
    ax1.set_title("Predicted vs True"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    order   = np.argsort(true_vals)
    x_pos   = np.arange(len(order))
    width   = 0.35
    ax2.bar(x_pos - width/2, true_vals[order],  width, label="True (PAMES)",
            color=[colors[i] for i in order], alpha=0.85, edgecolor="k", linewidth=0.5)
    ax2.bar(x_pos + width/2, pred_vals[order], width, label="Predicted",
            color="gray", alpha=0.75, edgecolor="k", linewidth=0.5)
    ax2.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([labels[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Purity"); ax2.set_title("Per-biopsy (sorted by true purity)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / f"{patient_id}_{suffix}_biopsy_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    csv_path = out_dir / f"{patient_id}_{suffix}_biopsy_predictions.csv"
    df[["biopsy_id", "PAMES_purity", "pred_purity", "TvsN_Predict"]].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient",          required=True)
    p.add_argument("--work-dir",         type=Path, default=DEFAULT_WORK)
    p.add_argument("--dataset-dir",      type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir",          type=Path, default=DEFAULT_OUT)
    p.add_argument("--sri24-dir",        type=Path, default=DEFAULT_SRI24)
    p.add_argument("--folds",            type=int, nargs="+", default=None)
    p.add_argument("--loo",              action="store_true",
                   help="Auto-detect the held-out fold for this patient (LOO CV mode). "
                        "Overrides --folds.")
    p.add_argument("--combined-suffix",  default="_biopsy_purity_absneg_innertumor_enhancing",
                   help="Label file suffix used to identify fully-labeled patients in LOO mode. "
                        "Pass '' to use all patients with images (older datasets with only "
                        "_biopsy_purity.nii.gz labels).")
    p.add_argument("--suffix",           default=None,
                   help="Output filename suffix. Defaults to 'swinunetr' or 'swinunetr_pretrained'.")
    p.add_argument("--pretrained-only",  action="store_true",
                   help="Run with raw BraTS pretrained weights (no fine-tuning)")
    p.add_argument("--pretrained-ckpt",  type=Path,
                   default=Path(__file__).resolve().parents[2] / "pretrained_weights/swinunetr_v1_ssl.pt")
    p.add_argument("--overlap",          type=float, default=0.5)
    p.add_argument("--device",           default=None)
    p.add_argument("--no-save-nifti",    action="store_true")
    p.add_argument("--planes",           nargs="+", default=["axial", "coronal", "sagittal"],
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--n-slices",         type=int, default=6)
    p.add_argument("--feature-size",     type=int, default=None,
                   help="SwinUNETR feature size. Auto-detected from metrics_summary.json if not set.")
    return p.parse_args()


def main():
    args = parse_args()
    pid = args.patient
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    
    img_path = args.dataset_dir / "images" / f"{pid}.nii.gz"
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")
    nii   = nib.load(str(img_path))
    raw   = nii.get_fdata(dtype=np.float32)
    image = np.moveaxis(raw, -1, 0)   # (C, X, Y, Z)
    print(f"Loaded {pid}: shape={image.shape}")

    global_mods, patients_info = load_metadata(args.dataset_dir)

    if args.loo:
        from sklearn.model_selection import LeaveOneOut
        labels_dir = args.dataset_dir / "labels"
        combined_suffix = args.combined_suffix
        if combined_suffix:
            fully_labeled_pids = [
                p for p, info in patients_info.items()
                if info.get("has_labels", False)
                and (args.dataset_dir / "images" / f"{p}.nii.gz").exists()
                and (labels_dir / f"{p}{combined_suffix}.nii.gz").exists()
            ]
        else:
            fully_labeled_pids = sorted(
                p for p, info in patients_info.items()
                if (args.dataset_dir / "images" / f"{p}.nii.gz").exists()
            )
        loo = LeaveOneOut()
        held_out_fold = None
        for fi, (_, va_idx) in enumerate(loo.split(range(len(fully_labeled_pids)))):
            if fully_labeled_pids[va_idx[0]] == pid:
                held_out_fold = fi
                break
        if held_out_fold is None:
            sys.exit(f"{pid} not found in fully-labeled patient list: {fully_labeled_pids}")
        folds = [held_out_fold]
        print(f"  LOO mode: using fold_{held_out_fold} (held-out fold for {pid})")
    else:
        folds = args.folds if args.folds is not None else [0]

    norm = np.empty_like(image)
    for ch in range(image.shape[0]):
        norm[ch] = _normalize_channel(image[ch])

    brain_mask = load_brain_mask(pid, args.sri24_dir)
    if brain_mask is not None:
        print(f"  Brain mask: {np.sum(brain_mask)} voxels ({100*np.mean(brain_mask):.1f}%)")
    else:
        print(f"  Warning: SRI24 T1G not found. NIfTI will not be brain-masked.")

    seg_path = Path(__file__).resolve().parents[2] / "TumorSynth_Outputs" / f"{pid}_tumor_mask.nii.gz"
    seg_vol = nib.load(str(seg_path)).get_fdata(dtype=np.float32) if seg_path.exists() else None
    if seg_vol is not None:
        print(f"  Loaded segmentation")

    biopsy_df = None
    biopsy_csv = args.dataset_dir / "biopsy_metadata.csv"
    if not biopsy_csv.exists():
        biopsy_csv = Path(__file__).resolve().parents[2] / "dataset_final" / "biopsy_metadata.csv"
    if biopsy_csv.exists():
        df = pd.read_csv(biopsy_csv)
        sub = df[df["mri_patient"] == pid]
        if not sub.empty:
            biopsy_df = sub.reset_index(drop=True)
            print(f"  Found {len(biopsy_df)} biopsy points (from {biopsy_csv})")
        else:
            print(f"  No biopsy rows for {pid} in {biopsy_csv}")
    else:
        print(f"  [skip biopsy] no biopsy_metadata.csv found")

    in_channels = image.shape[0]

    feature_size = args.feature_size
    if feature_size is None:
        summary_path = args.work_dir / "metrics_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            feature_size = summary.get("feature_size", FEATURE_SIZE)
            print(f"  feature_size={feature_size} (from metrics_summary.json)")
        else:
            feature_size = FEATURE_SIZE
            print(f"  feature_size={feature_size} (default)")

    fold_preds = []

    if args.pretrained_only:
        if not args.pretrained_ckpt.exists():
            sys.exit(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")
        print(f"  Loading raw BraTS pretrained weights from {args.pretrained_ckpt}")
        model = build_swinunetr(
            in_channels=in_channels,
            out_channels=1,
            spatial_dims=3,
            feature_size=feature_size,
            pretrained=True,
            pretrained_ckpt=args.pretrained_ckpt,
        ).to(device)
        fold_preds.append(sliding_window_inference(model, norm, PATCH_SIZE, args.overlap, device))
    else:
        for fold in folds:
            ckpt_path = args.work_dir / f"fold_{fold}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  [skip] {ckpt_path}")
                continue
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model = build_swinunetr(
                in_channels=in_channels,
                out_channels=1,
                spatial_dims=3,
                feature_size=feature_size,
                pretrained=False,
            ).to(device)
            model.load_state_dict(state["model_state"])
            print(f"  Fold {fold}: val_loss={state.get('val_loss', float('nan')):.5f}")
            fold_preds.append(sliding_window_inference(model, norm, PATCH_SIZE, args.overlap, device))

    if not fold_preds:
        sys.exit("No checkpoints found. Check --work-dir and --folds.")

    pred_vol = np.mean(fold_preds, axis=0)
    print(f"  Ensembled {len(fold_preds)} fold(s). Range: [{pred_vol.min():.3f}, {pred_vol.max():.3f}]")

    if args.pretrained_only:
        suffix = args.suffix or "swinunetr_pretrained"
    else:
        suffix = args.suffix or "swinunetr_finetuned"

    if not args.no_save_nifti:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        affine = nii.affine

        nib.save(nib.Nifti1Image(np.clip(pred_vol, 0, 1).astype(np.float32), affine),
                 str(args.out_dir / f"{pid}_purity_{suffix}.nii.gz"))

        uint8 = apply_brain_mask_uint8(pred_vol, brain_mask)
        nib.save(nib.Nifti1Image(uint8, affine),
                 str(args.out_dir / f"{pid}_purity_{suffix}_uint8.nii.gz"))

        print(f"  Saved: _purity_{suffix}.nii.gz (float [0,1]), _purity_{suffix}_uint8.nii.gz (brain-masked overlay)")

        if biopsy_df is not None and not biopsy_df.empty:
            needed = {"sri24_vox_x", "sri24_vox_y", "sri24_vox_z"}
            if needed.issubset(biopsy_df.columns):
                marker_vol = np.zeros(pred_vol.shape, dtype=np.int16)
                X, Y, Z = pred_vol.shape
                radius = 4  # voxels in XY plane (~4 mm at 1 mm isotropic)
                for b_idx, (_, row) in enumerate(biopsy_df.iterrows(), start=1):
                    cx = int(np.clip(round(row["sri24_vox_x"]), 0, X - 1))
                    cy = int(np.clip(round(row["sri24_vox_y"]), 0, Y - 1))
                    cz = int(np.clip(round(row["sri24_vox_z"]), 0, Z - 1))
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:  # 2D disk, no dz loop
                                xi = cx + dx; yi = cy + dy
                                if 0 <= xi < X and 0 <= yi < Y:
                                    marker_vol[xi, yi, cz] = b_idx
                nib.save(nib.Nifti1Image(marker_vol, affine),
                         str(args.out_dir / f"{pid}_biopsy_markers_{suffix}.nii.gz"))
                print(f"  Saved: _biopsy_markers_{suffix}.nii.gz ({len(biopsy_df)} biopsies, radius={radius} vox)")

    t1_idx = global_mods.index("T01") if "T01" in global_mods else 0
    visualize(pid, image[t1_idx], pred_vol, biopsy_df, seg_vol,
              args.out_dir, args.planes, args.n_slices, suffix)
    plot_biopsy_locations(pid, image[t1_idx], pred_vol, biopsy_df, args.out_dir, suffix)
    plot_biopsy_predictions(pid, pred_vol, biopsy_df, args.out_dir, suffix)
    print("Done.")


if __name__ == "__main__":
    main()
