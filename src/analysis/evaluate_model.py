from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_semisupervised_regression import (   
    UNet3D,
    _normalize_channel,
    load_metadata,
    build_annotation_mask,
    COMBINED_SUFFIX,
)
from train_semisupervised_regression_swinunetr import ( 
    build_swinunetr,
)

PROJECT_DIR   = Path(__file__).resolve().parents[2]
DEFAULT_SEG   = PROJECT_DIR / "TumorSynth_Outputs"
PATCH_SIZE    = (160, 160, 128)
BASE_CHANNELS = 32
DEPTH         = 4
ABSNEG_LABELS = [3, 4, 5, 6, 11, 12]



def find_best_fold(work_dir: Path, best_by: str = "biopsy_auc") -> Tuple[int, Path]:
    summary_path = work_dir / "metrics_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        folds = summary.get("folds", [])
        if folds:
            valid = [(f[best_by], f["fold_idx"]) for f in folds
                     if not math.isnan(f.get(best_by, float("nan")))]
            if valid:
                best_val, best_idx = max(valid)
                print(f"  Best fold by {best_by}: fold_{best_idx}  ({best_by}={best_val:.4f})")
                return best_idx, work_dir / f"fold_{best_idx}"

    fold_dirs = sorted(work_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {work_dir}")
    print(f"  No metrics_summary.json with fold data; using fold_{fold_dirs[0].name.split('_')[1]}")
    return int(fold_dirs[0].name.split("_")[1]), fold_dirs[0]


def find_checkpoint(fold_dir: Path) -> Path:
    matches = sorted(fold_dir.glob("best_model_attn-*.pt"))
    if not matches:
        raise FileNotFoundError(f"No best_model_attn-*.pt in {fold_dir}")
    return matches[0]


def detect_arch(sd: dict) -> Tuple[int, float, str]:
    """Return (in_channels, dropout, attention) from state dict."""
    in_channels = sd["enc_blocks.0.block.0.weight"].shape[1]
    dropout = 0.2 if any("block.4.weight" in k for k in sd) else 0.0
    attn_keys = [k for k in sd if k.startswith("bottleneck_attn")]
    if not attn_keys:
        attention = "none"
    elif any("channel_attn" in k for k in attn_keys):
        attention = "cbam"
    elif any("fc" in k for k in attn_keys):
        attention = "channel"
    else:
        attention = "spatial"
    return in_channels, dropout, attention


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state"]
    in_ch, dropout, attention = detect_arch(sd)
    print(f"  Checkpoint: in_channels={in_ch}  dropout={dropout}  attention={attention}")
    model = UNet3D(in_channels=in_ch, base_channels=BASE_CHANNELS, depth=DEPTH,
                   dropout=dropout, attention=attention).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model



def find_swinunetr_checkpoint(fold_dir: Path) -> Optional[Path]:
    p = fold_dir / "best_model.pt"
    return p if p.exists() else None


def detect_swinunetr_in_channels(sd: dict) -> int:
    """Infer input channels from SwinUNETR state dict."""
    if "adapter.weight" in sd:
        return sd["adapter.weight"].shape[1]
    # No adapter → direct 4-channel pretrained
    return 4


def load_swinunetr_model(ckpt_path: Path, device: torch.device,
                         feature_size: int = 48) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state"]
    in_ch = detect_swinunetr_in_channels(sd)
    print(f"  Checkpoint: in_channels={in_ch}  feature_size={feature_size}  (SwinUNETR fine-tuned)")
    model = build_swinunetr(in_channels=in_ch, feature_size=feature_size,
                            pretrained=False).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_swinunetr_pretrained_only(pretrained_ckpt: Path, in_channels: int,
                                   feature_size: int, device: torch.device) -> torch.nn.Module:
    print(f"  Pretrained-only SwinUNETR: in_channels={in_channels}  feature_size={feature_size}")
    model = build_swinunetr(in_channels=in_channels, feature_size=feature_size,
                            pretrained=True,
                            pretrained_ckpt=pretrained_ckpt).to(device)
    model.eval()
    return model



def _pad_to_patch(vol: np.ndarray, patch: tuple) -> Tuple[np.ndarray, list]:
    pads = [(0, 0)]
    for s, p in zip(vol.shape[1:], patch):
        diff = max(0, p - s)
        pads.append((diff // 2, diff - diff // 2))
    return np.pad(vol, pads, mode="constant"), pads


def sliding_window_inference(model, image: np.ndarray, patch_size: tuple,
                              overlap: float = 0.5,
                              device: torch.device = torch.device("cpu")) -> np.ndarray:
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



def load_patient_image(pid: str, dataset_dir: Path) -> np.ndarray:
    img_path = dataset_dir / "images" / f"{pid}.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    data = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    # Load all channels as-is; the model was trained on this exact image layout
    img = np.moveaxis(data, -1, 0).copy()
    for c in range(img.shape[0]):
        img[c] = _normalize_channel(img[c])
    return img


def infer_all_patients(model: torch.nn.Module, dataset_dir: Path,
                       pids: List[str],
                       device: torch.device) -> Dict[str, np.ndarray]:
    preds = {}
    for pid in pids:
        print(f"    Inference: {pid} ...", end=" ", flush=True)
        try:
            img = load_patient_image(pid, dataset_dir)
            vol = sliding_window_inference(model, img, PATCH_SIZE, overlap=0.5, device=device)
            preds[pid] = vol
            print("done")
        except FileNotFoundError as e:
            print(f"SKIPPED ({e})")
    return preds



def compute_metrics(gt: np.ndarray, pred: np.ndarray,
                    threshold: float) -> dict:
    mse  = float(np.mean((gt - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(gt - pred)))
    gt_bin   = (gt   >= threshold).astype(int)
    pred_bin = (pred >= threshold).astype(int)
    accuracy = float(np.mean(gt_bin == pred_bin))
    auc = float("nan")
    if np.unique(gt_bin).size >= 2:
        auc = float(roc_auc_score(gt_bin, pred))
    return {"n": len(gt), "mse": mse, "rmse": rmse, "mae": mae,
            "accuracy": accuracy, "auc": auc}


def eval_all_labeled(pred_vols: Dict[str, np.ndarray], dataset_dir: Path,
                     seg_dir: Path, pids: List[str],
                     combined_suffix: str, threshold: float) -> dict:
    all_pred, all_true = [], []
    for pid in pids:
        if pid not in pred_vols:
            continue
        lbl_path = dataset_dir / "labels" / f"{pid}{combined_suffix}.nii.gz"
        seg_path = seg_dir / f"{pid}_tumor_mask.nii.gz"
        if not lbl_path.exists():
            continue
        lbl_nii = nib.load(str(lbl_path))
        lbl_data = lbl_nii.get_fdata(dtype=np.float32)
        mask = build_annotation_mask(lbl_data, seg_path, ABSNEG_LABELS)
        pred = pred_vols[pid]
        if pred.shape != lbl_data.shape:
            print(f"    WARNING: {pid} pred shape {pred.shape} != label shape {lbl_data.shape}, skipping")
            continue
        idx = mask.ravel().astype(bool)
        all_pred.append(pred.ravel()[idx])
        all_true.append(lbl_data.ravel()[idx])

    if not all_pred:
        return {}
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred), threshold)


def extract_roi_mean(vol: np.ndarray, cx, cy, cz, vox_mm, roi_mm: float) -> float:
    half = [max(1, int(round((roi_mm / 2.0) / float(v)))) for v in vox_mm]
    x0, x1 = max(0, cx - half[0]), min(vol.shape[0], cx + half[0] + 1)
    y0, y1 = max(0, cy - half[1]), min(vol.shape[1], cy + half[1] + 1)
    z0, z1 = max(0, cz - half[2]), min(vol.shape[2], cz + half[2] + 1)
    roi = vol[x0:x1, y0:y1, z0:z1]
    return float(roi.mean()) if roi.size > 0 else 0.0


def eval_biopsy_points(pred_vols: Dict[str, np.ndarray], dataset_dir: Path,
                       pids: List[str], threshold: float,
                       roi_mm: float = 10.0) -> Tuple[dict, pd.DataFrame]:
    bcsv = dataset_dir / "biopsy_metadata.csv"
    if not bcsv.exists():
        print("  biopsy_metadata.csv not found — skipping biopsy-point eval")
        return {}, pd.DataFrame()

    bdf = pd.read_csv(bcsv)
    rows = []
    for pid in pids:
        if pid not in pred_vols:
            continue
        pred = pred_vols[pid]
        lbl_path = dataset_dir / "labels" / f"{pid}_biopsy_purity.nii.gz"
        if not lbl_path.exists():
            # Fallback: get voxel size from any label file
            candidates = sorted((dataset_dir / "labels").glob(f"{pid}*.nii.gz"))
            if not candidates:
                continue
            lbl_path = candidates[0]
        vox_mm = nib.load(str(lbl_path)).header.get_zooms()[:3]
        dfp = bdf[bdf["mri_patient"] == pid]
        for _, row in dfp.iterrows():
            xi = int(row["sri24_vox_x"])
            yi = int(row["sri24_vox_y"])
            zi = int(row["sri24_vox_z"])
            if not (0 <= xi < pred.shape[0] and 0 <= yi < pred.shape[1] and 0 <= zi < pred.shape[2]):
                continue
            rows.append({
                "mri_patient": pid,
                "biopsy_id":   row.get("biopsy_id", ""),
                "PAMES_purity": float(row["PAMES_purity"]),
                "pred_roi_mean": extract_roi_mean(pred, xi, yi, zi, vox_mm, roi_mm),
            })

    if not rows:
        return {}, pd.DataFrame()

    df = pd.DataFrame(rows)
    metrics = compute_metrics(df["PAMES_purity"].values, df["pred_roi_mean"].values, threshold)
    return metrics, df



def infer_dataset_dir(work_dir: Path) -> Optional[Path]:

    parts = work_dir.parts
    for i, p in enumerate(parts):
        if p.startswith("mod_") and i + 1 < len(parts) and parts[i + 1].startswith("attn_"):
            work_root = Path(*parts[:i])
            mod_label = p
            candidate = work_root / "datasets" / mod_label
            if candidate.exists() and (candidate / "images").exists():
                return candidate
    # Fallback: project dataset_final
    default = PROJECT_DIR / "dataset_final"
    if default.exists():
        return default
    return None


def _fmt(m: dict) -> str:
    if not m:
        return "  N/A"
    auc_s = f"{m['auc']:.4f}" if not math.isnan(m["auc"]) else "  N/A "
    return (f"  n={m['n']:5d}  MSE={m['mse']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  Acc={m['accuracy']:.4f}  AUC={auc_s}")


def print_results(label: str, metrics_labeled: dict, metrics_biopsy: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  All annotated voxels :{_fmt(metrics_labeled)}")
    print(f"  Biopsy points only   :{_fmt(metrics_biopsy)}")


# ─── Single-run evaluation ────────────────────────────────────────────────────

def _resolve_dataset_and_check(dataset_dir: Optional[Path], work_dir: Path,
                                ckpt_path: Path, model_type: str) -> Optional[Path]:
    if dataset_dir is None:
        dataset_dir = infer_dataset_dir(work_dir)
    if dataset_dir is None or not dataset_dir.exists():
        print(f"  ERROR: cannot find dataset dir. Pass --dataset-dir explicitly.")
        return None
    print(f"  Dataset: {dataset_dir}")

    sample_imgs = sorted((dataset_dir / "images").glob("P*.nii.gz"))
    if not sample_imgs:
        print(f"  ERROR: no images found in {dataset_dir / 'images'}")
        return None

    n_img_ch = nib.load(str(sample_imgs[0])).shape[3]
    sd = torch.load(ckpt_path, map_location="cpu")["model_state"]
    if model_type == "swinunetr":
        n_mod_ch = detect_swinunetr_in_channels(sd)
    else:
        n_mod_ch = sd["enc_blocks.0.block.0.weight"].shape[1]

    if n_img_ch != n_mod_ch:
        print(f"  ERROR: model expects {n_mod_ch} channels but dataset has {n_img_ch}.")
        print(f"  Rebuild the dataset with the correct modalities and pass --dataset-dir.")
        return None
    return dataset_dir


def _run_eval(model, dataset_dir: Path, seg_dir: Path, work_dir: Path,
              fold_idx, args, label: str) -> Tuple[dict, dict]:
    _, patients_info = load_metadata(dataset_dir)
    pids = [pid for pid, info in patients_info.items() if info.get("has_labels", False)]
    print(f"  Fully-labeled patients ({len(pids)}): {pids}")

    device = next(model.parameters()).device
    print("  Running inference...")
    pred_vols = infer_all_patients(model, dataset_dir, pids, device)

    combined_suffix = COMBINED_SUFFIX
    summary_path = work_dir / "metrics_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            combined_suffix = json.load(f).get("combined_suffix", COMBINED_SUFFIX)

    print("  Computing metrics...")
    m_labeled = eval_all_labeled(pred_vols, dataset_dir, seg_dir, pids,
                                 combined_suffix, args.auc_threshold)
    m_biopsy, biopsy_df = eval_biopsy_points(pred_vols, dataset_dir, pids,
                                              args.auc_threshold, args.biopsy_roi_mm)
    print_results(label, m_labeled, m_biopsy)

    if not biopsy_df.empty:
        out_csv = work_dir / "eval_biopsy_predictions.csv"
        biopsy_df.to_csv(out_csv, index=False)
        print(f"  Biopsy predictions saved: {out_csv}")

    return m_labeled, m_biopsy


def evaluate_pretrained_only(dataset_dir: Optional[Path], seg_dir: Path,
                              args) -> Tuple[dict, dict]:
    print(f"\n--- Evaluating: SwinUNETR pretrained-only ---")
    if dataset_dir is None:
        dataset_dir = PROJECT_DIR / "dataset_final"
    if not dataset_dir.exists():
        print(f"  ERROR: dataset not found: {dataset_dir}")
        return {}, {}
    print(f"  Dataset: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_imgs = sorted((dataset_dir / "images").glob("P*.nii.gz"))
    in_ch = nib.load(str(sample_imgs[0])).shape[3] if sample_imgs else 4
    pretrained_ckpt = Path(args.pretrained_ckpt) if args.pretrained_ckpt else None
    model = load_swinunetr_pretrained_only(pretrained_ckpt, in_ch,
                                           args.feature_size, device)
    work_dir = PROJECT_DIR / "swinunetr_pretrained_only_eval"
    work_dir.mkdir(exist_ok=True)
    return _run_eval(model, dataset_dir, seg_dir, work_dir, "pretrained",
                     args, "SwinUNETR pretrained-only (no fine-tuning)")


def evaluate_run(work_dir: Path, dataset_dir: Optional[Path],
                 seg_dir: Path, args) -> Tuple[dict, dict]:
    work_dir = work_dir.resolve()
    print(f"\n--- Evaluating: {work_dir} ---")

    model_type = getattr(args, "model_type", "unet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find best fold and load checkpoint
    fold_idx, fold_dir = find_best_fold(work_dir, args.best_by)

    if model_type == "swinunetr":
        ckpt_path = find_swinunetr_checkpoint(fold_dir)
        if ckpt_path is None:
            print(f"  ERROR: best_model.pt not found in {fold_dir}")
            return {}, {}
        print(f"  Checkpoint: {ckpt_path.name}")
        dataset_dir = _resolve_dataset_and_check(dataset_dir, work_dir, ckpt_path, "swinunetr")
        if dataset_dir is None:
            return {}, {}
        model = load_swinunetr_model(ckpt_path, device, args.feature_size)
    else:
        ckpt_path = find_checkpoint(fold_dir)
        if ckpt_path is None:
            print(f"  ERROR: no checkpoint found in {fold_dir}")
            return {}, {}
        print(f"  Checkpoint: {ckpt_path.name}")
        dataset_dir = _resolve_dataset_and_check(dataset_dir, work_dir, ckpt_path, "unet")
        if dataset_dir is None:
            return {}, {}
        model = load_model(ckpt_path, device)

    label = f"{work_dir.parent.name}/{work_dir.name}  (fold {fold_idx})"
    return _run_eval(model, dataset_dir, seg_dir, work_dir, fold_idx, args, label)



def scan_and_compare(scan_root: Path, seg_dir: Path, args) -> None:
    run_dirs = sorted(scan_root.glob("*/attn_*/loss_*"))
    if not run_dirs:
        # Try shallower structure (e.g. swinunetr: mod_X directly)
        run_dirs = sorted(scan_root.glob("mod_*"))
    print(f"\nFound {len(run_dirs)} run(s) under {scan_root}")

    rows = []
    for rd in run_dirs:
        if not (rd / "metrics_summary.json").exists() and not list(rd.glob("fold_*")):
            continue
        m_l, m_b = evaluate_run(rd, None, seg_dir, args)
        rows.append({
            "run":          str(rd.relative_to(scan_root)),
            "labeled_n":    m_l.get("n",        ""),
            "labeled_mse":  m_l.get("mse",      float("nan")),
            "labeled_rmse": m_l.get("rmse",     float("nan")),
            "labeled_mae":  m_l.get("mae",       float("nan")),
            "labeled_acc":  m_l.get("accuracy",  float("nan")),
            "labeled_auc":  m_l.get("auc",       float("nan")),
            "biopsy_n":     m_b.get("n",         ""),
            "biopsy_mse":   m_b.get("mse",       float("nan")),
            "biopsy_rmse":  m_b.get("rmse",      float("nan")),
            "biopsy_mae":   m_b.get("mae",        float("nan")),
            "biopsy_acc":   m_b.get("accuracy",   float("nan")),
            "biopsy_auc":   m_b.get("auc",        float("nan")),
        })

    if not rows:
        print("No completed runs found.")
        return

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("biopsy_auc", ascending=False)
    out_csv = scan_root / "eval_comparison.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (sorted by biopsy AUC)")
    print(f"{'='*80}")
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out_csv}")



def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained U-Net on labelled + biopsy points")
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--work-dir",   type=Path,
                     help="Path to a single run directory (contains fold_N/ subdirs)")
    grp.add_argument("--scan-root",  type=Path,
                     help="Scan all run dirs under this root and print comparison table")
    p.add_argument("--dataset-dir",  type=Path, default=None,
                   help="Dataset directory (auto-detected if omitted)")
    p.add_argument("--seg-dir",      type=Path, default=DEFAULT_SEG,
                   help="TumorSynth segmentation directory")
    p.add_argument("--auc-threshold", type=float, default=0.7,
                   help="Purity threshold for binary accuracy/AUC (default: 0.7)")
    p.add_argument("--biopsy-roi-mm", type=float, default=10.0,
                   help="ROI sphere radius (mm) for biopsy-point evaluation (default: 10.0)")
    p.add_argument("--best-by",      type=str, default="biopsy_auc",
                   choices=["biopsy_auc", "biopsy_rmse", "best_val_auc", "best_val_rmse"],
                   help="Metric used to pick the best fold (default: biopsy_auc)")
    # Model type
    p.add_argument("--model-type",   type=str, default="unet",
                   choices=["unet", "swinunetr"],
                   help="Model architecture (default: unet)")
    # SwinUNETR options
    p.add_argument("--pretrained-only", action="store_true",
                   help="Evaluate raw pretrained SwinUNETR without any fine-tuning")
    p.add_argument("--pretrained-ckpt", type=Path, default=None,
                   help="Path to pretrained SwinUNETR weights (e.g. pretrained_weights/swinunetr_v1_ssl.pt)")
    p.add_argument("--feature-size",  type=int, default=48,
                   help="SwinUNETR feature size (default: 48)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.scan_root:
        scan_and_compare(args.scan_root, args.seg_dir, args)
    elif getattr(args, "pretrained_only", False):
        evaluate_pretrained_only(args.dataset_dir, args.seg_dir, args)
    else:
        evaluate_run(args.work_dir, args.dataset_dir, args.seg_dir, args)


if __name__ == "__main__":
    main()
