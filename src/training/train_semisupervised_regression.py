from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.data import DataLoader, Dataset

from monai.transforms import Compose, EnsureTyped, RandAffined, RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd

DEFAULT_BASE    = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DEFAULT_BASE / "dataset_final"
DEFAULT_SEG     = DEFAULT_BASE / "TumorSynth_Outputs"
DEFAULT_WORK    = DEFAULT_BASE / "semisup_regression_work"

COMBINED_SUFFIX = "_biopsy_purity_absneg_innertumor_L3-4-11-12_enhancing"



class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        mid_ch = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + max)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max], dim=1)
        return x * self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        layers += [
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4, dropout: float = 0.0, attention: str = "none"):
        super().__init__()
        self.attention = attention
        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.enc_blocks.append(DoubleConv(prev, ch, dropout))
            self.pool_layers.append(nn.MaxPool3d(2))
            prev = ch

        self.bottleneck = DoubleConv(prev, prev * 2, dropout)
        btn = prev * 2

        if attention == "channel":
            self.bottleneck_attn = ChannelAttention(btn)
        elif attention == "spatial":
            self.bottleneck_attn = SpatialAttention()
        elif attention == "cbam":
            self.bottleneck_attn = CBAM(btn)
        else:
            self.bottleneck_attn = None

        self.up_layers  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attns  = nn.ModuleList()
        dec_in = btn
        for ch in reversed(chs):
            self.up_layers.append(nn.ConvTranspose3d(dec_in, ch, 2, stride=2))
            self.dec_blocks.append(DoubleConv(dec_in, ch, dropout))
            # Decoder attention
            if attention == "channel":
                self.dec_attns.append(ChannelAttention(ch))
            elif attention == "spatial":
                self.dec_attns.append(SpatialAttention())
            elif attention == "cbam":
                self.dec_attns.append(CBAM(ch))
            else:
                self.dec_attns.append(None)
            dec_in = ch

        self.final_conv = nn.Conv3d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc, pool in zip(self.enc_blocks, self.pool_layers):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x)
        for up, dec, attn, skip in zip(self.up_layers, self.dec_blocks, self.dec_attns, reversed(skips)):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                diff = [s - xs for s, xs in zip(skip.shape[-3:], x.shape[-3:])]
                pad = []
                for d in reversed(diff):
                    pad += [d // 2, d - d // 2]
                x = F.pad(x, pad)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
            if attn is not None:
                x = attn(x)
        return torch.sigmoid(self.final_conv(x))

def load_metadata(dataset_dir: Path) -> Tuple[List[str], Dict[str, dict]]:
    with open(dataset_dir / "dataset_info.json") as f:
        info = json.load(f)
    return info["channel_order"], info["patients"]


def _load_image(path: Path, global_mods: List[str], patient_mods: List[str]) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"{path}: expected 4D image, got {data.shape}")
    n_ch = len(global_mods)
    if data.shape[3] == n_ch:
        out = data
    else:
        out = np.zeros((*data.shape[:3], n_ch), dtype=np.float32)
        for i, mod in enumerate(patient_mods):
            out[..., global_mods.index(mod)] = data[..., i]
    return np.moveaxis(out, -1, 0)  # (C, X, Y, Z)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    mask = channel != 0
    values = channel[mask] if np.any(mask) else channel.reshape(-1)

    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    out = channel.astype(np.float32, copy=True)
    if np.any(mask):
        out[mask] = (out[mask] - mean) / std
    else:
        out = (out - mean) / std
    return out.astype(np.float32, copy=False)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    processed = np.empty_like(image, dtype=np.float32)
    for ch in range(image.shape[0]):
        processed[ch] = _normalize_channel(image[ch])
    return processed


def _crop_slices_from_center(
    center: np.ndarray,
    spatial: Tuple[int, int, int],
    patch: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    slices = []
    for axis, (size, patch_len) in enumerate(zip(spatial, patch)):
        start = int(center[axis]) - patch_len // 2
        start = max(0, min(start, size - patch_len))
        slices.append(slice(start, start + patch_len))
    return tuple(slices)


def _random_or_foreground_crop(
    mask: np.ndarray,
    spatial: Tuple[int, int, int],
    patch: Tuple[int, int, int],
    foreground_sample_prob: float,
) -> Tuple[slice, slice, slice]:
    if np.any(mask > 0) and np.random.rand() < foreground_sample_prob:
        coords = np.argwhere(mask[0] > 0)
        if coords.size > 0:
            center = coords[np.random.randint(0, len(coords))]
            return _crop_slices_from_center(center, spatial, patch)
    return _random_crop_indices(spatial, patch)


def build_training_transforms() -> Compose:
    keys = ["image", "target", "ann_mask", "wmap"]
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandAffined(
            keys=keys,
            prob=0.25,
            rotate_range=(0.10, 0.10, 0.10),
            scale_range=(0.10, 0.10, 0.10),
            mode=("bilinear", "bilinear", "nearest", "nearest"),
            padding_mode="border",
            cache_grid=True,
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=["image"], prob=0.20, factors=0.10),
        RandShiftIntensityd(keys=["image"], prob=0.20, offsets=0.10),
        EnsureTyped(keys=keys, dtype=torch.float32),
    ])


def build_validation_transforms() -> Compose:
    return Compose([
        EnsureTyped(keys=["image", "target", "ann_mask", "wmap"], dtype=torch.float32),
    ])


def build_annotation_mask(
    combined_label: np.ndarray,
    seg_path: Optional[Path],
    absneg_labels: List[int],
) -> np.ndarray:
    mask = combined_label != 0  
    if seg_path is not None and seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int32)
        if seg.shape == combined_label.shape:
            mask = mask | np.isin(seg, absneg_labels)
        else:
            print(f"WARNING: seg shape {seg.shape} != label shape {combined_label.shape}; skipping seg mask for this patient.")
    else:
        print(f"WARNING: segmentation not found at {seg_path}; absneg voxels excluded from mask.")
    return mask.astype(np.float32)  # (X, Y, Z)


class PatientRecord:
    __slots__ = ("pid", "image", "target", "ann_mask", "biopsy_weight_map", "has_biopsy")

    def __init__(
        self,
        pid: str,
        image: np.ndarray,         # (C, X, Y, Z)
        target: np.ndarray,        # (1, X, Y, Z)
        ann_mask: np.ndarray,      # (1, X, Y, Z) float32 0/1
        biopsy_weight_map: np.ndarray,  # (1, X, Y, Z)
        has_biopsy: bool,
    ):
        self.pid = pid
        self.image = image
        self.target = target
        self.ann_mask = ann_mask
        self.biopsy_weight_map = biopsy_weight_map
        self.has_biopsy = has_biopsy


def load_all_patients(
    dataset_dir: Path,
    seg_dir: Path,
    global_mods: List[str],
    patients_info: Dict[str, dict],
    absneg_labels: List[int],
    biopsy_roi_mm: float,
    biopsy_weight: float,
    combined_suffix: str,
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    bcsv = dataset_dir / "biopsy_metadata.csv"
    biopsy_df = pd.read_csv(bcsv) if bcsv.exists() else None

    fully_labeled:   List[PatientRecord] = []
    partially_labeled: List[PatientRecord] = []

    for pid, pinfo in patients_info.items():
        img_path  = images_dir / f"{pid}.nii.gz"
        lbl_path  = labels_dir / f"{pid}{combined_suffix}.nii.gz"
        seg_path  = seg_dir / f"{pid}_tumor_mask.nii.gz"

        if not img_path.exists():
            print(f"WARNING: image not found for {pid}, skipping.")
            continue
        if not lbl_path.exists():
            print(f"WARNING: combined label not found for {pid}, skipping.")
            continue

        image  = _load_image(img_path, global_mods, pinfo["available_modalities"])
        image = preprocess_image(image)
        lbl_nii = nib.load(str(lbl_path))
        lbl_data = np.clip(lbl_nii.get_fdata(dtype=np.float32), 0.0, 1.0)  # (X, Y, Z)

        ann_mask = build_annotation_mask(lbl_data, seg_path, absneg_labels)

        target = lbl_data[None]      # (1, X, Y, Z)
        ann_mask = ann_mask[None]    # (1, X, Y, Z)

        # Biopsy-ROI weight map (only meaningful for fully-labeled patients)
        has_biopsy = bool(pinfo.get("has_labels", False))
        wmap = _build_weight_map(
            pid, target, biopsy_df, lbl_nii, biopsy_roi_mm, biopsy_weight
        ) if (has_biopsy and biopsy_weight > 1.0 and biopsy_df is not None) else np.ones_like(target)

        rec = PatientRecord(
            pid=pid, image=image, target=target,
            ann_mask=ann_mask, biopsy_weight_map=wmap, has_biopsy=has_biopsy,
        )
        if has_biopsy:
            fully_labeled.append(rec)
        else:
            partially_labeled.append(rec)

    return fully_labeled, partially_labeled


def _build_weight_map(
    pid: str,
    target: np.ndarray,
    biopsy_df: Optional[pd.DataFrame],
    lbl_nii: nib.Nifti1Image,
    roi_mm: float,
    weight: float,
) -> np.ndarray:
    w = np.ones_like(target, dtype=np.float32)
    if biopsy_df is None:
        return w
    vox = lbl_nii.header.get_zooms()[:3]
    half = [max(1, int(round((roi_mm / 2.0) / float(v)))) for v in vox]
    dfp = biopsy_df[biopsy_df["mri_patient"] == pid]
    shape = target.shape[1:]
    for _, row in dfp.iterrows():
        xi, yi, zi = int(row["sri24_vox_x"]), int(row["sri24_vox_y"]), int(row["sri24_vox_z"])
        if not (0 <= xi < shape[0] and 0 <= yi < shape[1] and 0 <= zi < shape[2]):
            continue
        x0, x1 = max(0, xi - half[0]), min(shape[0], xi + half[0] + 1)
        y0, y1 = max(0, yi - half[1]), min(shape[1], yi + half[1] + 1)
        z0, z1 = max(0, zi - half[2]), min(shape[2], zi + half[2] + 1)
        w[0, x0:x1, y0:y1, z0:z1] = np.maximum(w[0, x0:x1, y0:y1, z0:z1], float(weight))
    return w

def _random_crop_indices(
    shape: Tuple[int, int, int], patch: Tuple[int, int, int]
) -> Tuple[slice, slice, slice]:
    slices = []
    for s, p in zip(shape, patch):
        start = np.random.randint(0, max(1, s - p + 1))
        slices.append(slice(start, start + p))
    return tuple(slices)


class SupervisedDataset(Dataset):

    def __init__(
        self,
        records: List[PatientRecord],
        patch: Optional[Tuple[int, int, int]],
        foreground_sample_prob: float = 0.33,
        transform: Optional[Compose] = None,
    ):
        self.records = records
        self.patch   = patch
        self.foreground_sample_prob = foreground_sample_prob
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = r.image
        tgt = r.target
        msk = r.ann_mask
        wt  = r.biopsy_weight_map

        if self.patch is not None:
            spatial = img.shape[1:]
            if all(p <= s for p, s in zip(self.patch, spatial)):
                sl = _random_or_foreground_crop(
                    msk,
                    spatial,
                    self.patch,
                    self.foreground_sample_prob,
                )
                img = img[(slice(None), *sl)]
                tgt = tgt[(slice(None), *sl)]
                msk = msk[(slice(None), *sl)]
                wt  = wt[ (slice(None), *sl)]

        sample = {"image": img, "target": tgt, "ann_mask": msk, "wmap": wt}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample["image"], sample["target"], sample["ann_mask"], sample["wmap"]


class UnlabeledDataset(Dataset):

    def __init__(
        self,
        records: List[PatientRecord],
        patch: Optional[Tuple[int, int, int]],
        foreground_sample_prob: float = 0.33,
    ):
        self.records = records
        self.patch   = patch
        self.foreground_sample_prob = foreground_sample_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = r.image
        if self.patch is not None:
            spatial = img.shape[1:]
            if all(p <= s for p, s in zip(self.patch, spatial)):
                sl = _random_or_foreground_crop(
                    r.ann_mask,
                    spatial,
                    self.patch,
                    self.foreground_sample_prob,
                )
                img = img[(slice(None), *sl)]
        return torch.from_numpy(img.copy())

def masked_regression_loss(
    preds: torch.Tensor,   # (B, 1, X, Y, Z)
    targets: torch.Tensor, # (B, 1, X, Y, Z)
    ann_mask: torch.Tensor,# (B, 1, X, Y, Z)  0/1
    wmap: torch.Tensor,    # (B, 1, X, Y, Z)
    loss_type: str,
) -> torch.Tensor:

    n = ann_mask.sum()
    if n == 0:
        return preds.sum() * 0.0

    diff = preds - targets

    if loss_type == "l1":
        elem = F.l1_loss(preds, targets, reduction="none")
    elif loss_type in ("l2", "mse"):
        elem = F.mse_loss(preds, targets, reduction="none")
    elif loss_type == "smoothl1":
        elem = F.smooth_l1_loss(preds, targets, beta=0.1, reduction="none")
    elif loss_type == "huber":
        elem = F.huber_loss(preds, targets, delta=0.1, reduction="none")
    elif loss_type == "logcosh":
        elem = diff + torch.nn.functional.softplus(-2 * diff) - math.log(2.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return (elem * ann_mask * wmap).sum() / (n + 1e-8)


def consistency_loss(
    model: nn.Module,
    imgs: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    p1 = model(imgs + torch.randn_like(imgs) * noise_std)
    with torch.no_grad():
        p2 = model(imgs + torch.randn_like(imgs) * noise_std)
    return F.mse_loss(p1, p2)


def rampup(epoch: int, n_epochs: int) -> float:
    return min(1.0, epoch / n_epochs) if n_epochs > 0 else 1.0


def extract_roi_mean(vol: np.ndarray, cx, cy, cz, vox_mm, roi_mm) -> float:
    half = [max(1, int(round((roi_mm / 2.0) / float(v)))) for v in vox_mm]
    x0, x1 = max(0, cx - half[0]), min(vol.shape[0], cx + half[0] + 1)
    y0, y1 = max(0, cy - half[1]), min(vol.shape[1], cy + half[1] + 1)
    z0, z1 = max(0, cz - half[2]), min(vol.shape[2], cz + half[2] + 1)
    roi = vol[x0:x1, y0:y1, z0:z1]
    return float(roi.mean()) if roi.size > 0 else 0.0


def train_fold(
    fold_idx: int,
    train_records: List[PatientRecord],
    val_records: List[PatientRecord],
    partial_records: List[PatientRecord],
    dataset_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(args.seed + fold_idx)
    np.random.seed(args.seed + fold_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = tuple(args.patch_size) if args.patch_size else None

    all_train = train_records + partial_records

    train_transform = build_training_transforms()
    val_transform = build_validation_transforms()

    ds_train = SupervisedDataset(
        all_train,
        patch,
        foreground_sample_prob=args.foreground_sample_prob,
        transform=train_transform,
    )
    ds_val   = SupervisedDataset(val_records, patch=None, transform=val_transform)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=1,               shuffle=False, num_workers=0)

    # Consistency loader: partially-labeled patients (no biopsy GT)
    use_cons = args.lambda_consistency > 0.0 and len(partial_records) > 0
    dl_cons  = None
    if use_cons:
        dl_cons = DataLoader(
            UnlabeledDataset(
                partial_records,
                patch,
                foreground_sample_prob=args.foreground_sample_prob,
            ),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )

    model = UNet3D(
        in_channels=all_train[0].image.shape[0],
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
        attention=args.attention,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_dir = work_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_auc  = float("nan")
    best_val_rmse = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = args.lambda_consistency * rampup(epoch, args.consistency_rampup)
        cons_iter = iter(dl_cons) if dl_cons is not None else None

        sum_sup  = 0.0
        sum_cons = 0.0

        for imgs, tgts, masks, wmaps in dl_train:
            imgs  = imgs.to(device)
            tgts  = tgts.to(device)
            masks = masks.to(device)
            wmaps = wmaps.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            sup = masked_regression_loss(preds, tgts, masks, wmaps, args.loss)

            cons = torch.tensor(0.0, device=device)
            if cons_iter is not None and lam > 0:
                try:
                    u_imgs = next(cons_iter).to(device)
                except StopIteration:
                    cons_iter = iter(dl_cons)
                    u_imgs = next(cons_iter).to(device)
                cons = consistency_loss(model, u_imgs, args.consistency_noise)

            loss = sup + lam * cons
            loss.backward()
            optimizer.step()

            sum_sup  += sup.item()  * imgs.size(0)
            sum_cons += cons.item() * imgs.size(0)

        n = max(len(ds_train), 1)
        sum_sup /= n; sum_cons /= n

        model.eval()
        val_loss  = 0.0
        all_pred: List[np.ndarray] = []
        all_true: List[np.ndarray] = []

        with torch.no_grad():
            for imgs, tgts, masks, wmaps in dl_val:
                imgs  = imgs.to(device)
                tgts  = tgts.to(device)
                masks = masks.to(device)
                wmaps = wmaps.to(device)
                preds = model(imgs)
                vl = masked_regression_loss(preds, tgts, masks, wmaps, args.loss)
                val_loss += vl.item() * imgs.size(0)

                biopsy_sel = (masks > 0).cpu().numpy().ravel().astype(bool)
                p_np = preds.cpu().numpy().ravel()
                t_np = tgts.cpu().numpy().ravel()
                if biopsy_sel.any():
                    all_pred.append(p_np[biopsy_sel])
                    all_true.append(t_np[biopsy_sel])

        val_loss /= max(len(ds_val), 1)
        val_rmse = val_auc = float("nan")
        if all_pred:
            yp = np.concatenate(all_pred)
            yl = np.concatenate(all_true)
            val_rmse = float(np.sqrt(np.mean((yl - yp) ** 2)))
            yb = (yl >= args.auc_threshold).astype(int)
            if np.unique(yb).size >= 2:
                val_auc = float(roc_auc_score(yb, yp))

        print(
            f"Fold {fold_idx} | Ep {epoch:3d}/{args.epochs} | "
            f"sup={sum_sup:.4f} cons={sum_cons:.4f}(λ={lam:.3f}) | "
            f"val={val_loss:.4f} RMSE={val_rmse:.4f} AUC={val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc  = val_auc
            best_val_rmse = val_rmse
            ckpt_name = f"best_model_attn-{args.attention}.pt"
            torch.save({"model_state": model.state_dict(), "val_loss": val_loss}, fold_dir / ckpt_name)

    print(f"Fold {fold_idx} done. Best val={best_val_loss:.4f} RMSE={best_val_rmse:.4f} AUC={best_val_auc:.4f}")

    biopsy_mae = biopsy_rmse = biopsy_auc = float("nan")
    bcsv = dataset_dir / "biopsy_metadata.csv"
    ckpt_name = f"best_model_attn-{args.attention}.pt"
    ckpt_path = fold_dir / ckpt_name
    if bcsv.exists() and ckpt_path.exists():
        bdf = pd.read_csv(bcsv)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        rows = []
        for rec in val_records:
            with torch.no_grad():
                pred_vol = model(torch.from_numpy(rec.image[None]).to(device)).cpu().numpy()[0, 0]
            lp = dataset_dir / "labels" / f"{rec.pid}_biopsy_purity.nii.gz"
            if not lp.exists():
                continue
            vox_mm = nib.load(str(lp)).header.get_zooms()[:3]
            for _, row in bdf[bdf["mri_patient"] == rec.pid].iterrows():
                xi, yi, zi = int(row["sri24_vox_x"]), int(row["sri24_vox_y"]), int(row["sri24_vox_z"])
                if not (0 <= xi < pred_vol.shape[0] and 0 <= yi < pred_vol.shape[1] and 0 <= zi < pred_vol.shape[2]):
                    continue
                rows.append({
                    "mri_patient": rec.pid,
                    "PAMES_purity": float(row["PAMES_purity"]),
                    "pred_roi_mean": extract_roi_mean(pred_vol, xi, yi, zi, vox_mm, args.biopsy_roi_mm),
                })
        if rows:
            bx = pd.DataFrame(rows)
            gt, pr = bx["PAMES_purity"].values, bx["pred_roi_mean"].values
            biopsy_mae  = float(np.mean(np.abs(gt - pr)))
            biopsy_rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
            gb = (gt >= args.auc_threshold).astype(int)
            if np.unique(gb).size >= 2:
                biopsy_auc = float(roc_auc_score(gb, pr))
            print(f"Fold {fold_idx} biopsy | n={len(bx)} MAE={biopsy_mae:.4f} RMSE={biopsy_rmse:.4f} AUC={biopsy_auc:.4f}")
            bx.to_csv(fold_dir / "biopsy_predictions.csv", index=False)

    return {
        "fold_idx": fold_idx,
        "best_val_mae": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "best_val_auc": best_val_auc,
        "biopsy_mae": biopsy_mae,
        "biopsy_rmse": biopsy_rmse,
        "biopsy_auc": biopsy_auc,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semi-supervised 3D U-Net purity regression (dataset_final)")
    p.add_argument("--dataset-dir",    type=Path, default=DEFAULT_DATASET)
    p.add_argument("--seg-dir",        type=Path, default=DEFAULT_SEG,
                   help="Directory containing PXX_tumor_mask.nii.gz files (TumorSynth_Outputs)")
    p.add_argument("--work-dir",       type=Path, default=DEFAULT_WORK)
    p.add_argument("--absneg-labels",  type=int, nargs="+", default=[3, 4, 5, 6, 11, 12],
                   help="Segmentation label IDs treated as absolute negatives (purity=0)")
    p.add_argument("--combined-suffix", type=str,
                   default=COMBINED_SUFFIX,
                   help="Suffix for the combined label file between PID and .nii.gz")
    # Training
    p.add_argument("--n-folds",        type=int,   default=3)
    p.add_argument("--leave-one-out",  action="store_true",
                   help="Use leave-one-out cross-validation instead of K-fold")
    p.add_argument("--start-fold",     type=int,   default=0,
                   help="Skip folds before this index (resume interrupted run)")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=1)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--base-channels",  type=int,   default=32)
    p.add_argument("--depth",          type=int,   default=4)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--patch-size",     type=int,   nargs=3, default=[160, 160, 128])
    p.add_argument("--foreground-sample-prob", type=float, default=0.33,
                   help="Probability of sampling a patch around an annotated voxel instead of uniformly at random")
    p.add_argument("--attention",      choices=["none", "channel", "spatial", "cbam"], default="none")
    p.add_argument("--loss",           choices=["l1", "l2", "mse", "smoothl1", "huber", "logcosh"], default="smoothl1")
    p.add_argument("--auc-threshold",  type=float, default=0.7)
    p.add_argument("--biopsy-weight",  type=float, default=5.0,
                   help="Extra weight on biopsy-ROI voxels in supervised loss (1=off)")
    p.add_argument("--biopsy-roi-mm",  type=float, default=10.0)
    # Semi-supervised
    p.add_argument("--lambda-consistency", type=float, default=0.1,
                   help="Weight of consistency loss on partially-labeled patients (0=off)")
    p.add_argument("--consistency-rampup", type=int,   default=15,
                   help="Epochs to linearly ramp lambda from 0 to --lambda-consistency")
    p.add_argument("--consistency-noise",  type=float, default=0.05,
                   help="Gaussian noise std for consistency augmentation")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    global_mods, patients_info = load_metadata(args.dataset_dir)

    fully_labeled, partially_labeled = load_all_patients(
        dataset_dir=args.dataset_dir,
        seg_dir=args.seg_dir,
        global_mods=global_mods,
        patients_info=patients_info,
        absneg_labels=args.absneg_labels,
        biopsy_roi_mm=args.biopsy_roi_mm,
        biopsy_weight=args.biopsy_weight,
        combined_suffix=args.combined_suffix,
    )

    fully_ids = [r.pid for r in fully_labeled]
    partial_ids = [r.pid for r in partially_labeled]

    print(f"\nFully labeled  ({len(fully_labeled)}): {fully_ids}")
    print(f"Partially labeled ({len(partially_labeled)}): {partial_ids}")
    print(f"Modalities: {global_mods}")
    print(f"Annotation sources: biopsy + absneg(seg labels {args.absneg_labels}) + innertumor")
    print(f"Loss: masked {args.loss} | biopsy weight={args.biopsy_weight} roi={args.biopsy_roi_mm}mm")
    print(f"Attention: {args.attention}")
    print(f"Augmentation: foreground-sample-prob={args.foreground_sample_prob}")
    print(f"Consistency: λ={args.lambda_consistency} rampup={args.consistency_rampup}ep noise={args.consistency_noise}")

    for r in fully_labeled[:3]:
        n_ann = int(r.ann_mask.sum())
        n_biopsy = int((r.target > 0).sum())
        n_inner  = int((r.target >= 0.99).sum())
        print(f"  {r.pid}: {n_ann} annotated voxels ({n_biopsy} biopsy, {n_inner} inner-tumor)")

    if args.leave_one_out:
        if len(fully_labeled) < 2:
            raise ValueError("Leave-one-out requires at least 2 fully labeled patients.")
        splitter = LeaveOneOut()
        split_iter = splitter.split(range(len(fully_labeled)))
        n_folds = len(fully_labeled)
        cv_name = "LeaveOneOut"
    else:
        n_folds = min(args.n_folds, len(fully_labeled))
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(range(len(fully_labeled)))
        cv_name = f"KFold(n_splits={n_folds})"

    print(f"CV strategy: {cv_name}")
    fold_metrics = []

    for fi, (tr_idx, va_idx) in enumerate(split_iter):
        if fi < args.start_fold:
            print(f"Skipping fold {fi} (--start-fold={args.start_fold})")
            continue
        train_recs = [fully_labeled[i] for i in tr_idx]
        val_recs   = [fully_labeled[i] for i in va_idx]
        print(f"\n{'='*60}")
        print(f"FOLD {fi}/{n_folds-1}")
        print(f"Train (fully labeled): {[r.pid for r in train_recs]}")
        print(f"Train (partial):       {partial_ids}")
        print(f"Val:                   {[r.pid for r in val_recs]}")

        m = train_fold(
            fold_idx=fi,
            train_records=train_recs,
            val_records=val_recs,
            partial_records=partially_labeled,
            dataset_dir=args.dataset_dir,
            work_dir=args.work_dir,
            args=args,
        )
        fold_metrics.append(m)

    print("\nAll folds finished.")
    valid_rmses = [m["best_val_rmse"] for m in fold_metrics if not math.isnan(m["best_val_rmse"])]
    mean_rmse = float(np.mean(valid_rmses)) if valid_rmses else float("nan")
    mean_mse  = float(np.mean([r**2 for r in valid_rmses])) if valid_rmses else float("nan")

    summary = {
        "folds": fold_metrics,
        "mean_val_rmse": mean_rmse,
        "mean_val_mse": mean_mse,
        "combined_suffix": args.combined_suffix,
        "absneg_labels": args.absneg_labels,
        "lambda_consistency": args.lambda_consistency,
    }
    sp = args.work_dir / "metrics_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary: {sp}")
    print(f"Mean val RMSE: {mean_rmse:.4f}  MSE: {mean_mse:.4f}")


if __name__ == "__main__":
    main()
