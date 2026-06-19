#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BASE = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_DIR = BASE / "summary_metrics"


MODEL_REGISTRY = {
    "swinunetr_all_modalities/metrics_summary.json":
        ("SwinUNETR 13ch",          "swinunetr", "all_modalities",  "SwinUNETR"),
    "swinunetr_modality_ablation_work/swinunetr_nonPET/metrics_summary.json":
        ("SwinUNETR nonPET",        "swinunetr", "nonPET",          "SwinUNETR"),
    "swinunetr_modality_ablation_work/swinunetr_standard_MRI/metrics_summary.json":
        ("SwinUNETR MRI (4ch)",     "swinunetr", "standard_MRI",    "SwinUNETR"),
    "swinunetr_modality_ablation_work/swinunetr_standard_MRI_PET/metrics_summary.json":
        ("SwinUNETR MRI+PET (8ch)", "swinunetr", "standard_MRI_PET","SwinUNETR"),
    "unet_all_modalities/metrics_summary.json":
        ("UNet 13ch",               "unet_mod",  "all_modalities",  "UNet-none"),
    "unet_modality_ablation_work/unet_ADC_FET/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet ADC+FET",            "unet_mod",  "ADC_FET",         "UNet-none"),
    "unet_modality_ablation_work/unet_diffusion/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet diffusion",          "unet_mod",  "diffusion",       "UNet-none"),
    "unet_modality_ablation_work/unet_nonPET/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet nonPET",             "unet_mod",  "nonPET",          "UNet-none"),
    "unet_modality_ablation_work/unet_perfusion/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet perfusion",          "unet_mod",  "perfusion",       "UNet-none"),
    "unet_modality_ablation_work/unet_PET/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet PET",                "unet_mod",  "PET",             "UNet-none"),
    "unet_modality_ablation_work/unet_standard_MRI/metrics_summary.json":
        ("UNet MRI (4ch)",          "unet_mod",  "standard_MRI",    "UNet-none"),
    "unet_modality_ablation_work/unet_standard_MRI_PET/attn_none/loss_smoothl1/metrics_summary.json":
        ("UNet MRI+PET (8ch)",      "unet_mod",  "standard_MRI_PET","UNet-none"),
    "unet_attention_work/CNN_dataset_ADC_FET/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet ADC+FET (ch-attn)",  "unet_attn", "ADC_FET",         "UNet-channel"),
    "unet_attention_work/CNN_dataset_ADC_FET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet ADC+FET (sp-attn)",  "unet_attn", "ADC_FET",         "UNet-spatial"),
    "unet_attention_work/CNN_dataset_all_modalities/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet 13ch (ch-attn)",     "unet_attn", "all_modalities",  "UNet-channel"),
    "unet_attention_work/CNN_dataset_all_modalities/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet 13ch (sp-attn)",     "unet_attn", "all_modalities",  "UNet-spatial"),
    "unet_attention_work/CNN_dataset_diffusion/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet diffusion (ch-attn)","unet_attn", "diffusion",       "UNet-channel"),
    "unet_attention_work/CNN_dataset_diffusion/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet diffusion (sp-attn)","unet_attn", "diffusion",       "UNet-spatial"),
    "unet_attention_work/CNN_dataset_nonPET/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet nonPET (ch-attn)",   "unet_attn", "nonPET",          "UNet-channel"),
    "unet_attention_work/CNN_dataset_nonPET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet nonPET (sp-attn)",   "unet_attn", "nonPET",          "UNet-spatial"),
    "unet_attention_work/CNN_dataset_perfusion/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet perfusion (ch-attn)","unet_attn", "perfusion",       "UNet-channel"),
    "unet_attention_work/CNN_dataset_perfusion/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet perfusion (sp-attn)","unet_attn", "perfusion",       "UNet-spatial"),
    "unet_attention_work/CNN_dataset_PET/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet PET (ch-attn)",      "unet_attn", "PET",             "UNet-channel"),
    "unet_attention_work/CNN_dataset_PET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet PET (sp-attn)",      "unet_attn", "PET",             "UNet-spatial"),
    "unet_attention_work/CNN_dataset_standard_MRI/attn_channel/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet MRI (ch-attn)",      "unet_attn", "standard_MRI",    "UNet-channel"),
    "unet_attention_work/CNN_dataset_standard_MRI/attn_spatial/loss_smoothl1_ep50/metrics_summary.json":
        ("UNet MRI (sp-attn)",      "unet_attn", "standard_MRI",    "UNet-spatial"),
    "unet_attention_work/CNN_dataset_standard_MRI_PET/attn_channel/loss_smoothl1/metrics_summary.json":
        ("UNet MRI+PET (ch-attn)",  "unet_attn", "standard_MRI_PET","UNet-channel"),
    "unet_attention_work/CNN_dataset_standard_MRI_PET/attn_spatial/loss_smoothl1/metrics_summary.json":
        ("UNet MRI+PET (sp-attn)",  "unet_attn", "standard_MRI_PET","UNet-spatial"),
}

AUC_THRESHOLD = 0.6   
METRICS = ["biopsy_auc", "biopsy_mae", "biopsy_rmse", "best_val_auc", "best_val_rmse"]
METRIC_LABELS = {
    "biopsy_auc":    "Biopsy AUC ↑",
    "biopsy_mae":    "Biopsy MAE ↓",
    "biopsy_rmse":   "Biopsy RMSE ↓",
    "best_val_auc":  "Val AUC ↑",
    "best_val_rmse": "Val RMSE ↓",
}
HIGHER_IS_BETTER = {"biopsy_auc": True, "biopsy_mae": False,
                    "biopsy_rmse": False, "best_val_auc": True, "best_val_rmse": False}

ARCH_COLORS = {
    "SwinUNETR":    "#4477AA",
    "UNet-none":    "#66CCEE",
    "UNet-channel": "#228833",
    "UNet-spatial": "#CCBB44",
}



def load_all(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for rel, (display, group, dataset, arch) in MODEL_REGISTRY.items():
        path = metrics_dir / rel
        if not path.exists():
            print(f"  [MISSING] {rel}")
            continue
        data = json.loads(path.read_text())
        for fold in data["folds"]:
            rows.append({
                "model_key":  rel,
                "display":    display,
                "group":      group,
                "dataset":    dataset,
                "arch":       arch,
                "fold_idx":   fold["fold_idx"],
                **{m: fold.get(m, np.nan) for m in METRICS},
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, group_by: str = "model_key") -> pd.DataFrame:
    agg = {}
    for m in METRICS:
        sub = df.dropna(subset=[m])
        g = sub.groupby(group_by)[m]
        agg[f"{m}_mean"] = g.mean()
        agg[f"{m}_std"]  = g.std()
        agg[f"{m}_n"]    = g.count()
    summary = pd.DataFrame(agg)
    meta = df.drop_duplicates(group_by).set_index(group_by)[["display","group","dataset","arch"]]
    return summary.join(meta)



def _cell_color(val: float, vmin: float, vmax: float, higher_is_better: bool) -> str:
    if np.isnan(val):
        return "#f0f0f0"
    t = (val - vmin) / max(vmax - vmin, 1e-9)
    if not higher_is_better:
        t = 1 - t
    t = np.clip(t, 0, 1)
    r, g, b, _ = plt.cm.RdYlGn(t)
    return mcolors.to_hex((r, g, b))


def _shade(hex_color: str, alpha: float = 0.35) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r2 = r + (1 - r) * (1 - alpha)
    g2 = g + (1 - g) * (1 - alpha)
    b2 = b + (1 - b) * (1 - alpha)
    return mcolors.to_hex((r2, g2, b2))



def render_table(
    summary: pd.DataFrame,
    title: str,
    out_path: Path,
    show_metrics: list[str] | None = None,
    sort_by: str = "biopsy_auc_mean",
    ascending: bool = False,
) -> None:
    show_metrics = show_metrics or METRICS
    cols = [f"{m}_mean" for m in show_metrics]
    col_labels = [METRIC_LABELS[m] for m in show_metrics]

    df = summary.sort_values(sort_by, ascending=ascending).copy()
    row_labels = df["display"].tolist()
    data = df[cols].values

    n_rows, n_cols = data.shape
    fig_h = max(3.0, 0.45 * n_rows + 1.5)
    fig_w = max(8,   1.8 * n_cols + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    col_mins = np.nanmin(data, axis=0)
    col_maxs = np.nanmax(data, axis=0)

    cell_text = []
    cell_colors = []
    best_idx = []
    for ci, m in enumerate(show_metrics):
        col = data[:, ci]
        if HIGHER_IS_BETTER[m]:
            best_idx.append(int(np.nanargmax(col)))
        else:
            best_idx.append(int(np.nanargmin(col)))

    for ri in range(n_rows):
        row_text, row_col = [], []
        for ci, m in enumerate(show_metrics):
            v = data[ri, ci]
            std_v = df.iloc[ri][f"{m}_std"]
            txt = f"{v:.3f}±{std_v:.3f}" if not np.isnan(v) else "—"
            bg = _cell_color(v, col_mins[ci], col_maxs[ci], HIGHER_IS_BETTER[m])
            row_text.append(txt)
            row_col.append(_shade(bg))
        cell_text.append(row_text)
        cell_colors.append(row_col)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)

    for ci in range(n_cols):
        tbl[(0, ci)].set_facecolor("#2C3E50")
        tbl[(0, ci)].get_text().set_color("white")
        tbl[(0, ci)].get_text().set_fontweight("bold")
        tbl[(best_idx[ci] + 1, ci)].get_text().set_fontweight("bold")

    for ri in range(n_rows):
        cell = tbl[(ri + 1, -1)]
        cell.set_facecolor("#ECF0F1")
        cell.get_text().set_ha("right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")



def render_barplot(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    color_by_arch: bool = True,
    sort: bool = True,
    figsize: tuple | None = None,
) -> None:
    df = summary.copy()
    if sort:
        df = df.sort_values(f"{metric}_mean", ascending=not HIGHER_IS_BETTER[metric])

    labels  = df["display"].tolist()
    means   = df[f"{metric}_mean"].values
    stds    = df[f"{metric}_std"].fillna(0).values
    archs   = df["arch"].tolist()

    n = len(labels)
    fig_h = figsize[1] if figsize else max(4.0, 0.45 * n + 1.5)
    fig_w = figsize[0] if figsize else 8.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = [ARCH_COLORS.get(a, "#999999") for a in archs] if color_by_arch else ["#4477AA"] * n
    bars = ax.barh(range(n), means, xerr=stds, color=colors,
                   edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 1, "ecolor": "#555555", "capsize": 3},
                   height=0.7)

    for i, (bar, v) in enumerate(zip(bars, means)):
        ax.text(v + 0.005 + stds[i], i, f"{v:.3f}", va="center", fontsize=7.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.15, means.max() + stds.max() + 0.12))
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.25)

    if color_by_arch:
        handles = [plt.Rectangle((0,0),1,1, color=c) for c in ARCH_COLORS.values()]
        ax.legend(handles, list(ARCH_COLORS.keys()), fontsize=8, loc="lower right",
                  framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def render_grouped_barplot(
    summary: pd.DataFrame,
    metric: str,
    datasets: list[str],
    archs_order: list[str],
    title: str,
    out_path: Path,
) -> None:
    ds_labels = {
        "all_modalities": "all_modalities\n(13ch)",
        "nonPET":         "nonPET\n(9ch)",
        "standard_MRI":   "standard_MRI\n(4ch)",
    }
    n_ds   = len(datasets)
    n_arch = len(archs_order)
    bar_w  = 0.7 / n_arch
    x      = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 2.5), 5))

    for ai, arch in enumerate(archs_order):
        means, stds = [], []
        for ds in datasets:
            row = summary[(summary["dataset"] == ds) & (summary["arch"] == arch)]
            if row.empty:
                means.append(np.nan); stds.append(0)
            else:
                means.append(row[f"{metric}_mean"].iloc[0])
                stds.append(row[f"{metric}_std"].iloc[0])

        offset = (ai - n_arch / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w * 0.9, yerr=stds,
               label=arch, color=ARCH_COLORS.get(arch, "#999"),
               edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 1, "ecolor": "#555", "capsize": 3})

    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels.get(d, d) for d in datasets], fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, min(1.15, ax.get_ylim()[1] + 0.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")



BIOPSY_META_CSV = BASE / "datasets" / "CNN_dataset_all_modalities" / "biopsy_metadata.csv"

KEY_MODELS = [
    "unet_attention_work/CNN_dataset_standard_MRI/attn_channel/loss_smoothl1_ep50/metrics_summary.json",       # 1st  0.778
    "unet_attention_work/CNN_dataset_standard_MRI_PET/attn_channel/loss_smoothl1/metrics_summary.json",        # 2nd  0.776
    "unet_modality_ablation_work/unet_PET/attn_none/loss_smoothl1/metrics_summary.json",                       # 3rd  0.769
    "swinunetr_modality_ablation_work/swinunetr_standard_MRI/metrics_summary.json",                            # 4th  0.767
    "unet_modality_ablation_work/unet_ADC_FET/attn_none/loss_smoothl1/metrics_summary.json",                   # 5th  0.755
]

STRAT_COLS = {
    "Histology": ["Glioblastoma", "Astrocytoma", "Oligoastrocytoma"],
    "IDH":       ["WT", "MT"],
    "Grade":     ["IV", "II"],
}


def build_biopsy_dataframe(metrics_dir: Path) -> pd.DataFrame:
    """
    Collect all per-biopsy predictions from fold-level biopsy_predictions.csv
    files and join with tumour-type metadata.
    Returns one row per biopsy per model with columns:
      model_key, display, group, dataset, arch,
      mri_patient, PAMES_purity, pred_roi_mean,
      Histology, Grade, IDH, TvsN_Predict, Cellularity_mean
    """
    if not BIOPSY_META_CSV.exists():
        print(f"  [WARN] biopsy metadata not found: {BIOPSY_META_CSV}")
        return pd.DataFrame()

    meta = pd.read_csv(BIOPSY_META_CSV)[
        ["mri_patient", "PAMES_purity", "Histology", "Grade",
         "IDH", "TvsN_Predict", "Cellularity_mean"]
    ]

    all_rows = []
    for rel, (display, group, dataset, arch) in MODEL_REGISTRY.items():
        json_path = metrics_dir / rel
        if not json_path.exists():
            continue
        fold_dir = json_path.parent
        csvs = sorted(fold_dir.glob("fold_*/biopsy_predictions.csv"))
        if not csvs:
            continue
        for csv_path in csvs:
            try:
                preds = pd.read_csv(csv_path)
            except Exception:
                continue
            preds["model_key"] = rel
            preds["display"]   = display
            preds["group"]     = group
            preds["dataset"]   = dataset
            preds["arch"]      = arch
            all_rows.append(preds)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.merge(meta, on=["mri_patient", "PAMES_purity"], how="left")
    return df


def _safe_auc(y_true_cont: pd.Series, y_score: pd.Series,
              threshold: float = AUC_THRESHOLD) -> float:
    labels = (y_true_cont > threshold).astype(int)
    if labels.nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(labels, y_score)
    except Exception:
        return np.nan


def _compute_strat_metrics(bdf: pd.DataFrame,
                            strat_col: str,
                            groups: list[str]) -> pd.DataFrame:

    records = {}
    for model_key, mdf in bdf.groupby("model_key"):
        row = {"display": mdf["display"].iloc[0],
               "arch":    mdf["arch"].iloc[0],
               "group":   mdf["group"].iloc[0]}
        for grp in groups:
            sub = mdf[mdf[strat_col] == grp]
            if len(sub) < 2:
                row[f"{grp}_auc"]  = np.nan
                row[f"{grp}_mae"]  = np.nan
                row[f"{grp}_rmse"] = np.nan
                row[f"{grp}_r"]    = np.nan
                continue
            # Patient-mean AUC: average per-patient AUC within this subgroup,
            # matching the equal-patient weighting used for overall AUC
            pt_aucs = []
            for _, pdf in sub.groupby("mri_patient"):
                if len(pdf) >= 2:
                    a = _safe_auc(pdf["PAMES_purity"], pdf["pred_roi_mean"])
                    if not np.isnan(a):
                        pt_aucs.append(a)
            row[f"{grp}_auc"]  = float(np.mean(pt_aucs)) if pt_aucs else np.nan
            row[f"{grp}_mae"]  = (sub["PAMES_purity"] - sub["pred_roi_mean"]).abs().mean()
            row[f"{grp}_rmse"] = np.sqrt(((sub["PAMES_purity"] - sub["pred_roi_mean"])**2).mean())
            r, _ = scipy_stats.pearsonr(sub["PAMES_purity"], sub["pred_roi_mean"]) \
                   if len(sub) >= 3 else (np.nan, np.nan)
            row[f"{grp}_r"] = r
        records[model_key] = row
    return pd.DataFrame(records).T


def render_strat_barplot(strat_df: pd.DataFrame, groups: list[str],
                         metric_suffix: str, metric_label: str,
                         title: str, out_path: Path,
                         key_models: list[str] | None = None) -> None:
    df = strat_df.copy()
    if key_models:
        df = df[df.index.isin(key_models)]
    if df.empty:
        return

    if key_models:
        order = {k: i for i, k in enumerate(key_models)}
        df = df.iloc[df.index.map(order).argsort()]
    else:
        df = df.sort_values(f"{groups[0]}_{metric_suffix}", ascending=False,
                            na_position="last")
    model_labels = df["display"].tolist()
    n_models = len(model_labels)
    n_groups  = len(groups)
    bar_w     = 0.7 / n_groups
    x         = np.arange(n_models)

    GROUP_COLORS = {
        # Histology
        "Glioblastoma":    "#E63946",
        "Astrocytoma":     "#457B9D",
        "Oligoastrocytoma":"#2A9D8F",
        # IDH
        "WT": "#E63946",
        "MT": "#457B9D",
        # Grade
        "IV": "#E63946",
        "II": "#457B9D",
    }

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * n_models + 1.5)))
    for gi, grp in enumerate(groups):
        col_key = f"{grp}_{metric_suffix}"
        vals = df[col_key].values.astype(float)
        offset = (gi - n_groups / 2 + 0.5) * bar_w
        ax.barh(x + offset, vals, height=bar_w * 0.88,
                label=grp,
                color=GROUP_COLORS.get(grp, f"C{gi}"),
                edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + offset, vals):
            if not np.isnan(v):
                ax.text(v + 0.005, xi, f"{v:.3f}", va="center", fontsize=11)

    ax.set_yticks(x)
    ax.set_yticklabels(model_labels, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=8)
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, framealpha=0.8)
    ax.grid(axis="x", alpha=0.25)
    if "auc" in metric_suffix or metric_suffix == "r":
        ax.axvline(AUC_THRESHOLD, color="gray", lw=0.8, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def render_strat_table(strat_df: pd.DataFrame, groups: list[str],
                       strat_name: str, title: str,
                       out_path: Path,
                       key_models: list[str] | None = None) -> None:
    df = strat_df.copy()
    if key_models:
        df = df[df.index.isin(key_models)]
    if df.empty:
        return

    if key_models:
        order = {k: i for i, k in enumerate(key_models)}
        df = df.iloc[df.index.map(order).argsort()]
    else:
        df = df.sort_values(f"{groups[0]}_auc", ascending=False, na_position="last")
    row_labels = df["display"].tolist()

    col_keys   = [f"{g}_{m}" for g in groups for m in ["auc", "mae", "rmse"]]
    col_labels = [f"{g}\n{m.upper()}" for g in groups for m in ["auc", "mae", "rmse"]]
    data = df[col_keys].values.astype(float)

    n_rows, n_cols = data.shape
    fig_h = max(3.0, 0.45 * n_rows + 1.5)
    fig_w = max(8,   1.6 * n_cols + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    higher_better = [True, False, False] * len(groups)  # auc↑, mae↓, rmse↓

    cell_text   = []
    cell_colors = []
    col_mins = np.nanmin(data, axis=0)
    col_maxs = np.nanmax(data, axis=0)

    best_idx = []
    for ci, hb in enumerate(higher_better):
        col = data[:, ci]
        if np.all(np.isnan(col)):
            best_idx.append(0)
        elif hb:
            best_idx.append(int(np.nanargmax(col)))
        else:
            best_idx.append(int(np.nanargmin(col)))

    for ri in range(n_rows):
        rt, rc = [], []
        for ci, hb in enumerate(higher_better):
            v = data[ri, ci]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            bg  = _cell_color(v, col_mins[ci], col_maxs[ci], hb)
            rt.append(txt); rc.append(_shade(bg))
        cell_text.append(rt); cell_colors.append(rc)

    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    for ci in range(n_cols):
        tbl[(0, ci)].set_facecolor("#2C3E50")
        tbl[(0, ci)].get_text().set_color("white")
        tbl[(0, ci)].get_text().set_fontweight("bold")
        tbl[(best_idx[ci] + 1, ci)].get_text().set_fontweight("bold")
    for ri in range(n_rows):
        cell = tbl[(ri + 1, -1)]
        cell.set_facecolor("#ECF0F1")
        cell.get_text().set_ha("right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def render_tumor_vs_normal(bdf: pd.DataFrame, out_path: Path,
                           key_models: list[str] | None = None) -> None:
    df = bdf.copy()
    if key_models and "model_key" in df.columns:
        df = df[df["model_key"].isin(key_models)]
    if df.empty:
        return

    records = {}
    for model_key, mdf in df.groupby("model_key"):
        display = mdf["display"].iloc[0]
        t_mean = mdf.loc[mdf["TvsN_Predict"] == "Tumor",  "pred_roi_mean"].mean()
        n_mean = mdf.loc[mdf["TvsN_Predict"] == "Normal", "pred_roi_mean"].mean()
        t_std  = mdf.loc[mdf["TvsN_Predict"] == "Tumor",  "pred_roi_mean"].std()
        n_std  = mdf.loc[mdf["TvsN_Predict"] == "Normal", "pred_roi_mean"].std()
        records[model_key] = {"display": display,
                              "tumor_mean": t_mean, "tumor_std": t_std,
                              "normal_mean": n_mean, "normal_std": n_std}

    rdf = pd.DataFrame(records).T
    rdf = rdf.sort_values("tumor_mean", ascending=False, na_position="last")

    n  = len(rdf)
    bw = 0.35
    x  = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * n + 1.5)))

    ax.barh(x - bw/2, rdf["tumor_mean"].astype(float),  bw,
            xerr=rdf["tumor_std"].astype(float),
            label="Tumor biopsies", color="#E63946",
            edgecolor="white", linewidth=0.4,
            error_kw={"elinewidth": 1, "ecolor": "#555", "capsize": 3})
    ax.barh(x + bw/2, rdf["normal_mean"].astype(float), bw,
            xerr=rdf["normal_std"].astype(float),
            label="Normal biopsies", color="#457B9D",
            edgecolor="white", linewidth=0.4,
            error_kw={"elinewidth": 1, "ecolor": "#555", "capsize": 3})

    ax.set_yticks(x)
    ax.set_yticklabels(rdf["display"].tolist(), fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Mean predicted purity", fontsize=13)
    ax.set_title("Predicted purity: Tumour vs Normal biopsies", fontsize=15, pad=8)
    ax.axvline(AUC_THRESHOLD, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def render_cellularity_scatter(bdf: pd.DataFrame, out_path: Path,
                               key_models: list[str] | None = None) -> None:
    df = bdf.copy()
    if key_models:
        df = df[df["model_key"].isin(key_models)]
    if df.empty or "Cellularity_mean" not in df.columns:
        return

    df = df.dropna(subset=["Cellularity_mean", "pred_roi_mean"])
    models = df["model_key"].unique()
    cmap   = plt.cm.get_cmap("tab10", len(models))

    fig, ax = plt.subplots(figsize=(8, 6))
    for mi, mk in enumerate(models):
        sub = df[df["model_key"] == mk]
        lbl = sub["display"].iloc[0]
        ax.scatter(sub["Cellularity_mean"], sub["pred_roi_mean"],
                   color=cmap(mi), alpha=0.55, s=30, label=lbl, linewidths=0)

    ax.set_xlabel("Cellularity (mean)", fontsize=13)
    ax.set_ylabel("Predicted purity", fontsize=13)
    ax.set_title("Predicted purity vs Tumour Cellularity", fontsize=15, pad=8)
    ax.axhline(AUC_THRESHOLD, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.8, ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return pvalues
    order   = np.argsort(pvalues)
    ranks   = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = pvalues * n / ranks
    # enforce monotonicity from right
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return "ns"


def _rank_biserial(aucs: list[float], mu: float = 0.5) -> float:
    diffs = [a - mu for a in aucs if not np.isnan(a - mu)]
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return float("nan")
    ranks = scipy_stats.rankdata(np.abs(diffs))
    W_plus  = sum(r for r, d in zip(ranks, diffs) if d > 0)
    W_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    n = len(diffs)
    return (W_plus - W_minus) / (n * (n + 1) / 2)


def compute_significance_table(metrics_dir: Path, key_models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for rel, (display, group, dataset, arch) in MODEL_REGISTRY.items():
        p = metrics_dir / rel
        if not p.exists():
            continue
        d = json.load(open(p))
        aucs = [f["biopsy_auc"] for f in d["folds"] if not np.isnan(f["biopsy_auc"])]

        # Wilcoxon vs 0.5
        w_stat, w_p = (np.nan, np.nan)
        rb = np.nan
        if len(aucs) >= 6:
            try:
                res = scipy_stats.wilcoxon(
                    [a - 0.5 for a in aucs],
                    alternative="greater", zero_method="pratt"
                )
                w_stat, w_p = res.statistic, res.pvalue
                rb = _rank_biserial(aucs)
            except Exception:
                pass

        # Spearman r on pooled biopsies
        trues, preds = [], []
        for fold in d["folds"]:
            fold_idx = fold["fold_idx"]
            csv = p.parent / f"fold_{fold_idx}" / "biopsy_predictions.csv"
            if csv.exists():
                bdf = pd.read_csv(csv)
                trues.extend(bdf["PAMES_purity"].tolist())
                preds.extend(bdf["pred_roi_mean"].tolist())
        sp_r, sp_p = (np.nan, np.nan)
        if len(trues) >= 5:
            try:
                sp_r, sp_p = scipy_stats.spearmanr(trues, preds)
            except Exception:
                pass

        rows.append({
            "model_key": rel, "display": display, "group": group,
            "n_folds": len(aucs),
            "mean_auc": np.mean(aucs) if aucs else np.nan,
            "std_auc":  np.std(aucs)  if aucs else np.nan,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "effect_size_rb": rb,
            "n_biopsies": len(trues),
            "spearman_r": sp_r, "spearman_p": sp_p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if key_models:
        df = df[df["model_key"].isin(key_models)].reset_index(drop=True)

    # FDR correction separately for Wilcoxon and Spearman p-values
    mask_w = df["wilcoxon_p"].notna()
    mask_s = df["spearman_p"].notna()
    df["wilcoxon_p_fdr"] = np.nan
    df["spearman_p_fdr"] = np.nan
    if mask_w.any():
        df.loc[mask_w, "wilcoxon_p_fdr"] = _fdr_correction(df.loc[mask_w, "wilcoxon_p"].values)
    if mask_s.any():
        df.loc[mask_s, "spearman_p_fdr"] = _fdr_correction(df.loc[mask_s, "spearman_p"].values)

    return df.sort_values("mean_auc", ascending=False).reset_index(drop=True)


def render_significance_table(sig_df: pd.DataFrame, out_path: Path) -> None:
    cols = ["Model", "n", "AUC (mean±std)", "Wilcoxon p*", "Sig", "Effect (rb)", "Spearman r", "Spearman p*"]
    rows = []
    for _, row in sig_df.iterrows():
        auc_str   = f"{row['mean_auc']:.3f} ± {row['std_auc']:.3f}" if not np.isnan(row['mean_auc']) else "—"
        w_p_str   = f"{row['wilcoxon_p_fdr']:.3f}"  if not np.isnan(row['wilcoxon_p_fdr'])  else "—"
        sig_str   = _sig_stars(row['wilcoxon_p_fdr']) if not np.isnan(row['wilcoxon_p_fdr']) else "—"
        rb_str    = f"{row['effect_size_rb']:.3f}"   if not np.isnan(row['effect_size_rb'])  else "—"
        sp_r_str  = f"{row['spearman_r']:.3f}"       if not np.isnan(row['spearman_r'])       else "—"
        sp_p_str  = f"{row['spearman_p_fdr']:.3f}"   if not np.isnan(row['spearman_p_fdr'])   else "—"
        rows.append([row["display"], int(row["n_folds"]), auc_str,
                     w_p_str, sig_str, rb_str, sp_r_str, sp_p_str])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(15, max(3, 0.38 * n_rows + 1.8)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols))))

    # colour header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # colour sig column and AUC
    for i, row_data in enumerate(rows, start=1):
        sig = row_data[4]
        bg = "#FFFFFF"
        if sig in ("***", "**"):     bg = "#D4EDDA"
        elif sig == "*":             bg = "#FFF3CD"
        elif sig == ".":             bg = "#FFF9E6"
        tbl[i, 4].set_facecolor(bg)
        # stripe alternate rows
        row_bg = "#F8F8F8" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(cols)):
            if j != 4:
                tbl[i, j].set_facecolor(row_bg)

    ax.set_title("Statistical significance of biopsy AUC (one-sided Wilcoxon vs 0.5)\n"
                 "* FDR-corrected (Benjamini-Hochberg)  |  Sig: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1",
                 fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def compute_pairwise_comparison(sig_df: pd.DataFrame, metrics_dir: Path,
                                reference_key: str) -> pd.DataFrame:
    ref_row = sig_df[sig_df["model_key"] == reference_key]
    if ref_row.empty:
        return pd.DataFrame()

    ref_rel  = reference_key
    ref_data = json.load(open(metrics_dir / ref_rel))
    ref_aucs_by_fold = {f["fold_idx"]: f["biopsy_auc"] for f in ref_data["folds"]}

    rows = []
    for _, row in sig_df.iterrows():
        if row["model_key"] == reference_key:
            continue
        cmp_data = json.load(open(metrics_dir / row["model_key"]))
        cmp_aucs_by_fold = {f["fold_idx"]: f["biopsy_auc"] for f in cmp_data["folds"]}

        common = sorted(set(ref_aucs_by_fold) & set(cmp_aucs_by_fold))
        ref_vals = [ref_aucs_by_fold[k] for k in common]
        cmp_vals = [cmp_aucs_by_fold[k] for k in common]

        pairs = [(r, c) for r, c in zip(ref_vals, cmp_vals)
                 if not (np.isnan(r) or np.isnan(c))]
        if len(pairs) < 6:
            rows.append({"model_key": row["model_key"], "display": row["display"],
                         "ref_mean": np.nanmean(ref_vals), "cmp_mean": np.mean([c for _, c in pairs]),
                         "delta": np.nan, "wilcoxon_p": np.nan, "wilcoxon_p_fdr": np.nan})
            continue

        ref_p, cmp_p = zip(*pairs)
        delta = np.mean(ref_p) - np.mean(cmp_p)
        try:
            diffs = [r - c for r, c in zip(ref_p, cmp_p)]
            stat, p = scipy_stats.wilcoxon(diffs, alternative="two-sided", zero_method="pratt")
        except Exception:
            stat, p = np.nan, np.nan

        rows.append({"model_key": row["model_key"], "display": row["display"],
                     "ref_mean": np.mean(ref_p), "cmp_mean": np.mean(cmp_p),
                     "delta": delta, "wilcoxon_p": p, "wilcoxon_p_fdr": np.nan})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    mask = df["wilcoxon_p"].notna()
    if mask.any():
        df.loc[mask, "wilcoxon_p_fdr"] = _fdr_correction(df.loc[mask, "wilcoxon_p"].values)
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


def render_pairwise_table(pair_df: pd.DataFrame, ref_display: str, out_path: Path) -> None:
    cols = ["Comparison model", "Ref AUC", "Model AUC", "ΔAUC", "Wilcoxon p*", "Sig"]
    rows = []
    for _, row in pair_df.iterrows():
        delta_str = f"{row['delta']:+.3f}" if not np.isnan(row['delta']) else "—"
        p_str     = f"{row['wilcoxon_p_fdr']:.3f}" if not np.isnan(row['wilcoxon_p_fdr']) else "—"
        sig_str   = _sig_stars(row['wilcoxon_p_fdr']) if not np.isnan(row['wilcoxon_p_fdr']) else "—"
        rows.append([row["display"],
                     f"{row['ref_mean']:.3f}", f"{row['cmp_mean']:.3f}",
                     delta_str, p_str, sig_str])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13, max(3, 0.38 * n_rows + 2.2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols))))

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, row_data in enumerate(rows, start=1):
        sig = row_data[5]
        bg = "#F8F8F8" if i % 2 == 0 else "#FFFFFF"
        if sig in ("***", "**"):   bg = "#D4EDDA"
        elif sig == "*":           bg = "#FFF3CD"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)
        delta_val = row_data[3]
        if delta_val.startswith("+"):
            tbl[i, 3].set_text_props(color="#228833", fontweight="bold")
        elif delta_val.startswith("-"):
            tbl[i, 3].set_text_props(color="#CC3333")

    ax.set_title(f"Pairwise comparison: {ref_display} vs all models\n"
                 "Wilcoxon paired signed-rank (two-sided, FDR-corrected)  |  ΔAUC = ref − other",
                 fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def render_subgroup_stats_table(bdf: pd.DataFrame, out_path: Path,
                                key_models: list[str] | None = None) -> None:
    df = bdf.copy()
    if key_models and "model_key" in df.columns:
        df = df[df["model_key"].isin(key_models)]
    if df.empty:
        return

    comparisons = [
        ("Tumour vs Normal",  "TvsN_Predict",  "Tumor",        "Normal"),
        ("GBM vs non-GBM",    "Histology",      "Glioblastoma", None),
        ("IDH-WT vs IDH-MT",  "IDH",            "WT",           "MT"),
    ]

    rows = []
    for model_key, mdf in df.groupby("model_key"):
        display = mdf["display"].iloc[0]
        row = {"display": display}
        for label, col, g1, g2 in comparisons:
            if col not in mdf.columns:
                row[f"{label}_U"] = np.nan
                row[f"{label}_p"] = np.nan
                continue
            grp1 = mdf.loc[mdf[col] == g1, "pred_roi_mean"].dropna()
            if g2 is None:
                grp2 = mdf.loc[mdf[col] != g1, "pred_roi_mean"].dropna()
            else:
                grp2 = mdf.loc[mdf[col] == g2, "pred_roi_mean"].dropna()
            if len(grp1) < 3 or len(grp2) < 3:
                row[f"{label}_U"] = np.nan
                row[f"{label}_p"] = np.nan
                continue
            try:
                U, p = scipy_stats.mannwhitneyu(grp1, grp2, alternative="two-sided")
                row[f"{label}_U"] = U
                row[f"{label}_p"] = p
            except Exception:
                row[f"{label}_U"] = np.nan
                row[f"{label}_p"] = np.nan
        rows.append(row)

    stat_df = pd.DataFrame(rows)
    if stat_df.empty:
        return

    for label, _, _, _ in comparisons:
        col = f"{label}_p"
        mask = stat_df[col].notna()
        if mask.any():
            stat_df.loc[mask, f"{label}_p_fdr"] = _fdr_correction(stat_df.loc[mask, col].values)
        else:
            stat_df[f"{label}_p_fdr"] = np.nan

    header = ["Model"] + [f"{lbl}\np (FDR)" for lbl, _, _, _ in comparisons] + \
             [f"{lbl}\nSig"       for lbl, _, _, _ in comparisons]
    col_order = ["display"] + \
                [f"{lbl}_p_fdr"  for lbl, _, _, _ in comparisons] + \
                [f"{lbl}_sig"    for lbl, _, _, _ in comparisons]

    for lbl, _, _, _ in comparisons:
        stat_df[f"{lbl}_sig"] = stat_df[f"{lbl}_p_fdr"].apply(
            lambda p: _sig_stars(p) if not np.isnan(p) else "—"
        )

    tbl_rows = []
    for _, r in stat_df.iterrows():
        row_vals = [r["display"]]
        for lbl, _, _, _ in comparisons:
            p = r.get(f"{lbl}_p_fdr", np.nan)
            row_vals.append(f"{p:.3f}" if not np.isnan(p) else "—")
        for lbl, _, _, _ in comparisons:
            row_vals.append(r.get(f"{lbl}_sig", "—"))
        tbl_rows.append(row_vals)

    n_rows = len(tbl_rows)
    fig, ax = plt.subplots(figsize=(14, max(3, 0.38 * n_rows + 2.2)))
    ax.axis("off")
    tbl = ax.table(cellText=tbl_rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(header))))

    for j in range(len(header)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    n_cmp = len(comparisons)
    for i, row_vals in enumerate(tbl_rows, start=1):
        bg = "#F8F8F8" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(header)):
            tbl[i, j].set_facecolor(bg)
        for k in range(n_cmp):
            sig = row_vals[1 + n_cmp + k]
            if sig in ("***", "**"):   tbl[i, 1 + n_cmp + k].set_facecolor("#D4EDDA")
            elif sig == "*":           tbl[i, 1 + n_cmp + k].set_facecolor("#FFF3CD")

    ax.set_title("Subgroup significance: Mann-Whitney U (two-sided, FDR-corrected)\n"
                 "Compares predicted purity distributions between clinical subgroups",
                 fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


ATTN_DATASETS = {
    "all_modalities":  {
        "none":    "unet_all_modalities/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_all_modalities/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_all_modalities/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "nonPET": {
        "none":    "unet_modality_ablation_work/unet_nonPET/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_nonPET/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_nonPET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "standard_MRI": {
        "none":    "unet_modality_ablation_work/unet_standard_MRI/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_standard_MRI/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_standard_MRI/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "standard_MRI_PET": {
        "none":    "unet_modality_ablation_work/unet_standard_MRI_PET/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_standard_MRI_PET/attn_channel/loss_smoothl1/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_standard_MRI_PET/attn_spatial/loss_smoothl1/metrics_summary.json",
    },
    "PET": {
        "none":    "unet_modality_ablation_work/unet_PET/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_PET/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_PET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "ADC_FET": {
        "none":    "unet_modality_ablation_work/unet_ADC_FET/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_ADC_FET/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_ADC_FET/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "diffusion": {
        "none":    "unet_modality_ablation_work/unet_diffusion/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_diffusion/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_diffusion/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
    "perfusion": {
        "none":    "unet_modality_ablation_work/unet_perfusion/attn_none/loss_smoothl1/metrics_summary.json",
        "channel": "unet_attention_work/CNN_dataset_perfusion/attn_channel/loss_smoothl1_ep50/metrics_summary.json",
        "spatial": "unet_attention_work/CNN_dataset_perfusion/attn_spatial/loss_smoothl1_ep50/metrics_summary.json",
    },
}


def compute_attention_comparison(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    comparisons = [("channel", "none"), ("spatial", "none"), ("channel", "spatial")]

    for dataset, paths in ATTN_DATASETS.items():
        aucs = {}
        for attn, rel in paths.items():
            p = metrics_dir / rel
            if not p.exists():
                continue
            d = json.load(open(p))
            aucs[attn] = {f["fold_idx"]: f["biopsy_auc"] for f in d["folds"]}

        for attn_a, attn_b in comparisons:
            if attn_a not in aucs or attn_b not in aucs:
                continue
            common = sorted(set(aucs[attn_a]) & set(aucs[attn_b]))
            pairs = [(aucs[attn_a][k], aucs[attn_b][k]) for k in common
                     if not (np.isnan(aucs[attn_a][k]) or np.isnan(aucs[attn_b][k]))]
            if len(pairs) < 6:
                continue
            a_vals, b_vals = zip(*pairs)
            delta = float(np.mean(a_vals) - np.mean(b_vals))
            try:
                diffs = [a - b for a, b in pairs]
                _, p = scipy_stats.wilcoxon(diffs, alternative="two-sided", zero_method="pratt")
            except Exception:
                p = np.nan
            rows.append({
                "dataset": dataset,
                "comparison": f"{attn_a} vs {attn_b}",
                "mean_a": float(np.mean(a_vals)),
                "mean_b": float(np.mean(b_vals)),
                "delta": delta,
                "n_pairs": len(pairs),
                "wilcoxon_p": p,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    mask = df["wilcoxon_p"].notna()
    if mask.any():
        df.loc[mask, "wilcoxon_p_fdr"] = _fdr_correction(df.loc[mask, "wilcoxon_p"].values)
    else:
        df["wilcoxon_p_fdr"] = np.nan
    return df


def render_attention_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = ["Dataset", "Comparison", "Mean A", "Mean B", "ΔAUC", "p (FDR)*", "Sig"]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["dataset"],
            r["comparison"],
            f"{r['mean_a']:.3f}",
            f"{r['mean_b']:.3f}",
            f"{r['delta']:+.3f}",
            f"{r['wilcoxon_p_fdr']:.3f}" if not np.isnan(r.get("wilcoxon_p_fdr", np.nan)) else "—",
            _sig_stars(r["wilcoxon_p_fdr"]) if not np.isnan(r.get("wilcoxon_p_fdr", np.nan)) else "—",
        ])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13, max(3, 0.38 * n_rows + 2.2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols))))

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    prev_dataset = None
    for i, row_data in enumerate(rows, start=1):
        bg = "#F0F4F8" if row_data[0] != prev_dataset and (i % 2 == 0) else (
             "#FFFFFF" if row_data[0] == prev_dataset else "#F8F8F8")
        prev_dataset = row_data[0]
        sig = row_data[6]
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)
        if sig in ("***", "**"):   tbl[i, 6].set_facecolor("#D4EDDA")
        elif sig == "*":           tbl[i, 6].set_facecolor("#FFF3CD")
        delta_val = row_data[4]
        if delta_val.startswith("+"):
            tbl[i, 4].set_text_props(color="#228833", fontweight="bold")
        elif delta_val.startswith("-"):
            tbl[i, 4].set_text_props(color="#CC3333")

    ax.set_title(
        "Attention type comparison: Wilcoxon paired signed-rank (two-sided, FDR-corrected)\n"
        "ΔAUC = mean(A) − mean(B)  |  * FDR (Benjamini-Hochberg)  |  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    args = ap.parse_args()

    metrics_dir = args.metrics_dir
    out_dir = metrics_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics…")
    df_all = load_all(metrics_dir)
    if df_all.empty:
        print("No data found — check --metrics-dir")
        return

    summary = summarise(df_all)

    csv_path = out_dir / "all_models_summary.csv"
    export_cols = (
        [c for c in ["display", "group", "dataset", "arch"]]
        + [f"{m}_mean" for m in METRICS]
        + [f"{m}_std"  for m in METRICS]
        + [f"{m}_n"    for m in METRICS]
    )
    summary[export_cols].sort_values("biopsy_auc_mean", ascending=False).to_csv(csv_path)
    print(f"\nSaved: {csv_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 01 — Overview: all 24 models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 01: Overview ──")
    render_barplot(
        summary, "biopsy_auc",
        "All models — Biopsy AUC (mean ± std across LOO folds)",
        out_dir / "01_overview_biopsy_auc_barplot.png",
    )
    render_table(
        summary,
        "Overview — all models (sorted by Biopsy AUC)",
        out_dir / "01_overview_all_metrics_table.png",
        show_metrics=METRICS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Section 02 — SwinUNETR vs UNet on overlapping datasets
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 02: SwinUNETR vs UNet overlap ──")
    overlap_datasets = ["all_modalities", "nonPET", "standard_MRI"]
    overlap_archs    = ["SwinUNETR", "UNet-none", "UNet-channel", "UNet-spatial"]
    overlap_mask = (
        summary["dataset"].isin(overlap_datasets) &
        summary["arch"].isin(overlap_archs)
    )
    summary_overlap = summary[overlap_mask].copy()
    summary_overlap = summary_overlap.sort_values(
        ["dataset", "arch"],
        key=lambda s: s.map(
            {d: i for i, d in enumerate(overlap_datasets)} if s.name == "dataset"
            else {a: i for i, a in enumerate(overlap_archs)}
        )
    )

    render_grouped_barplot(
        summary_overlap, "biopsy_auc",
        overlap_datasets, overlap_archs,
        "SwinUNETR vs UNet — Biopsy AUC by dataset",
        out_dir / "02_swinunetr_vs_unet_overlap_barplot.png",
    )
    render_table(
        summary_overlap,
        "SwinUNETR vs UNet — overlapping datasets (Biopsy AUC, MAE, RMSE)",
        out_dir / "02_swinunetr_vs_unet_overlap_table.png",
        show_metrics=["biopsy_auc", "biopsy_mae", "biopsy_rmse"],
        sort_by="biopsy_auc_mean",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Section 03 — All SwinUNETR models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 03: All SwinUNETR models ──")
    summary_swi = summary[summary["group"] == "swinunetr"].copy()
    render_barplot(
        summary_swi, "biopsy_auc",
        "SwinUNETR models — Biopsy AUC",
        out_dir / "03_swinunetr_all_barplot.png",
        figsize=(8, 3.5),
    )
    render_table(
        summary_swi,
        "All SwinUNETR models",
        out_dir / "03_swinunetr_all_table.png",
        show_metrics=METRICS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Section 04 — UNet modality ablations (no-attention)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 04: UNet modality ablations ──")
    summary_mod = summary[summary["group"] == "unet_mod"].copy()
    render_barplot(
        summary_mod, "biopsy_auc",
        "UNet modality ablations (no attention) — Biopsy AUC",
        out_dir / "04_unet_modality_ablation_barplot.png",
    )
    render_table(
        summary_mod,
        "UNet modality ablations (no attention)",
        out_dir / "04_unet_modality_ablation_table.png",
        show_metrics=METRICS,
    )

    # Also a multi-metric barplot for the ablation (side-by-side AUC/MAE/RMSE)
    biopsy_metrics = ["biopsy_auc", "biopsy_mae", "biopsy_rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.45 * len(summary_mod) + 1.5)))
    df_s = summary_mod.sort_values("biopsy_auc_mean", ascending=False)
    labels = df_s["display"].tolist()
    colors = [ARCH_COLORS.get(a, "#999") for a in df_s["arch"].tolist()]

    for ax, m in zip(axes, biopsy_metrics):
        means = df_s[f"{m}_mean"].values
        stds  = df_s[f"{m}_std"].fillna(0).values
        ax.barh(range(len(labels)), means, xerr=stds,
                color=colors, edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 1, "ecolor": "#555", "capsize": 3},
                height=0.7)
        for i, v in enumerate(means):
            ax.text(v + 0.005 + stds[i], i, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if m == "biopsy_auc" else [], fontsize=9)
        ax.set_xlabel(METRIC_LABELS[m], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        if m == "biopsy_auc":
            ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    fig.suptitle("UNet modality ablations — biopsy metrics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "04_unet_modality_ablation_multimetric_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_unet_modality_ablation_multimetric_barplot.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 05 — UNet attention work
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 05: UNet attention work ──")
    summary_attn = summary[summary["group"] == "unet_attn"].copy()
    render_barplot(
        summary_attn, "biopsy_auc",
        "UNet attention ablations — Biopsy AUC",
        out_dir / "05_unet_attention_barplot.png",
    )
    render_table(
        summary_attn,
        "UNet attention ablations (channel / spatial)",
        out_dir / "05_unet_attention_table.png",
        show_metrics=["biopsy_auc", "biopsy_mae", "biopsy_rmse"],
    )

    attn_datasets = ["all_modalities", "nonPET", "standard_MRI", "standard_MRI_PET",
                     "ADC_FET", "diffusion", "perfusion", "PET"]
    attn_archs    = ["UNet-channel", "UNet-spatial"]
    render_grouped_barplot(
        summary_attn, "biopsy_auc",
        attn_datasets, attn_archs,
        "UNet: channel-attn vs spatial-attn — Biopsy AUC by dataset",
        out_dir / "05_unet_attention_channel_vs_spatial_barplot.png",
    )

    print(f"\nAll outputs in: {out_dir}/")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 06 — Tumour-type stratified analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 06: Tumour-type analysis ──")
    bdf = build_biopsy_dataframe(metrics_dir)
    if not bdf.empty:
        bdf.to_csv(out_dir / "06_biopsy_predictions_all.csv", index=False)
        print(f"  Saved biopsy dataframe: {len(bdf)} rows × {bdf.shape[1]} cols")

        for strat_col, groups in STRAT_COLS.items():
            strat_df = _compute_strat_metrics(bdf, strat_col, groups)
            if strat_df.empty:
                print(f"  [skip] {strat_col}: no data")
                continue
            slug = strat_col.lower()
            render_strat_barplot(
                strat_df, groups, "auc",
                f"Biopsy AUC by {strat_col}",
                f"Biopsy AUC stratified by {strat_col}",
                out_dir / f"06_{slug}_auc_barplot.png",
                key_models=KEY_MODELS,
            )
            render_strat_table(
                strat_df, groups, strat_col,
                f"Biopsy metrics by {strat_col}",
                out_dir / f"06_{slug}_table.png",
                key_models=KEY_MODELS,
            )

        render_tumor_vs_normal(
            bdf,
            out_dir / "06_tumor_vs_normal_barplot.png",
            key_models=KEY_MODELS,
        )
        render_cellularity_scatter(
            bdf,
            out_dir / "06_cellularity_scatter.png",
            key_models=KEY_MODELS,
        )
    else:
        print("  No biopsy prediction data found — skipping section 06")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 07 — Statistical analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Section 07: Statistical analysis ──")

    print("\n  07a: Model significance vs chance...")
    sig_df = compute_significance_table(metrics_dir, key_models=KEY_MODELS)
    if not sig_df.empty:
        sig_df.to_csv(out_dir / "07_significance_table.csv", index=False)
        render_significance_table(sig_df, out_dir / "07a_model_significance.png")
    else:
        print("  No data for significance analysis")

    print("\n  07b: Pairwise model comparison (top 5)...")
    if not sig_df.empty:
        best_key  = sig_df.loc[sig_df["mean_auc"].idxmax(), "model_key"]
        best_disp = sig_df.loc[sig_df["mean_auc"].idxmax(), "display"]
        pair_df = compute_pairwise_comparison(sig_df, metrics_dir, reference_key=best_key)
        if not pair_df.empty:
            render_pairwise_table(pair_df, best_disp, out_dir / "07b_pairwise_model_comparison.png")

    print("\n  07c: Attention type comparison per dataset...")
    attn_df = compute_attention_comparison(metrics_dir)
    if not attn_df.empty:
        attn_df.to_csv(out_dir / "07_attention_comparison.csv", index=False)
        render_attention_comparison_table(attn_df, out_dir / "07c_attention_comparison.png")
    else:
        print("  No attention comparison data")

    print("\n  07d: Subgroup significance (tumor types)...")
    if not bdf.empty:
        render_subgroup_stats_table(
            bdf, out_dir / "07d_subgroup_tumor_types.png", key_models=KEY_MODELS
        )
    else:
        print("  No biopsy data for subgroup stats")

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
