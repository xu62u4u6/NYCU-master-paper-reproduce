"""成分層級突變與克隆分析(對應論文 §3.2.1,Fig16–19)。

來源 notebook:paper/paper_scripts/result3.2.1.ipynb(去除大量重複/探索 cell)。

內容:
  Fig16a:11 例 ASC 的 A/T/S 成分 oncoplot(top30 突變基因)
  Fig16b:成分層級 COSMIC signature heatmap
  Fig16c:成分層級基因體指標(MSI/TMB/Ti-Tv/HRD,paired,A/T/S 間無顯著差異)
  Fig17 :SMG(EGFR/TP53/RBM10)在各成分的突變(driver gene oncoplot)
  Fig18 :case 層級突變組成 — shared / A-private / S-private 表 + Venn
  Fig19 :PyClone CCF 熱圖(major clone 跨成分共享,支持單株起源)

可測數字(compute_recurrent_mutations,變異層級、recurrent=≥2 cases):
  EGFR L858R recurrent shared      → 5 cases
  RPS3A  missense recurrent A-private → 2 cases
  HLA-DQB1 missense recurrent S-private → 2 cases

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/component_mutation.py
"""

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))

from pymaftools import MAF, OncoPlot, PivotTable  # noqa: E402
from asc.cohort_io import ASCDataLoader  # noqa: E402
from matplotlib_venn import venn3  # noqa: E402

OUT_DIR = REPRO_DIR / "outputs" / "component_mutation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PYCLONE_DIR = "data/pyclone_output"
AST_MAF = "data/WES/all_case_maf_AST.csv"

METRICS = ["MSI", "TMB", "ti/tv", "HRD-sum"]


# ---------------- 可測:變異層級 recurrent shared / private ----------------

def _find_mutations(col):
    """把某樣本欄(mutation_id -> 突變型別 或 False)轉成 '變異字串' 集合。"""
    df = pd.DataFrame(col)
    nf = df[df != False]
    return {m for m in (nf.index + "|" + nf.values) if pd.notna(m) and m != ""}


def compute_recurrent_mutations(dataloader):
    """計算成分間 shared / A-private / S-private 的 recurrent(≥2 cases)變異。

    回傳每個基因在「≥2 cases」出現的變異字串 -> case 數。
    shared = A∩S∩T;A-private = A−(A∩S);S-private = S−(A∩S)。
    """
    ATS = dataloader.get_ATS_subset()
    case_IDs = ATS.SNV.sample_metadata.case_ID.unique()
    mt = MAF.read_csv(AST_MAF).to_mutation_table()

    shared, a_priv, s_priv = Counter(), Counter(), Counter()
    for cid in case_IDs:
        A = _find_mutations(mt.loc[:, f"{cid}_A"])
        S = _find_mutations(mt.loc[:, f"{cid}_S"])
        T = _find_mutations(mt.loc[:, f"{cid}_T"])
        as_shared = A & S
        for m in (A & S & T):
            shared[m] += 1
        for m in (A - as_shared):
            a_priv[m] += 1
        for m in (S - as_shared):
            s_priv[m] += 1

    def recurrent(counter, gene, missense_only=False):
        out = {}
        for m, n in counter.items():
            parts = m.split("|")
            if parts[0] != gene or n < 2:
                continue
            if missense_only and parts[-1] != "Missense_Mutation":
                continue
            out[m] = n
        return out

    return {
        "egfr_shared": recurrent(shared, "EGFR", missense_only=True),
        "rps3a_a_private": recurrent(a_priv, "RPS3A", missense_only=True),
        "hla_dqb1_s_private": recurrent(s_priv, "HLA-DQB1", missense_only=True),
    }


# ---------------- 可測:成分層級基因體指標(paired,無顯著差異)----------------

def compute_component_metrics(dataloader):
    """A/T/S 三成分配對比較各指標(Friedman test)。論文:皆無顯著差異。"""
    ATS = dataloader.get_ATS_subset()
    meta = ATS.SNV.sample_metadata
    out = {}
    for metric in METRICS:
        wide = meta.pivot_table(index="case_ID", columns="sample_type", values=metric)
        wide = wide.dropna()
        out[metric] = float(friedmanchisquare(wide["A"], wide["T"], wide["S"])[1])
    return out


# ---------------- 圖 ----------------

def fig16a_component_oncoplot(dataloader):
    ATS = dataloader.get_ATS_subset()
    cm = dataloader.cm
    snv = ATS.SNV.add_freq().sort_features("freq", ascending=False)
    cat = ["sample_type", "sex", "smoke"]
    cmap_dict = {"sample_type": cm.get_cmap("sample_type", alpha=0.6),
                 "sex": cm.get_cmap("sex", alpha=0.7), "smoke": cm.get_cmap("smoke", alpha=0.7)}
    (OncoPlot(snv.head(30))
     .set_config(categorical_columns=cat, figsize=(30, 10), width_ratios=[25, 1, 0, 3])
     .mutation_heatmap().plot_freq().plot_bar()
     .plot_categorical_metadata(cmap_dict=cmap_dict).plot_all_legends().add_xticklabel()
     .save(str(OUT_DIR / "fig16a_component_oncoplot.tiff"), dpi=300))


def fig16b_signature(dataloader):
    ATS = dataloader.get_ATS_subset()
    cm = dataloader.cm
    cat = ["sample_type", "sex", "smoke"]
    cmap_dict = {"sample_type": cm.get_cmap("sample_type", alpha=0.6),
                 "sex": cm.get_cmap("sex", alpha=0.7), "smoke": cm.get_cmap("smoke", alpha=0.7)}
    (OncoPlot(ATS.signature)
     .set_config(figsize=(30, 10), categorical_columns=cat, width_ratios=[25, 1, 0.5, 1])
     .numeric_heatmap(cmap="Blues").plot_categorical_metadata(cmap_dict=cmap_dict)
     .plot_all_legends().add_xticklabel()
     .save(str(OUT_DIR / "fig16b_signature.tiff"), dpi=300))


def fig16c_metrics_boxplot(dataloader):
    ATS = dataloader.get_ATS_subset()
    cm = dataloader.cm
    snv = ATS.SNV.add_freq()
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()
    for i, col in enumerate(METRICS):
        snv.plot.plot_boxplot_with_annot(
            group_col="sample_type", test_col=col, test="Wilcoxon",
            palette=cm.get_cmap("sample_type", alpha=0.6), order=["A", "T", "S"],
            ax=axs[i], fontsize=16, is_paired=True, pair_col="case_ID")
    fig.savefig(OUT_DIR / "fig16c_metrics_boxplot.tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def fig17_driver_oncoplot(dataloader):
    ATS = dataloader.get_ATS_subset()
    cm = dataloader.cm
    snv = ATS.SNV.add_freq()
    mt = MAF.read_csv(AST_MAF).to_mutation_table()
    subset = mt.subset(features=mt.index.str.contains(r"^(TP53|EGFR|RBM10)\|", regex=True),
                       samples=snv.columns)
    subset.sample_metadata = snv.sample_metadata
    cat = ["sample_type", "sex", "smoke"]
    cmap_dict = {"sample_type": cm.get_cmap("sample_type", alpha=0.6),
                 "sex": cm.get_cmap("sex", alpha=0.7), "smoke": cm.get_cmap("smoke", alpha=0.7)}
    plot = (OncoPlot(subset.add_freq())
            .set_config(categorical_columns=cat, figsize=(30, 15), width_ratios=[25, 1, 0, 3])
            .mutation_heatmap().add_xticklabel().plot_freq()
            .plot_categorical_metadata(cmap_dict=cmap_dict).plot_all_legends())
    plot.ax_bar.set_axis_off()
    plot.save(str(OUT_DIR / "fig17_driver_gene_oncoplot.tiff"), dpi=300)


def fig18_shared_private_tables(dataloader):
    """shared / A-private / S-private 突變表(freq>0.1 oncoplot)+ 11 例 Venn。"""
    ATS = dataloader.get_ATS_subset()
    case_IDs = ATS.SNV.add_freq().sample_metadata.case_ID.unique()
    mt = MAF.read_csv(AST_MAF).to_mutation_table()

    A_dfs, S_dfs, AST_dfs = [], [], []
    fig, axs = plt.subplots(2, 6, figsize=(15, 6))
    axs = axs.flatten()
    for ind, cid in enumerate(case_IDs):
        A = {m for m in _find_mutations(mt.loc[:, f"{cid}_A"])}
        S = {m for m in _find_mutations(mt.loc[:, f"{cid}_S"])}
        T = {m for m in _find_mutations(mt.loc[:, f"{cid}_T"])}
        venn3([A, T, S], set_labels=("A", "T", "S"), ax=axs[ind])
        axs[ind].set_title(cid)
        as_shared = A & S
        A_dfs.append(pd.DataFrame(True, index=list(A - as_shared), columns=[cid]))
        S_dfs.append(pd.DataFrame(True, index=list(S - as_shared), columns=[cid]))
        AST_dfs.append(pd.DataFrame(True, index=list(A & S & T), columns=[cid]))
    for i in range(len(case_IDs), len(axs)):
        axs[i].axis("off")
    fig.suptitle("Venn of all mutations in 11 ASC cases", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig18_venn.tiff", dpi=200, bbox_inches="tight")
    plt.close(fig)

    def to_table(dfs):
        merged = pd.concat(dfs, axis=1).fillna(False)
        return PivotTable(merged).add_freq().sort_features()

    for name, dfs in [("A_private", A_dfs), ("S_private", S_dfs), ("AST_shared", AST_dfs)]:
        table = to_table(dfs).filter_by_freq(0.1)
        plot = (OncoPlot(table).set_config(figsize=(20, 10), width_ratios=[20, 1, 0, 2])
                .mutation_heatmap(cmap_dict={True: "gray", False: "white"})
                .add_xticklabel().plot_freq().plot_all_legends(fontsize=12))
        plot.ax_bar.set_axis_off()
        plot.save(str(OUT_DIR / f"fig18_{name}_table.tiff"), dpi=300)


def fig19_pyclone_heatmap(dataloader):
    """11 例 PyClone CCF 熱圖,每例左側 mean-CCF strip + 右側 sample×mutation 熱圖。"""
    ATS = dataloader.get_ATS_subset()
    case_IDs = ATS.SNV.sample_metadata.case_ID.unique()
    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(2, 6, wspace=0.18, hspace=0.28, figure=fig)
    for i, ax_spec in enumerate(outer):
        if i >= len(case_IDs):
            fig.add_subplot(ax_spec).axis("off")
            continue
        cid = case_IDs[i]
        df = pd.read_csv(os.path.join(PYCLONE_DIR, f"{cid}.pyclone.tsv"), sep="\t")
        ccf = df.pivot(index="mutation_id", columns="sample_id", values="cellular_prevalence")
        table = PivotTable(ccf)
        table.feature_metadata["mean_ccf"] = df.groupby("mutation_id")["cellular_prevalence"].mean()
        table.feature_metadata["cluster"] = df.groupby("mutation_id")["cluster_id"].first()
        order = table.feature_metadata.groupby("cluster")["mean_ccf"].mean().sort_values(ascending=False).index
        fm = (table.feature_metadata.set_index("cluster", append=True)
              .reorder_levels(["cluster", "mutation_id"])
              .sort_values(by=["cluster", "mean_ccf"],
                           key=lambda c: pd.Categorical(c, categories=order, ordered=True)
                           if c.name == "cluster" else c, ascending=[True, False]).reset_index())
        ts = table.subset(features=fm.mutation_id)
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_spec,
                                                 width_ratios=[0.1, 1], wspace=0.05)
        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1])
        sns.heatmap(pd.DataFrame(fm["mean_ccf"].values.reshape(-1, 1), columns=["mean_ccf"]),
                    ax=ax0, cmap="Reds", vmin=0, vmax=1, cbar=False, yticklabels=False)
        sns.heatmap(ts, ax=ax1, cmap="Blues", vmin=0, vmax=1, cbar=False, yticklabels=False)
        ax1.set_title(cid, fontsize=12)
        ax1.set_xlabel(""); ax1.set_ylabel("")
    fig.subplots_adjust(right=0.9)
    fig.savefig(OUT_DIR / "fig19_pyclone_heatmap.tiff", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    dataloader = ASCDataLoader()

    print("[figs] Fig16a/b/c, Fig17, Fig18, Fig19 ...")
    fig16a_component_oncoplot(dataloader)
    fig16b_signature(dataloader)
    fig16c_metrics_boxplot(dataloader)
    fig17_driver_oncoplot(dataloader)
    fig18_shared_private_tables(dataloader)
    fig19_pyclone_heatmap(dataloader)

    print("\n========== §3.2.1 recurrent 突變(變異層級,≥2 cases)==========")
    rec = compute_recurrent_mutations(dataloader)
    print(f"  EGFR L858R shared        : {list(rec['egfr_shared'].values())} cases  (論文 5)")
    print(f"  RPS3A missense A-private : {list(rec['rps3a_a_private'].values())} cases  (論文 2)")
    print(f"  HLA-DQB1 missense S-priv : {list(rec['hla_dqb1_s_private'].values())} cases  (論文 2)")

    print("\n========== §3.2.1 成分層級指標(Friedman paired)==========")
    met = compute_component_metrics(dataloader)
    for k, p in met.items():
        print(f"  {k:8s} p={p:.3f} {'(ns)' if p > 0.05 else '(SIG!)'}")
    print("  (論文:A/T/S 間皆無顯著差異)")
    print(f"\n[OK] 圖輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
