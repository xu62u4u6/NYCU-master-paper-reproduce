"""亞型層級突變圖譜、基因體指標、signature、DDR(對應論文 §3.1.1,Fig5–9 + Supp Fig2)。

來源 notebook:paper/paper_scripts/result3.1.1.ipynb(已去除大量探索性/重複 cell,
只保留進入論文的圖與數字)。

產出的圖(存到 reproduce/outputs/mutation_profile/):
  fig5_subtype_oncoplot.tiff     亞型 SNV/INDEL oncoplot(freq>0.1)
  fig6_{gene}_lollipop.tiff      TP53 / EGFR / RBM10 蛋白層級 lollipop
  fig7_metrics_boxplot.tiff      MSI / TMB / Ti-Tv / HRD across subtypes
  fig8_signature_subtypes.tiff   COSMIC SBS signature 比例 boxplot
  fig8b_SBS_heatmap.tiff         signature heatmap
  fig9_smoking_TMB.tiff          各亞型抽菸 vs 非抽菸 TMB
  fig9_DDR_genes.tiff / fig9_DDR_pathways.tiff   DDR 基因/路徑 oncoplot
  suppfig2_ASC_smoke_oncoplot.tiff               ASC 依抽菸狀態的 oncoplot

可測數字(compute_key_stats):
  RBM10–TP53 互斥:chi2 p=1.24e-4、OR=0.0813
  LUAD 的 EGFR L858R / exon19del 樣本比例

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/mutation_profile.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))

from asc.cohort_io import ASCDataLoader  # noqa: E402

# pymaftools 的繪圖類別(與 notebook 用法一致)
from pymaftools import MAF, OncoPlot, LollipopPlot  # noqa: E402

OUT_DIR = REPRO_DIR / "outputs" / "mutation_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENES = ["TP53", "EGFR", "RBM10"]
GENE_DOMAINS = {
    "EGFR": {
        "Recep_L_domain": "Receptor L domain",
        "FU": "Furin-like domain",
        "Furin-like": "Furin-like domain",
        "TM_ErbB1": "Transmembrane domain of ErbB1 (EGFR)",
        "PTKc_EGFR": "Protein tyrosine kinase (EGFR catalytic domain)",
        "TyrKc": "Tyrosine kinase domain",
    },
    "TP53": {
        "P53_TAD": "p53 Transactivation domain",
        "P53": "p53 DNA-binding domain (core DBD)",
        "P53_tetramer": "p53 Tetramerization domain",
    },
    "RBM10": {
        "COG0724": "COG0724 conserved domain (splicing-related)",
        "RRM": "RNA recognition motif",
        "ZnF_RBZ": "Zinc finger RNA-binding domain (RBZ)",
        "G_patch": "G-patch RNA-binding domain",
    },
}


def load_snv(dataloader):
    """取 subtype-level(T 樣本)的 SNV 表並加上頻率欄位。"""
    cohort_T = dataloader.cohort.subset(
        samples=dataloader.cohort.sample_metadata.sample_type == "T"
    )
    return cohort_T, cohort_T.SNV.add_freq()


# ---------------- Fig 5:亞型 SNV oncoplot ----------------

def fig5_subtype_oncoplot(SNV, cm, freq=0.1):
    # 取突變頻率 > freq 的基因,依頻率排序,樣本依亞型分組
    table = (SNV.filter_by_freq(freq)
                .sort_features(by="freq")
                .sort_samples_by_group(group_col="subtype",
                                       group_order=["LUAD", "ASC", "LUSC"], top=10))
    # 各亞型分別的突變頻率(供右側 freq bar 用)
    table = table.add_freq(groups={
        "LUAD": table.subset(samples=table.sample_metadata.subtype == "LUAD"),
        "ASC": table.subset(samples=table.sample_metadata.subtype == "ASC"),
        "LUSC": table.subset(samples=table.sample_metadata.subtype == "LUSC"),
    })
    cat = ["subtype", "sex", "smoke"]
    cmap_dict = {k: cm.get_cmap(k, alpha=0.7) for k in cat}
    (OncoPlot(table)
     .set_config(categorical_columns=cat, figsize=(30, 14),
                 width_ratios=[25, 3, 0, 2], ytick_fontsize=12)
     .mutation_heatmap(linewidths=0.5)
     .plot_freq(freq_columns=["freq", "LUAD_freq", "ASC_freq", "LUSC_freq"],
                linewidths=0.5, xtick_fontsize=12)
     .plot_bar()
     .plot_categorical_metadata(cmap_dict=cmap_dict, linewidths=0.5)
     .plot_all_legends()
     .save(str(OUT_DIR / "fig5_subtype_oncoplot.tiff"), dpi=300))


# ---------------- Fig 6:driver gene lollipop ----------------

def fig6_lollipops():
    sns.set_style("ticks")
    LUAD = MAF.read_csv("data/WES/LUAD.maf", reindex=True)
    ASC = MAF.read_csv("data/WES/ASC.maf", reindex=True)
    LUSC = MAF.read_csv("data/WES/LUSC.maf", reindex=True)
    cohorts = {"LUAD": LUAD, "ASC": ASC, "LUSC": LUSC}

    for gene in GENES:
        cohorts_data = {}
        for name, maf in cohorts.items():
            try:
                AA_length, mut = maf.get_protein_info(gene)
                domains, refseq = MAF.get_domain_info(gene, AA_length)
                cohorts_data[name] = (AA_length, mut, domains, refseq)
            except Exception as e:  # 某些 cohort 可能缺該基因資訊
                print(f"  [warn] {gene} in {name}: {e}")
        if cohorts_data:
            LollipopPlot.plot_multi_cohort(
                gene=gene, cohorts_data=cohorts_data, figsize=(15, 6),
                width_ratios=[8, 2], save_path=str(OUT_DIR / f"fig6_{gene}_lollipop.tiff"),
                dpi=300, domain_label_map=GENE_DOMAINS[gene],
            )
            plt.close("all")
    return cohorts


# ---------------- Fig 7:基因體指標 boxplot ----------------

def fig7_metrics_boxplot(SNV, cm):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()
    for i, col in enumerate(["MSI", "TMB", "ti/tv", "HRD-sum"]):
        SNV.plot.plot_boxplot_with_annot(
            group_col="subtype", test_col=col,
            palette=cm.get_cmap("subtype", alpha=0.7),
            order=["LUAD", "ASC", "LUSC"], ax=axs[i], fontsize=16)
    fig.savefig(OUT_DIR / "fig7_metrics_boxplot.tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------- Fig 8:COSMIC signature ----------------

def fig8_signature(cohort_T, cm):
    signature = cohort_T.signature
    cat = ["subtype", "smoke", "sex"]
    cmap_dict = {k: cm.get_cmap(k, alpha=0.7) for k in cat}
    # 8b:signature heatmap
    (OncoPlot(signature)
     .set_config(figsize=(30, 10), categorical_columns=cat, width_ratios=[25, 1, 0.5, 1])
     .numeric_heatmap(cmap="Blues")
     .plot_categorical_metadata(cmap_dict=cmap_dict)
     .plot_all_legends()
     .save(str(OUT_DIR / "fig8b_SBS_heatmap.tiff"), dpi=300))
    # 8:各 SBS signature 的亞型 boxplot
    plot_data = pd.concat([pd.DataFrame(signature.T),
                           pd.DataFrame(signature.sample_metadata)], axis=1)
    fig, axs = plt.subplots(2, 5, figsize=(30, 12), sharex=True)
    axs = axs.flatten()
    for i, col in enumerate(signature.index):
        signature.plot.plot_boxplot_with_annot(
            data=plot_data, group_col="subtype", test_col=col,
            palette=cm.get_cmap("subtype", alpha=0.7),
            order=["LUAD", "ASC", "LUSC"], alpha=0.6, ax=axs[i], fontsize=14)
    fig.savefig(OUT_DIR / "fig8_signature_subtypes.tiff", dpi=300, bbox_inches="tight",
                format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------- Fig 9b:抽菸 TMB ----------------

def fig9_smoking_tmb(SNV, cm):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, subtype in enumerate(["LUAD", "ASC", "LUSC"]):
        subset = SNV.subset(samples=SNV.sample_metadata.subtype == subtype)
        subset.plot.plot_boxplot_with_annot(
            group_col="smoke", test_col="TMB",
            palette=cm.get_cmap("smoke", alpha=0.7), alpha=0.6, order=[1, 0],
            ax=axs[i], title=f"{subtype} TMB distribution by smoking")
    fig.savefig(OUT_DIR / "fig9_smoking_TMB.tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------- Fig 9a:DDR 基因/路徑 oncoplot ----------------

def fig9_ddr(SNV, cm):
    sns.set_style("ticks")
    df = pd.read_csv("data/ddr/ddr_genes.csv", skiprows=3, index_col=1)
    core = df.iloc[:, -9:].notna()
    core.columns = core.columns.str.replace(".1", "", regex=False)
    core = core[core.sum(axis=1) > 0]

    ddr_genes = SNV.reindex(core.index, fill_value=False)
    gene_to_pathway = core.apply(lambda c: c[c].index.tolist())
    ddr_pathways = SNV.reindex(core.columns, fill_value=False)
    for pathway, genes in gene_to_pathway.items():
        avail = [g for g in genes if g in ddr_genes.index]
        if avail:
            ddr_pathways.loc[pathway] = ddr_genes.loc[avail].any(axis=0)

    cm.add_cmap("binary", {True: "gray", False: "white"})
    ddr_genes = (ddr_genes.add_freq().sort_features(by="freq")
                 .sort_samples_by_group(group_col="subtype", group_order=["LUAD", "ASC", "LUSC"]))
    ddr_pathways = (ddr_pathways.add_freq().sort_features(by="freq")
                    .sort_samples_by_group(group_col="subtype", group_order=["LUAD", "ASC", "LUSC"]))
    cat = ["subtype", "sex", "smoke"]
    cmap_dict = {k: cm.get_cmap(k, alpha=0.7) for k in cat}
    (OncoPlot(ddr_genes)
     .set_config(categorical_columns=cat, figsize=(30, 15), width_ratios=[25, 1, 0, 3])
     .mutation_heatmap().plot_freq().plot_bar()
     .plot_categorical_metadata(cmap_dict=cmap_dict).plot_all_legends()
     .save(str(OUT_DIR / "fig9_DDR_genes.tiff"), dpi=300))
    (OncoPlot(ddr_pathways)
     .set_config(categorical_columns=cat, figsize=(30, 10), width_ratios=[25, 1, 0, 3])
     .mutation_heatmap(cmap_dict=cm.get_cmap("binary", alpha=0.7)).plot_freq().plot_bar()
     .plot_categorical_metadata(cmap_dict=cmap_dict).plot_all_legends()
     .save(str(OUT_DIR / "fig9_DDR_pathways.tiff"), dpi=300))


# ---------------- Supp Fig 2:ASC 依抽菸的 oncoplot ----------------

def suppfig2_asc_smoke(cohort_T, cm):
    sns.set_style("ticks")
    ASC_T = cohort_T.subset(samples=cohort_T.sample_metadata.subtype == "ASC")
    ASC_SNV = ASC_T.SNV.add_freq()
    table = (ASC_SNV.add_freq({
        "smoker": ASC_SNV.subset(samples=ASC_SNV.sample_metadata.smoke == 1),
        "non-smoker": ASC_SNV.subset(samples=ASC_SNV.sample_metadata.smoke == 0),
    }).sort_samples_by_group(group_col="smoke", group_order=[1, 0]).head(50))
    (OncoPlot(table)
     .set_config(categorical_columns=["smoke"], width_ratios=[25, 3, 0, 2])
     .mutation_heatmap()
     .plot_categorical_metadata(cmap_dict={"smoke": cm.get_cmap("smoke", alpha=0.7)})
     .plot_bar()
     .plot_freq(["freq", "smoker_freq", "non-smoker_freq"])
     .plot_all_legends().add_xticklabel()
     .save(str(OUT_DIR / "suppfig2_ASC_smoke_oncoplot.tiff"), dpi=300))


# ---------------- 可測數字 ----------------

def compute_key_stats(SNV, cohorts=None):
    """回傳 §3.1.1 的可驗證數字。"""
    # RBM10–TP53 互斥(chi-square + odds ratio)
    bt = pd.DataFrame(SNV.to_binary_table().subset(features=["RBM10", "TP53"]).T)
    ct = pd.crosstab(bt["RBM10"], bt["TP53"])
    chi2, p, _, _ = chi2_contingency(ct)
    a, b, c, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
    odds_ratio = (a * d) / (b * c)

    stats = {"rbm10_tp53_chi2": chi2, "rbm10_tp53_p": p, "rbm10_tp53_OR": odds_ratio}

    # LUAD 的 EGFR L858R / exon19del 樣本比例(需 MAF 檔)
    if cohorts is not None:
        LUAD, ASC, LUSC = cohorts["LUAD"], cohorts["ASC"], cohorts["LUSC"]
        all_case = MAF(pd.concat([LUAD, ASC, LUSC])).filter_maf(MAF.nonsynonymous_types)
        l858r = set(all_case.loc[all_case.HGVSp == "p.Leu858Arg"].sample_ID.unique())
        exon19 = set(all_case.loc[(all_case.Hugo_Symbol == "EGFR") &
                                  (all_case.EXON == "19/28") &
                                  (all_case.Variant_Type == "DEL")].sample_ID.unique())
        luad_ids = set(SNV.subset(samples=SNV.sample_metadata.subtype == "LUAD").columns)
        n_luad = len(luad_ids)
        stats["luad_L858R_freq"] = len(l858r & luad_ids) / n_luad
        stats["luad_exon19del_freq"] = len(exon19 & luad_ids) / n_luad
    return stats


def main():
    dataloader = ASCDataLoader()
    cohort_T, SNV = load_snv(dataloader)
    cm = dataloader.cm

    print("[1/8] Fig5 subtype oncoplot ...");  fig5_subtype_oncoplot(SNV, cm)
    print("[2/8] Fig6 lollipops ...");          cohorts = fig6_lollipops()
    print("[3/8] Fig7 metrics boxplot ...");    fig7_metrics_boxplot(SNV, cm)
    print("[4/8] Fig8 signature ...");          fig8_signature(cohort_T, cm)
    print("[5/8] Fig9 smoking TMB ...");        fig9_smoking_tmb(SNV, cm)
    print("[6/8] Fig9 DDR ...");                fig9_ddr(SNV, cm)
    print("[7/8] SuppFig2 ASC smoke ...");      suppfig2_asc_smoke(cohort_T, cm)

    print("[8/8] key stats ...")
    stats = compute_key_stats(SNV, cohorts)
    print(f"  RBM10-TP53: chi2={stats['rbm10_tp53_chi2']:.3f} "
          f"p={stats['rbm10_tp53_p']:.4g} OR={stats['rbm10_tp53_OR']:.4f}  (論文 p=1.24e-4, OR=0.0813)")
    print(f"  LUAD L858R freq    = {stats['luad_L858R_freq']:.4f}")
    print(f"  LUAD exon19del freq= {stats['luad_exon19del_freq']:.4f}")
    print(f"\n[OK] 圖輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
