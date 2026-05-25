"""相似度分析(對應論文 §3.1.4 / §3.2.4,Fig12、Fig13、Fig22、Fig23、Supp Fig4)。

來源 notebook:paper/paper_scripts/result3.1.4_3.2.4.ipynb(去除大量重複/探索 cell)。

兩個層面:
  A. 亞型層級(subtype):比較 ASC 與 LUAD / LUSC 的相似度,判斷 ASC 較接近誰。
     5 種 modality:SNV SMC(hamming)、SNV Jaccard、CNV gene/cluster/arm(cosine)。
     對每個 modality 比較「LUAD–ASC」與「ASC–LUSC」相似度分布(Mann–Whitney)。
     論文結論:除 CNV cluster 外皆顯著,且 ASC 一致地較接近 LUAD(Supp Fig4)。
  B. 成分層級(component / ATS):同一病例 within-case vs 不同病例 across-case 相似度。
     論文結論:within-case 顯著高於 across-case(Fig23,p<0.0001)。

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/similarity_analysis.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))

from pymaftools import SimilarityMatrix  # noqa: E402  (import 先於 chdir)
from asc.cohort_io import ASCDataLoader  # noqa: E402

OUT_DIR = REPRO_DIR / "outputs" / "similarity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERING_PATH = "data/clustering/best_k_cluster_labels.csv"

# (modality 顯示名稱, 相似度 method)
SUBTYPE_SPECS = [
    ("SNV_SMC", "hamming"),
    ("SNV_Jaccard", "jaccard"),
    ("CNV_gene", "cosine"),
    ("CNV_cluster", "cosine"),
    ("CNV_arm", "cosine"),
]


def _build_subtype_tables(cohort):
    """T-only 的 5 種 modality 表。"""
    T = cohort.subset(samples=cohort.sample_metadata.sample_type == "T")
    snv005 = T.SNV.add_freq().filter_by_freq(0.05).to_binary_table()
    cnv_gene, cnv_arm = T.CNV_gene, T.CNV_arm
    cnv_cluster = T.CNV_gene.to_cluster_table()
    cnv_cluster.index = cnv_cluster.index.astype(str)
    cnv_cluster.feature_metadata.index = cnv_cluster.feature_metadata.index.astype(str)
    return {
        "SNV_SMC": snv005, "SNV_Jaccard": snv005,
        "CNV_gene": cnv_gene, "CNV_cluster": cnv_cluster, "CNV_arm": cnv_arm,
    }


def compute_subtype_similarity(dataloader, save_fig=True):
    """對 5 種 modality 算 LUAD–ASC vs ASC–LUSC 的相似度與 Mann–Whitney p。

    Returns
    -------
    dict[str, dict]:每個 modality -> {mean_luad_asc, mean_asc_lusc, p, asc_closer_to_luad}
    """
    cohort = dataloader.cohort
    cm = dataloader.cm
    tables = _build_subtype_tables(cohort)

    results = {}
    box_rows = []  # 供箱型圖
    for name, method in SUBTYPE_SPECS:
        tbl = tables[name]
        r = SimilarityMatrix.analyze_similarity(
            table=tbl, method=method, title=name,
            groups=tbl.sample_metadata.subtype, group_order=["LUAD", "ASC", "LUSC"],
            group_cmap=cm.get_cmap("subtype", alpha=0.8),
            save_dir=str(OUT_DIR), file_format="tiff",
            utest_group_pairs=[("LUAD", "ASC"), ("ASC", "LUSC")],
        )
        m1 = float(np.mean(r["pair1"].values))  # LUAD–ASC
        m2 = float(np.mean(r["pair2"].values))  # ASC–LUSC
        results[name] = {
            "mean_luad_asc": m1, "mean_asc_lusc": m2,
            "p": float(r["pairwise_utest_p"]), "asc_closer_to_luad": m1 > m2,
        }
        for v in r["pair1"].values.flatten():
            box_rows.append({"modality": name, "pair": "LUAD–ASC", "similarity": v})
        for v in r["pair2"].values.flatten():
            box_rows.append({"modality": name, "pair": "ASC–LUSC", "similarity": v})

    if save_fig:
        df = pd.DataFrame(box_rows)
        g = sns.catplot(data=df, x="pair", y="similarity", col="modality", kind="box",
                        col_wrap=3, height=3.5, sharey=False,
                        palette={"LUAD–ASC": "orange", "ASC–LUSC": "blue"}, hue="pair")
        g.figure.savefig(OUT_DIR / "fig_supp4_subtype_similarity_boxplots.tiff",
                         dpi=200, bbox_inches="tight")
        plt.close("all")
    return results


# ---------------- 成分層級:within vs across-case ----------------

def _within_across_records(matrix, method):
    """回傳 (within, A-across, T-across, S-across) 的相似度長表。"""
    sim = matrix.compute_similarity(method=method)
    case_IDs = matrix.sample_metadata.case_ID.unique()
    rec = []
    for cid in case_IDs:
        for a, b in [("A", "T"), ("A", "S"), ("T", "S")]:
            rec.append((sim.loc[f"{cid}_{a}", f"{cid}_{b}"], "within"))
        for other in [c for c in case_IDs if c != cid]:
            rec.append((sim.loc[f"{cid}_A", f"{other}_A"], "A-across"))
            rec.append((sim.loc[f"{cid}_T", f"{other}_T"], "T-across"))
            rec.append((sim.loc[f"{cid}_S", f"{other}_S"], "S-across"))
    return pd.DataFrame(rec, columns=["similarity", "group"])


def compute_component_within_across(dataloader, save_fig=True):
    """成分層級 within-case vs across-case 相似度(Fig23)。

    Returns
    -------
    dict[str, dict]:每個 modality -> {within_vs_A/T/S_p}(within 對三種 across 的 Mann–Whitney)
    """
    ATS = dataloader.get_ATS_subset()
    snv005 = ATS.SNV.to_binary_table().add_freq().filter_by_freq(0.05)
    cnv_arm = ATS.CNV_arm
    cnv_cluster = ATS.CNV_gene.to_cluster_table()
    cnv_cluster.index = cnv_cluster.index.astype(str)
    cnv_cluster.feature_metadata.index = cnv_cluster.feature_metadata.index.astype(str)

    specs = [("SNV_Jaccard", snv005, "jaccard"), ("SNV_SMC", snv005, "hamming"),
             ("CNV_cluster", cnv_cluster, "cosine"), ("CNV_arm", cnv_arm, "cosine")]

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for (name, mat, method), ax in zip(specs, axes.flatten()):
        df = _within_across_records(mat, method)
        within = df[df.group == "within"].similarity
        ps = {}
        for grp in ["A-across", "T-across", "S-across"]:
            other = df[df.group == grp].similarity
            ps[f"within_vs_{grp}"] = float(mannwhitneyu(within, other, alternative="greater")[1])
        ps["within_mean"] = float(within.mean())
        results[name] = ps
        sns.boxplot(data=df, x="group", y="similarity", ax=ax, palette="Set2",
                    order=["within", "A-across", "T-across", "S-across"])
        ax.set_title(name, fontsize=10)
    if save_fig:
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig23_within_vs_across_similarity.tiff", dpi=200,
                    bbox_inches="tight")
    plt.close("all")
    return results


def main():
    dataloader = ASCDataLoader(clustering_path=CLUSTERING_PATH)

    print("\n========== §3.1.4 亞型層級相似度(LUAD–ASC vs ASC–LUSC)==========")
    sub = compute_subtype_similarity(dataloader)
    for name, r in sub.items():
        sig = "SIG" if r["p"] < 0.05 else "ns "
        arrow = "ASC→LUAD" if r["asc_closer_to_luad"] else "ASC→LUSC"
        print(f"  {name:12s} LUAD-ASC={r['mean_luad_asc']:.4f} "
              f"ASC-LUSC={r['mean_asc_lusc']:.4f} p={r['p']:.3e} {sig} {arrow}")
    print("  (論文:除 CNV_cluster 外皆顯著,且皆 ASC→LUAD)")

    print("\n========== §3.2.4 成分層級 within vs across-case ==========")
    comp = compute_component_within_across(dataloader)
    for name, r in comp.items():
        print(f"  {name:12s} within_mean={r['within_mean']:.4f} "
              f"p(within>A/T/S-across)={r['within_vs_A-across']:.1e}/"
              f"{r['within_vs_T-across']:.1e}/{r['within_vs_S-across']:.1e}")
    print(f"\n[OK] 圖輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
