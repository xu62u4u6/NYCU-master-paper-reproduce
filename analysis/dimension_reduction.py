"""降維分析 PCA / CCA(對應論文 §3.1.5 / §3.2.5,Fig14、Fig15、Fig24)。

來源 notebook:paper/paper_scripts/result3.1.5_3.2.5_dimension_reduction.ipynb

內容:
  Fig14:亞型層級 PCA(SNV freq0.05、CNV gene/cluster/arm),依 subtype 著色。
  Fig15:CCA — SNV 與 CNV-cluster 各自 PCA 降到 20 維後做 CCA,
         第一典型相關 = 0.756(論文),散佈圖依 subtype 著色。
  Fig24:成分層級 PCA(ATS),依 case_ID 著色、sample_type 形狀。

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/dimension_reduction.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cross_decomposition import CCA  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))

from asc.cohort_io import ASCDataLoader  # noqa: E402

OUT_DIR = REPRO_DIR / "outputs" / "dimension_reduction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERING_PATH = "data/clustering/best_k_cluster_labels.csv"


def _tables(cohort):
    snv = cohort.SNV.add_freq().to_binary_table()
    cnv_gene = cohort.CNV_gene
    cnv_arm = cohort.CNV_arm
    cnv_cluster = cohort.CNV_gene.to_cluster_table()
    cnv_cluster.index = cnv_cluster.index.astype(str)
    cnv_cluster.feature_metadata.index = cnv_cluster.feature_metadata.index.astype(str)
    return snv, cnv_gene, cnv_arm, cnv_cluster


# ---------------- Fig15:CCA ----------------

def compute_cca(dataloader, save_fig=True):
    """SNV(freq0.05)與 CNV-cluster 各自 PCA→20 維後做 CCA(n=1)。

    Returns
    -------
    r1 : float   第一典型相關(論文 0.756)
    """
    cohort = dataloader.cohort
    cm = dataloader.cm
    T = cohort.subset(samples=cohort.sample_metadata.sample_type == "T")
    snv, _, _, cnv_cluster = _tables(T)

    snv_005 = snv.filter_by_freq(0.05)
    X = snv_005.to_binary_table().T.values.astype(float)   # samples x SNV features
    Y = cnv_cluster.T.values.astype(float)                 # samples x cluster features

    # 各自 PCA 降到 20 維(random_state=0 與 notebook 一致)
    Xp = PCA(n_components=20, random_state=0).fit_transform(X)
    Yp = PCA(n_components=20, random_state=0).fit_transform(Y)

    snv_c, cnv_c = CCA(n_components=1, scale=False).fit_transform(Xp, Yp)
    r1 = float(np.corrcoef(snv_c[:, 0], cnv_c[:, 0])[0, 1])

    if save_fig:
        plot_df = pd.DataFrame({
            "snv_c": snv_c[:, 0], "cnv_c": cnv_c[:, 0],
            "subtype": T.SNV.sample_metadata.subtype.values,
        })
        plt.figure(figsize=(7, 5))
        sns.scatterplot(plot_df, x="snv_c", y="cnv_c", hue="subtype",
                        palette=cm.get_cmap("subtype"))
        plt.xlabel("Canonical var 1 (SNV)")
        plt.ylabel("Canonical var 1 (CNV)")
        plt.title(f"CCA: SNV vs CNV-cluster  (r1={r1:.3f})")
        plt.grid(True)
        plt.savefig(OUT_DIR / "fig15_CCA.tiff", dpi=300,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close("all")
    return r1


# ---------------- Fig14:亞型 PCA ----------------

def fig14_subtype_pca(dataloader):
    cohort = dataloader.cohort
    cm = dataloader.cm
    T = cohort.subset(samples=cohort.sample_metadata.sample_type == "T")
    snv, cnv_gene, cnv_arm, cnv_cluster = _tables(T)
    specs = [("SNV_005", snv.add_freq().filter_by_freq(0.05)),
             ("CNV_gene", cnv_gene), ("CNV_cluster", cnv_cluster), ("CNV_arm", cnv_arm)]
    for title, table in specs:
        table.plot.plot_pca_samples(
            figsize=(10, 6), color_col="subtype", palette=cm.get_cmap("subtype"),
            title=None, legend_item_spacing=0.04, legend_group_spacing=0.08, dpi=300, s=20,
            save_path=str(OUT_DIR / f"fig14_subtype_{title}.tiff"))
        plt.close("all")


# ---------------- Fig24:成分 PCA ----------------

def fig24_component_pca(dataloader):
    ATS = dataloader.get_ATS_subset()
    ats_snv, ats_gene, ats_arm, ats_cluster = _tables(ATS)
    case_ids = ats_snv.sample_metadata["case_ID"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(case_ids)))
    case_colors = {cid: mcolors.to_hex(c) for cid, c in zip(case_ids, colors)}
    specs = [("SNV_005", ats_snv.filter_by_freq(0.05)),
             ("CNV_gene", ats_gene), ("CNV_cluster", ats_cluster), ("CNV_arm", ats_arm)]
    for title, table in specs:
        table.plot.plot_pca_samples(
            figsize=(10, 6), shape_col="sample_type", color_col="case_ID",
            palette=case_colors, title=None, legend_item_spacing=0.04,
            legend_group_spacing=0.08, dpi=300, s=20,
            save_path=str(OUT_DIR / f"fig24_component_{title}.tiff"))
        plt.close("all")


def main():
    dataloader = ASCDataLoader(clustering_path=CLUSTERING_PATH)
    print("[1/3] Fig14 subtype PCA ...");    fig14_subtype_pca(dataloader)
    print("[2/3] Fig24 component PCA ...");  fig24_component_pca(dataloader)
    print("[3/3] Fig15 CCA ...")
    r1 = compute_cca(dataloader)
    print(f"\nCCA first canonical correlation = {r1:.4f}  (論文 0.756)")
    print(f"\n[OK] 圖輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
