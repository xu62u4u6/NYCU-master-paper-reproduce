"""亞型層級 CNV 地景與分群分析(對應論文 Fig 10、Fig 11、Supp Table 1、Supp Table 2)。

來源 notebook:paper/paper_scripts/result3.1.2.ipynb(已去除其中被棄用的 ANOVA 探索碼,
只保留論文真正採用的 Kruskal–Wallis 路徑)。

本腳本做的事:
  1. 用 ASCDataLoader 載入 cohort,並只取 subtype-level 的 T 樣本(LUAD/ASC/LUSC 共 200 例)。
  2. 取得三種解析度的 CNV 表:gene-level、arm-level、cluster-level(114 個 CNV cluster)。
  3. 對每一層級做 Kruskal–Wallis + pairwise Mann–Whitney + FDR,統計 q<0.1 的顯著數。
     → 論文關鍵數字:arm 31、gene 17,566/25,988、cluster 78/114。
  4. 計算 cluster 層級涉及的 arm 與 arm 層級顯著 arm 的交集(Venn),對應 Fig 11(重疊 29)。
  5. 輸出 Supp Table 1(arm)、Supp Table 2(top20 cluster)到 outputs/。

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/cnv_landscape_clustering.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- 路徑設定:把 reproduce/src 加進 import path,讓 `import asc...` 可用 ---
REPRO_DIR = Path(__file__).resolve().parent.parent       # reproduce/
sys.path.insert(0, str(REPRO_DIR / "src"))

from asc.cohort_io import ASCDataLoader  # noqa: E402
from asc.stats import kruskal_with_pairwise_utest  # noqa: E402

# --- 輸出資料夾 ---
OUT_DIR = REPRO_DIR / "outputs" / "cnv_landscape_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CNV cluster 需要 gene→cluster 的對應表才能組出 cluster-level 表
CLUSTERING_PATH = "data/clustering/best_k_cluster_labels.csv"


def chr_sort_key(arm):
    """讓染色體 arm 依 1,2,...,22,X,Y 的自然順序排序(供 Venn 顯示用)。"""
    m = re.match(r"(\d+|X|Y)([pq]?)", arm)
    if not m:
        return (999, arm)
    chr_part, arm_part = m.groups()
    if chr_part.isdigit():
        chr_num = int(chr_part)
    elif chr_part == "X":
        chr_num = 23
    elif chr_part == "Y":
        chr_num = 24
    else:
        chr_num = 999
    return (chr_num, arm_part)


def build_cnv_tables(cohort_T):
    """從 T-only cohort 取出 gene / arm / cluster 三種解析度的 CNV 表。"""
    cnv_gene = cohort_T.CNV_gene
    cnv_arm = cohort_T.CNV_arm

    # cluster-level:把同一群(co-amplified/co-deleted)的基因聚成一個 CNV block
    cnv_cluster = cohort_T.CNV_gene.to_cluster_table()
    # 統一 index 型別為字串,避免後續 merge / 篩選時型別不一致
    cnv_cluster.index = cnv_cluster.index.astype(str)
    cnv_cluster.feature_metadata.index = cnv_cluster.feature_metadata.index.astype(str)
    return cnv_gene, cnv_arm, cnv_cluster


def main():
    # === 1. 載入資料 ===
    # clustering_path 一定要給,否則 to_cluster_table() 無法取得 cluster 標籤
    dataloader = ASCDataLoader(clustering_path=CLUSTERING_PATH)
    cohort = dataloader.cohort

    # subtype-level 分析只用混合腫瘤(T)樣本:LUAD 89 + ASC(T)29 + LUSC 82 = 200
    cohort_T = cohort.subset(samples=cohort.sample_metadata.sample_type == "T")

    # === 2. 三種解析度的 CNV 表 ===
    cnv_gene, cnv_arm, cnv_cluster = build_cnv_tables(cohort_T)
    subtype = cnv_arm.sample_metadata.subtype  # 三個表的樣本一致,共用同一份標籤

    # === 3. 各層級的 Kruskal–Wallis 顯著性 ===
    arm_res = kruskal_with_pairwise_utest(cnv_arm, subtype)
    gene_res = kruskal_with_pairwise_utest(cnv_gene, subtype)
    cluster_res = kruskal_with_pairwise_utest(cnv_cluster, subtype)
    # 把 cluster 的染色體 arm / 基因清單等 metadata 併回來(供 Supp Table 2 與 Venn 用)
    cluster_res = pd.merge(
        cluster_res, cnv_cluster.feature_metadata,
        left_on="feature", right_index=True, how="left",
    )

    # q<0.1 視為顯著(與論文一致)
    sig_arm = arm_res[arm_res["KW_qval"] < 0.1]
    sig_gene = gene_res[gene_res["KW_qval"] < 0.1]
    sig_cluster = cluster_res[cluster_res["KW_qval"] < 0.1]

    print("\n========== 亞型層級 CNV 顯著性(q<0.1)==========")
    print(f"arm     : {len(sig_arm):>5d} / {len(arm_res):<5d}  (論文 31)")
    print(f"gene    : {len(sig_gene):>5d} / {len(gene_res):<5d}  (論文 17566 / 25988)")
    print(f"cluster : {len(sig_cluster):>5d} / {len(cluster_res):<5d}  (論文 78 / 114)")

    # === 4. Venn:cluster 涉及的 arm vs arm 層級顯著 arm(Fig 11)===
    sig_arm_set = set(sig_arm.feature)
    sig_cluster_arm_set = set()
    for arms in sig_cluster.unique_chr_arm.values:
        sig_cluster_arm_set |= set(arms)

    common = sorted(sig_cluster_arm_set & sig_arm_set, key=chr_sort_key)
    only_cluster = sorted(sig_cluster_arm_set - sig_arm_set, key=chr_sort_key)
    only_arm = sorted(sig_arm_set - sig_cluster_arm_set, key=chr_sort_key)

    print("\n========== Fig 11 Venn(arm 集合)==========")
    print(f"cluster 涉及的顯著 arm 數: {len(sig_cluster_arm_set)}  (論文 36)")
    print(f"arm 層級顯著 arm 數      : {len(sig_arm_set)}  (論文 31)")
    print(f"兩者交集(common)        : {len(common)}  (論文 29)")
    print(f"  common      : {common}")
    print(f"  only_cluster: {only_cluster}")
    print(f"  only_arm    : {only_arm}")

    # === 5. 輸出表格 ===
    # Supp Table 1:arm 層級顯著結果(依 q 值排序)
    arm_cols = ["feature", "KW_qval", "LUAD_vs_ASC_qval", "LUAD_vs_LUSC_qval", "ASC_vs_LUSC_qval"]
    sig_arm.loc[:, arm_cols].to_csv(OUT_DIR / "supp_table1_arm_features.csv", index=False)

    # Supp Table 2:cluster 層級 top20(依 q 值排序),含所在 arm 與基因數
    top20 = sig_cluster.head(20).loc[
        :, ["feature", "unique_chr_arm", "features_count", "KW_qval"]
    ]
    top20.to_csv(OUT_DIR / "supp_table2_top20_cluster_features.csv", index=False)

    print(f"\n[OK] 表格輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
