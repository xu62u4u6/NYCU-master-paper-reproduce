"""共用的統計檢定工具。

這裡放「跨多個結果章節都會用到」的統計函式。論文中亞型層級(LUAD/ASC/LUSC)
的多組比較一律採用無母數的 Kruskal–Wallis 檢定,再以 pairwise Mann–Whitney U
做兩兩比較,最後用 Benjamini–Hochberg(FDR)校正多重檢定。

這支模組原本散落在各 notebook(例如 result3.1.2 的 CELL 9),抽出來集中維護,
讓 arm / gene / cluster 各層級的 CNV 顯著性分析都呼叫同一份邏輯,確保一致。
"""

from itertools import combinations

import pandas as pd
import statsmodels.stats.multitest as smm
from scipy.stats import kruskal, mannwhitneyu


def kruskal_with_pairwise_utest(table, subtype, groups=("LUAD", "ASC", "LUSC"), fdr_alpha=0.05):
    """對 table 的每個 feature 做 Kruskal–Wallis,再做 pairwise Mann–Whitney U,並 FDR 校正。

    Parameters
    ----------
    table : PivotTable / DataFrame
        數值表,row = feature(基因/arm/cluster),col = sample。
    subtype : pd.Series
        每個 sample 的亞型標籤,index 必須與 table 的欄位對齊。
    groups : tuple
        要比較的亞型,預設 ("LUAD", "ASC", "LUSC")。順序只影響欄位命名,不影響統計。
    fdr_alpha : float
        保留參數(目前未在函式內用門檻過濾,交由呼叫端自行篩 q 值)。

    Returns
    -------
    pd.DataFrame
        每列一個 feature,欄位包含:
          - KW_stat / KW_pval / KW_qval:Kruskal–Wallis 統計量、原始 p、FDR 校正後 q
          - {g1}_vs_{g2}_stat / _pval / _qval:每對亞型的 Mann–Whitney 結果(亦各自 FDR 校正)
        並依 KW_qval 由小到大排序。
    """
    records = []

    for feature in table.index:
        # 依亞型把該 feature 的數值分成三組;dropna 以排除缺值樣本
        values = [table.loc[feature, subtype[subtype == g].index].dropna().values for g in groups]

        # 任一組樣本數 < 2 就跳過(無法做檢定)
        if any(len(v) < 2 for v in values):
            continue

        # 三組整體差異:Kruskal–Wallis
        kw_stat, kw_p = kruskal(*values)

        # 兩兩比較:對每一對亞型做雙尾 Mann–Whitney U
        pair_results = {}
        for g1, g2 in combinations(groups, 2):
            vals1 = table.loc[feature, subtype[subtype == g1].index].dropna().values
            vals2 = table.loc[feature, subtype[subtype == g2].index].dropna().values
            if len(vals1) > 1 and len(vals2) > 1:
                stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
                pair_results[f"{g1}_vs_{g2}_stat"] = stat
                pair_results[f"{g1}_vs_{g2}_pval"] = p
            else:
                pair_results[f"{g1}_vs_{g2}_stat"] = None
                pair_results[f"{g1}_vs_{g2}_pval"] = None

        record = {"feature": feature, "KW_stat": kw_stat, "KW_pval": kw_p}
        record.update(pair_results)
        records.append(record)

    df = pd.DataFrame(records)

    # 對 Kruskal–Wallis 的 p 值做 FDR(BH)校正
    df["KW_qval"] = smm.multipletests(df["KW_pval"], method="fdr_bh")[1]

    # 對每一對 pairwise 的 p 值各自做 FDR 校正
    pairwise_p_cols = [c for c in df.columns if c.endswith("_pval") and not c.startswith("KW")]
    for col in pairwise_p_cols:
        qvals = smm.multipletests(df[col].dropna(), method="fdr_bh")[1]
        df[col.replace("_pval", "_qval")] = qvals

    return df.sort_values("KW_qval")
