"""論文 §3.1.3(CNV gene 分群選 k)的關鍵數字鎖定。

對應分析腳本:analysis/clustering_selection.py
從快取的 metric_df 找最佳 k(不重跑數小時的分群評估)。
"""

from clustering_selection import compute_best_k


def test_best_k_is_114():
    best_k, n_clusters, n_genes = compute_best_k(save_fig=False)
    assert n_genes == 25988      # 參與分群的基因數
    assert best_k == 114         # 論文選出的最佳群數
    assert n_clusters == 114     # 快取標籤的實際群數應一致
