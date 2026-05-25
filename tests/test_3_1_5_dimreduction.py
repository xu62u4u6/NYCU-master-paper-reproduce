"""論文 §3.1.5(降維 PCA/CCA)的關鍵數字鎖定。

對應分析腳本:analysis/dimension_reduction.py
鎖定 CCA 第一典型相關 = 0.756。
"""

import pytest

from asc.cohort_io import ASCDataLoader
from dimension_reduction import compute_cca


def test_cca_first_canonical_correlation():
    dl = ASCDataLoader(clustering_path="data/clustering/best_k_cluster_labels.csv")
    r1 = compute_cca(dl, save_fig=False)
    # 論文:SNV 與 CNV-cluster(各 PCA→20 維)的第一典型相關為 0.756
    assert r1 == pytest.approx(0.756, abs=0.005)
