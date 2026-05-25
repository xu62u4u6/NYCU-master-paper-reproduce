"""論文 §3.1.2 / Fig10、Fig11、Supp Table 1、Supp Table 2 的關鍵數字鎖定。

對應分析腳本:analysis/cnv_landscape_clustering.py
這支測試把「亞型層級 CNV 顯著性」與「Fig11 Venn」的論文數字寫成斷言,
任何改動只要破壞重現性,pytest 就會立刻失敗。

註:compute() 會對 25,988 個基因跑 Kruskal–Wallis,約需數十秒。
用 module-scope fixture 讓整支測試只計算一次。
"""

import pytest

from cnv_landscape_clustering import compute


@pytest.fixture(scope="module")
def results():
    """整個測試模組只跑一次完整計算。"""
    return compute()


# ---------- 亞型層級 CNV 顯著性(q<0.1)----------

def test_arm_level_significant_count(results):
    # 論文:44 個 arm 中有 31 個顯著
    assert len(results["arm_res"]) == 44
    assert len(results["sig_arm"]) == 31


def test_gene_level_significant_count(results):
    # 論文:25,988 個基因中有 17,566 個顯著
    assert len(results["gene_res"]) == 25988
    assert len(results["sig_gene"]) == 17566


def test_cluster_level_significant_count(results):
    # 論文:114 個 CNV cluster 中有 78 個顯著
    assert len(results["cluster_res"]) == 114
    assert len(results["sig_cluster"]) == 78


# ---------- Fig 11:cluster-arm 與 arm 層級顯著 arm 的 Venn ----------

def test_venn_set_sizes(results):
    # 論文:cluster 涉及 36 arm、arm 層級 31 arm、交集 29
    assert len(results["sig_cluster_arm_set"]) == 36
    assert len(results["sig_arm_set"]) == 31
    assert len(results["common"]) == 29


def test_venn_only_arm_members(results):
    # 論文:僅在 arm 層級顯著(cluster 層級沒抓到)的是 15p、22p
    assert set(results["only_arm"]) == {"15p", "22p"}


# ---------- Supp Table 2:最顯著的 cluster 應為 3q 的 Cluster 47 ----------

def test_top_cluster_is_3q_cluster47(results):
    top = results["sig_cluster"].head(1).iloc[0]
    assert str(top["feature"]) == "47"          # Cluster 47
    assert "3q" in top["unique_chr_arm"]          # 位於 3q
    assert int(top["features_count"]) == 817      # 論文:817 個基因
