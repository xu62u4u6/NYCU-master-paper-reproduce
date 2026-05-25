"""論文 §3.1.4 / §3.2.4(相似度分析)的關鍵結論鎖定。

對應分析腳本:analysis/similarity_analysis.py
鎖定「結論型」結果:ASC 較接近 LUAD、CNV cluster 層級不顯著、within > across。
"""

import pytest

from asc.cohort_io import ASCDataLoader
from similarity_analysis import compute_subtype_similarity, compute_component_within_across


@pytest.fixture(scope="module")
def dataloader():
    return ASCDataLoader(clustering_path="data/clustering/best_k_cluster_labels.csv")


@pytest.fixture(scope="module")
def subtype(dataloader):
    return compute_subtype_similarity(dataloader, save_fig=False)


@pytest.fixture(scope="module")
def component(dataloader):
    return compute_component_within_across(dataloader, save_fig=False)


def test_asc_closer_to_luad_all_modalities(subtype):
    # 論文 Supp Fig4:所有 modality 下 ASC 都較接近 LUAD(LUAD–ASC > ASC–LUSC)
    for name, r in subtype.items():
        assert r["asc_closer_to_luad"], f"{name} 方向不符"


def test_significant_modalities(subtype):
    # 論文:SNV SMC/Jaccard、CNV gene/arm 顯著
    for name in ["SNV_SMC", "SNV_Jaccard", "CNV_gene", "CNV_arm"]:
        assert subtype[name]["p"] < 0.05, f"{name} 應顯著"


def test_cluster_level_not_significant(subtype):
    # 論文特別指出:CNV cluster 層級「無顯著差異」(focal 事件稀釋了訊號)
    assert subtype["CNV_cluster"]["p"] > 0.05


def test_within_case_higher_than_across(component):
    # 論文 Fig23:within-case 顯著高於所有 across-case 群(p<0.0001)
    for name, r in component.items():
        for grp in ["A-across", "T-across", "S-across"]:
            assert r[f"within_vs_{grp}"] < 1e-4, f"{name} within>{grp} 不顯著"
