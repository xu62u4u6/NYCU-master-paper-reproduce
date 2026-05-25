"""論文 §3.1.1(突變圖譜)的可驗證數字鎖定。

對應分析腳本:analysis/mutation_profile.py
這裡只鎖「數字型」結果(不渲染圖,圖的產出由 analysis 腳本負責)。
"""

import pytest

from asc.cohort_io import ASCDataLoader
from pymaftools import MAF
from mutation_profile import compute_key_stats


@pytest.fixture(scope="module")
def stats():
    dl = ASCDataLoader()
    cohort_T = dl.cohort.subset(samples=dl.cohort.sample_metadata.sample_type == "T")
    SNV = cohort_T.SNV.add_freq()
    cohorts = {
        "LUAD": MAF.read_csv("data/WES/LUAD.maf", reindex=True),
        "ASC": MAF.read_csv("data/WES/ASC.maf", reindex=True),
        "LUSC": MAF.read_csv("data/WES/LUSC.maf", reindex=True),
    }
    return compute_key_stats(SNV, cohorts)


def test_rbm10_tp53_mutual_exclusivity(stats):
    # 論文:RBM10 與 TP53 近乎互斥,chi-square p=1.24e-4、odds ratio=0.0813
    assert stats["rbm10_tp53_p"] == pytest.approx(1.24e-4, rel=0.02)
    assert stats["rbm10_tp53_OR"] == pytest.approx(0.0813, abs=1e-3)
    assert stats["rbm10_tp53_OR"] < 1  # OR<1 表示互斥方向


def test_luad_egfr_hotspot_freq(stats):
    # 迴歸鎖定值(以目前 pinned 環境重現結果為基準):
    # LUAD 的 EGFR L858R 約 38.2%、exon19 deletion 約 34.8%
    assert stats["luad_L858R_freq"] == pytest.approx(0.382, abs=0.01)
    assert stats["luad_exon19del_freq"] == pytest.approx(0.348, abs=0.01)
