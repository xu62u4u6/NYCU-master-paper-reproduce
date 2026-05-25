"""論文 §3.2.1(成分層級突變/克隆)的關鍵數字鎖定。

對應分析腳本:analysis/component_mutation.py
"""

import pytest

from asc.cohort_io import ASCDataLoader
from component_mutation import compute_recurrent_mutations, compute_component_metrics


@pytest.fixture(scope="module")
def dataloader():
    return ASCDataLoader()


def test_recurrent_shared_and_private(dataloader):
    rec = compute_recurrent_mutations(dataloader)
    # 論文:EGFR L858R 為 recurrent shared,出現於 5 個 case
    assert len(rec["egfr_shared"]) == 1
    assert list(rec["egfr_shared"].values())[0] == 5
    # EGFR L858R 在 GRCh38 對應位置 55191822
    assert "55191822" in list(rec["egfr_shared"].keys())[0]
    # 論文:RPS3A missense A-private、HLA-DQB1 missense S-private,各 2 個 case
    assert list(rec["rps3a_a_private"].values()) == [2]
    assert list(rec["hla_dqb1_s_private"].values()) == [2]


def test_component_metrics_no_significant_difference(dataloader):
    # 論文:A/T/S 三成分間 MSI/TMB/Ti-Tv/HRD 皆無顯著差異
    met = compute_component_metrics(dataloader)
    for metric, p in met.items():
        assert p > 0.05, f"{metric} 不應顯著 (p={p})"
