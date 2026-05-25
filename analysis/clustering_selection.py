"""CNV gene 分群的最佳群數 (k) 選擇(對應論文 §3.1.3 / Supp Fig 3)。

來源 notebook:paper/paper_scripts/result3.1.3.ipynb

⚠️ 重要:完整的分群評估非常昂貴
  - k_fold_clustering_evaluation(k=2..200):約 3 小時
  - 各 k 的 silhouette / ARI 指標計算:約 1.5 小時
  因此這支腳本「預設不重算」,而是讀取既有的快取產物:
    data/clustering/metric_df.csv            各 k 的分群品質指標
    data/clustering/best_k_cluster_labels.csv  最佳 k 下每個基因的 cluster 標籤
  正常流程只做:讀 metric_df → 找最佳 k(=114)→ 畫 Supp Fig 3。

  若真要從頭重算,呼叫 recompute_clustering()(見下方,需數小時),
  它會重新產生上述快取檔。這屬於「一次性前處理」,不在測試/日常重現範圍內。

執行(在 reproduce/ 下):
    PYTHONPATH=src uv run python analysis/clustering_selection.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無視窗環境
import pandas as pd  # noqa: E402

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))

# ⚠️ pymaftools 必須在 chdir 到專案目錄「之前」import,
# 否則 cwd 下的 pymaftools/ 資料夾會遮蔽已安裝的套件(namespace 衝突)。
from pymaftools import plot_clustering_metrics_and_find_best_k  # noqa: E402

DEFAULT_PROJECT_DIR = os.environ.get("ASC_PROJECT_DIR", "/home/data/data_dingyangliu/ASC_0217/")
OUT_DIR = REPRO_DIR / "outputs" / "clustering_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_CSV = "data/clustering/metric_df.csv"
LABELS_CSV = "data/clustering/best_k_cluster_labels.csv"


def compute_best_k(project_dir=DEFAULT_PROJECT_DIR, save_fig=True):
    """從快取的 metric_df 找最佳 k,並(可選)畫 Supp Fig 3。

    Returns
    -------
    best_k : int          最佳群數(論文 114)
    n_clusters : int      best_k_cluster_labels.csv 實際的 cluster 數(應 == best_k)
    n_genes : int         參與分群的基因數(25,988)
    """
    os.chdir(project_dir)

    metric_df = pd.read_csv(METRIC_CSV, index_col=0)
    # 各 k 的 5-fold 平均 silhouette(notebook CELL 5 的前處理)
    fold_cols = metric_df.columns[metric_df.columns.str.startswith("fold")]
    metric_df["5fold_mean_silhouette"] = metric_df[fold_cols].mean(axis=1)

    fig_path = str(OUT_DIR / "suppfig3_clustering_metrics.tiff") if save_fig else "/tmp/_k.tiff"
    best_k = plot_clustering_metrics_and_find_best_k(metric_df, fig_path, transparent=False)

    labels = pd.read_csv(LABELS_CSV, index_col=0)
    return int(best_k), int(labels["cluster"].nunique()), int(len(labels))


def recompute_clustering(project_dir=DEFAULT_PROJECT_DIR):
    """⚠️ 重型前處理(數小時):從原始 CNV gene 表重新做分群評估並覆寫快取。

    平常不要呼叫。保留此函式是為了讓「快取如何產生」可被完整追溯與重現。
    內容對應 notebook CELL 1–8:
      1. 取 LUAD+LUSC 的 gene-level CNV 表
      2. k_fold_clustering_evaluation(k=2..200) → cluster_labels_*.json
      3. 每個 k 算 majority-vote silhouette / ARI → metric_df.csv
      4. plot_clustering_metrics_and_find_best_k → best_k
      5. 以 best_k 重新 AgglomerativeClustering(cosine, average)→ best_k_cluster_labels.csv
    （此處不實作為可一鍵執行,以免誤觸數小時運算;需要時請依 notebook CELL 1–8 還原。）
    """
    raise NotImplementedError(
        "重型前處理(~4.5 小時)。請依 result3.1.3.ipynb CELL 1–8 重算;"
        "日常重現請直接使用 data/clustering/ 下的快取產物。"
    )


def main():
    best_k, n_clusters, n_genes = compute_best_k()
    print("\n========== §3.1.3 CNV gene clustering 選 k ==========")
    print(f"參與分群基因數 : {n_genes}     (論文 25,988)")
    print(f"最佳群數 best_k : {best_k}        (論文 114)")
    print(f"cluster 標籤群數: {n_clusters}        (應 == best_k)")
    print(f"\n[OK] Supp Fig 3 輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
