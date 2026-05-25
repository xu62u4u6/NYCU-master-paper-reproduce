# ASC 論文下游分析 — 可重現專案

論文:《肺腺鱗狀細胞癌的基因體景觀:與肺腺癌及鱗狀細胞癌的比較分析》(NYCU 碩論, 2025-10)
目標:讓論文**下游分析/圖表層**可被後續成員一鍵重現、可測試、可長期維護。

---

## 範圍與決策

- **範圍**:只做下游分析/圖表層。上游 WES pipeline(FASTQ→BAM→MAF/CNV)不在此重跑,只靠文件+版本紀錄。
- **目的**:內部長期可維護(非對外釋出)。
- **入口資料(上游已產出的中間產物,視為唯讀輸入)**:
  - `data/ASC_cohort.db`(74MB,pymaftools 序列化的 Cohort:SNV/INDEL + CNV + signature)
  - `data/processed_data/*.db`(`gene_cnv.db` / `arm_cnv.db` / `thresholded_cnv.db` / `signature_table.db` / `all_case_table.db`)
  - `data/clustering/cluster_table.db`
  - `data/all_case_metadata_0527.csv`(樣本 metadata)
- **PROJECT_DIR 慣例**:`/home/data/data_dingyangliu/ASC_0217`(與 `tmp/0525_fix_leakage/test_fs_leakage.py` 一致;與 `/home/dingyangliu/...` 為同一 inode)。

## 環境地基(待建)

- Python **3.12**(對齊 notebook metadata = 3.12.9;**注意**舊 conda 分析環境已刪、確切套件版本無法還原)。
- 用 **uv** 建 `pyproject.toml + uv.lock`,把 pymaftools 鬆綁的依賴(`pandas>2.0`、`numpy`…)**全部 pin 死**。
- pymaftools 以 editable 安裝。
- 策略:版本還原不了 → 用 pinned py3.12 新環境重跑,**把跑出來的數字當作新的「鎖定基線」**,README 註明可能與論文末位小數略有差異。
  - 版本穩健(幾乎不受影響):similarity、permutation test、PCA/CCA、TMB/Ti-Tv/MSI/HRD。
  - 版本敏感:RandomForest / StackingClassifier / RFECV(Table 1/2、soft score、feature importance)——但結論不會翻盤。

## ⚠️ 已知問題:result3.3 data leakage

`tmp/0525_fix_leakage/test_fs_leakage.py` 已證明 `paper/paper_scripts/result3.3.ipynb` 的 RFECV 是在**整個 cohort fit 一次**再做 CV → 特徵選擇洩漏。
→ 重構時保留兩版:`with-leakage`(重現論文原數字,標記 deprecated)與 `nested`(修正版,正式)。測試各鎖一份。

## 目標結構

```
reproduce/
  pyproject.toml + uv.lock     # py3.12 鎖死所有版本(含 pymaftools editable)
  src/asc/                     # 邏輯層(可 import、可測)
  analysis/                    # 薄呈現層,一支對一個 figure/table(承襲 test_fs_leakage.py 骨架)
  tests/test_key_results.py    # 鎖論文關鍵數字
  outputs/                     # 產出 CSV/figure
  run_all.sh
```

每支 `analysis/*.py` 骨架:設 `PROJECT_DIR` + `sys.path.insert` + 讀 db + 固定 seed(=42) + 輸出 CSV/figure + log。

## 結果 ↔ Notebook ↔ 輸入 對照表

| 論文結果 | 來源 notebook | 主要輸入 |
|---|---|---|
| Fig5/6 oncoplot + SMG(MutSigCV) | `result3.1.1.ipynb` | ASC_cohort.db |
| Fig7/8/9 genomic metrics / signature / HRD | `result3.1.2.ipynb` | ASC_cohort.db, signature_table.db |
| Fig10/11 CNV landscape + 114 clusters | `result3.1.3.ipynb` | gene_cnv.db, arm_cnv.db, cluster_table.db |
| Fig12/13/22/23 similarity | `result3.1.4_3.2.4.ipynb` | ASC_cohort.db, cnv dbs |
| Fig14/15/24 PCA/CCA | `result3.1.5_3.2.5_dimension_reduction.ipynb` | ASC_cohort.db, cnv dbs |
| Fig16–21 component-level | `result3.2.1/2/3.ipynb` | ASC_cohort.db, pyclone_output |
| **Table 1/2, Fig25–28 ML / soft score** | `result3.3.ipynb`, `result3.3-model.ipynb` | cnv dbs, cluster_table.db |

(對照表為初步推測,實作第 2 步會逐一核對 notebook 實際 import。)

## 關鍵數值錨點(寫進 test_key_results.py)

- Table 1:CNV-cluster ACC 0.953±0.036、AUC 0.992;STACK ACC 0.971、AUC 0.995
- Table 2:SNV-RFE–CNV-RFE ACC **0.976±0.025**、AUC **0.998±0.004**(最佳)
- held-out accuracy **0.962**(seed=42);stacking 權重 CNV 0.5321 / SNV 0.4679
- CNV:**114** clusters、**78** 顯著(q<0.1)、arm 顯著 **31**、cluster↔arm 重疊 **29**
- CCA 第一典型相關 **0.756**
- SMG:LUAD = TP53/EGFR/RBM10/OR10C1;ASC = EGFR/TP53;LUSC = TP53/OR14I1
- RBM10–TP53 互斥:chi-square p=1.24e-4, OR=0.0813
- Supp Table 1:3q KW q=1.18e-21(最顯著 arm);Supp Table 2:Cluster 47(3q,817 genes)q=1.91e-24

## 進度

- [x] 論文轉 md(`tmp/paper_20251021.md`)、通讀
- [x] 盤 code 結構、確認入口資料 = ASC_cohort.db + processed_data/*.db
- [x] 確認 py 版本 3.12.9;確認分析環境已刪、版本不可還原
- [x] 建立本資料夾骨架
- [x] 第 1 步(環境):建 py3.12 uv 環境 + `uv.lock`(sklearn 1.6.1 / pandas 2.3.3 / numpy 2.2.6 / scipy 1.15.3);pymaftools editable;`Cohort.read_sqlite("data/ASC_cohort.db")` 驗證可載入(222 樣本:LUAD 89 / LUSC 82 / ASC 51;表:SNV, CNV_gene, CNV_arm, CNV_thresholded, signature)
- [ ] 第 1 步(續):挑一支 notebook(建議 result3.1.2)實際跑通驗證
- [ ] 第 2 步:逐一核對 notebook import,完成對照表
- [ ] 第 3 步:抽 src/asc/ 模組 + analysis/*.py
- [ ] 第 4 步:test_key_results.py(leakage 版 + nested 版)
- [ ] 第 5 步:run_all.sh + 完善 README(db checksum、一鍵指令)
