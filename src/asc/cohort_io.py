"""ASC downstream data layer.

Ported from paper/utils/setup_env.py — the canonical entry point used by every
result notebook (result3.1.1 ... result3.3). Behaviour is kept identical so the
notebooks' standard header works unchanged:

    from asc.cohort_io import ASCDataLoader
    dataloader = ASCDataLoader(project_dir)
    cohort = dataloader.cohort
    cm = dataloader.cm
    fm = dataloader.fm

Two changes from the original for portability:
  1. project_dir is no longer a hard-coded module constant; it defaults to the
     ASC_PROJECT_DIR env var (falling back to the known lab path) and can be
     passed explicitly.
  2. The `sys.path.append("pymaftools")` hack is removed — pymaftools is
     installed (editable) into the pinned uv environment.

Source of truth for the cohort is NOT data/ASC_cohort.db (a snapshot) but the
constituent tables rebuilt here from data/CNV_0407, data/WES, data/signature,
plus metadata (LUSC "Subtype B" filtered out).
"""

import os

import numpy as np
import pandas as pd

from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.SmallVariationTable import SmallVariationTable
from pymaftools.core.SignatureTable import SignatureTable
from pymaftools.core.Cohort import Cohort

# Default project root: env override first, then the known lab path.
DEFAULT_PROJECT_DIR = os.environ.get(
    "ASC_PROJECT_DIR", "/home/data/data_dingyangliu/ASC_0217/"
)


class ASCDataLoader:
    def __init__(self, project_dir=DEFAULT_PROJECT_DIR, cohort_path=None, clustering_path=None):
        self.project_dir = project_dir
        self.clustering_path = clustering_path
        os.chdir(project_dir)
        print(f"[INFO] Working directory set to: {project_dir}")

        # Load metadata
        self.cosmic = self._load_cosmic()
        self.all_sample_metadata = self._load_all_metadata()
        self.AST_sample_metadata = self._load_AST_metadata()
        self.cm = self._default_cm()
        self.fm = self._default_fm()
        self.thresholded_cnv, self.gene_cnv, self.arm_cnv = self._load_cnv_table()
        self.snv_table = self._load_snv_table()
        self.signatrue_table = self._load_signature_table()
        self.signature_table = self.signatrue_table  # corrected alias (keeps original name too)
        self.cohort = self.to_cohort()
        sorted_snv = self.snv_table.sort_samples_by_group("subtype", ["LUAD", "ASC", "LUSC"])
        self.cohort = self.cohort.subset(samples=sorted_snv.columns.tolist())

    def _load_cosmic(self):
        path = os.path.join("data", "Census All Months Jan 27 2025.csv")
        cosmic = pd.read_csv(path).set_index("Gene Symbol")
        return cosmic

    def _load_all_metadata(self):
        path = os.path.join("data", "all_case_metadata_0527.csv")
        df = pd.read_csv(path)
        return df[df.LUSC_subtype != "Subtype B"]

    def _load_AST_metadata(self):
        path = os.path.join("data", "ASC_all_case_metadata_0312.csv")
        return pd.read_csv(path)

    def _load_cnv_table(self):
        # 原始資料以 PivotTable 格式儲存，需先讀取再轉換
        thresholded_cnv = CopyNumberVariationTable.read_sqlite("data/CNV_0407/thresholded_cnv.db")
        gene_cnv = CopyNumberVariationTable.read_sqlite("data/CNV_0407/gene_cnv.db")
        arm_cnv = CopyNumberVariationTable.read_sqlite("data/CNV_0407/arm_cnv.db")

        # add gene-level clustering
        if self.clustering_path is not None:
            self.clustering = pd.read_csv(self.clustering_path, index_col=0)
            gene_cnv.feature_metadata["cluster"] = self.clustering["cluster"]
        return thresholded_cnv, gene_cnv, arm_cnv

    def _load_snv_table(self):
        table = SmallVariationTable.read_sqlite("data/WES/all_case_table.db")
        return table

    def _load_signature_table(self):
        table = SignatureTable.read_sqlite("data/signature/signature_table.db")
        return table

    def get_ATS_subset(self) -> "Cohort":
        # Compute ATS sample list
        AS_cases = self.AST_sample_metadata.case_ID[self.AST_sample_metadata.AS_sample].values
        suffixes = np.tile(["_A", "_T", "_S"], len(AS_cases))
        repeated_cases = np.repeat(AS_cases, 3)
        AS_sample_IDs = (repeated_cases + suffixes).tolist()

        subseted_cohort = self.cohort.subset(samples=AS_sample_IDs)
        return subseted_cohort

    def to_cohort(self, name="ASC", description="ASC study using LUAD/LUSC/ASC samples"):
        cohort = Cohort(name, description)
        cohort.add_table(self.snv_table, "SNV")
        cohort.add_table(self.thresholded_cnv, "CNV_thresholded")
        cohort.add_table(self.gene_cnv, "CNV_gene")
        cohort.add_table(self.signatrue_table, "signature")
        cohort.add_table(self.arm_cnv, "CNV_arm")
        return cohort

    def setup_plotting(self, style="whitegrid", palette="Set2", figsize=(10, 6)):
        """設定繪圖環境"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use("default")
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["font.size"] = 12

        print(f"[INFO] 繪圖環境設定完成: style={style}, palette={palette}, figsize={figsize}")

    def _default_fm(self):
        from pymaftools.plot.FontManager import FontManager

        fm = FontManager()
        fm.register_fonts_from_directory(
            os.path.join(self.project_dir, "pymaftools", "pymaftools", "fonts")
        )
        fm.setup_matplotlib_fonts("Ubuntu", base_size=13)
        return fm

    def _default_cm(self):
        from pymaftools.plot.ColorManager import ColorManager

        cm = ColorManager()
        cm.add_cmap("subtype", {"LUAD": "orange", "ASC": "green", "LUSC": "blue"})
        cm.add_cmap("smoke", {1: "red", 0: "gray"})
        cm.add_cmap("sex", {"M": "blue", "F": "red"})
        cm.add_cmap("sample_type", {"A": "salmon", "T": "seagreen", "S": "steelblue"})
        return cm


if __name__ == "__main__":
    pass
