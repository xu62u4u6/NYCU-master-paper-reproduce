"""pytest 共用設定:把 reproduce/src 與 reproduce/analysis 加進 import path。

這樣測試可以直接 `from asc.stats import ...`、`import cnv_landscape_clustering`,
不需要把專案安裝成套件。
"""

import sys
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_DIR / "src"))
sys.path.insert(0, str(REPRO_DIR / "analysis"))
