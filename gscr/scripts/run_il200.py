from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "il200"
sys.path.insert(0, str(ROOT / "src"))

from il200_gscr.experiments import run_experiments


if __name__ == "__main__":
    result = run_experiments(CASE_ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
