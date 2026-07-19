#!/usr/bin/env python3
"""Verify that OPF dispatch generation reproduces the archived inputs.

Regenerates the four hour-0 dispatches with prepare_day_dispatches.py and
compares pg/pd/vBus/obj against the archived JSONs in data/dispatches_day96.
On the environment that produced the paper results the match is exact
(max |diff| = 0.0); tiny solver-version differences (<1e-9 pu) are tolerated.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-9


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dispatch_regen_"))
    cmd = [
        sys.executable, str(ROOT / "scripts" / "prepare_day_dispatches.py"),
        "--opf-case", str(ROOT / "cases" /
                          "IL200_opf2_aligned_to_dyn_swap0_17_pvd005_esd001_storage5_slack47_pmax730_sn730.xlsx"),
        "--curve-file", str(ROOT / "cases" / "CurveInterp.csv"),
        "--results-dir", str(tmp),
        "--hour-start", "0", "--hours", "1",
        "--wind-pref-alpha", "0.98", "--solar-pref-alpha", "0.98",
        "--workers", "1",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    worst = 0.0
    for d in range(4):
        name = f"h0d{d}_dispatch.json"
        a = json.load(open(ROOT / "data" / "dispatches_day96" / name))
        b = json.load(open(tmp / name))
        for key in ("pg", "pd", "vBus"):
            diff = float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
            worst = max(worst, diff)
        worst = max(worst, abs(a["obj"] - b["obj"]) / max(abs(a["obj"]), 1.0))
        print(f"  {name}: OK (cumulative max diff {worst:.3e})")

    if worst > TOL:
        print(f"FAIL: max deviation {worst:.3e} exceeds tolerance {TOL:.0e}")
        return 1
    print(f"PASS: dispatch regeneration matches archived inputs (max diff {worst:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
