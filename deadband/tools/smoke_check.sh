#!/usr/bin/env bash
# Quick verification that the environment can reproduce the paper pipeline.
#   1. patched ANDES/AMS import and expose the deadband extensions
#   2. OPF dispatch generation matches the archived inputs numerically
# Run from the repository root with the venv active:  bash tools/smoke_check.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python - <<'EOF'
from importlib.metadata import version
import andes, ams  # noqa: F401

print("andes:", version("andes"))
print("ams (ltbams):", version("ltbams"))

sys_ = andes.System()
pvd1 = sys_.PVD1
for attr in ("fdbd", "fdbdu", "Tfdb"):
    assert hasattr(pvd1, attr), f"PVD1 missing '{attr}' - is the andes patch applied?"
assert hasattr(sys_.ESD1, "fdbdu"), "ESD1 missing 'fdbdu' - is the andes patch applied?"
print("PASS: patched PVD1/ESD1 deadband extensions present (fdbd, fdbdu, Tfdb)")
EOF

python "$ROOT/tools/check_dispatch_regen.py"
echo "Smoke check complete."
