#!/usr/bin/env bash
# Set up a from-scratch environment that reproduces the deadband paper runs.
#
# Creates ./.venv, installs pinned dependencies, then installs the *patched*
# development versions of ANDES and AMS that the paper used:
#   - ANDES  CURENT/andes @ eda5163c9  + patches/andes-eda5163c9-deadband.patch
#       (PVD1/ESD1: symmetric upper deadband `fdbdu`, droop-output lag `Tfdb`,
#        available-power limit `pavail0`)
#   - AMS    CURENT/ams   @ 38325a1c   + patches/ams-38325a1c-compat.patch
#       (compatibility shims for the ANDES 2.0 API)
#
# Usage:  bash env/setup_env.sh        (from the repository root)
# Python: 3.12 recommended (3.12.13 was used for the archived results).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.12}"
ANDES_COMMIT=eda5163c9ee8d19945a1dd5d1771fec5da608c27
AMS_COMMIT=38325a1c310f0fff20d9db3495653652fcc669ba

command -v "$PYTHON" >/dev/null || { echo "error: $PYTHON not found (set PYTHON=...)"; exit 1; }

"$PYTHON" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT/env/requirements.txt"

mkdir -p "$ROOT/vendor"

if [ ! -d "$ROOT/vendor/andes" ]; then
    git clone https://github.com/CURENT/andes.git "$ROOT/vendor/andes"
fi
git -C "$ROOT/vendor/andes" checkout "$ANDES_COMMIT"
git -C "$ROOT/vendor/andes" apply --check "$ROOT/env/patches/andes-eda5163c9-deadband.patch" 2>/dev/null \
    && git -C "$ROOT/vendor/andes" apply "$ROOT/env/patches/andes-eda5163c9-deadband.patch" \
    || echo "andes patch already applied; skipping"
pip install -e "$ROOT/vendor/andes"

if [ ! -d "$ROOT/vendor/ams" ]; then
    git clone https://github.com/CURENT/ams.git "$ROOT/vendor/ams"
fi
git -C "$ROOT/vendor/ams" checkout "$AMS_COMMIT"
git -C "$ROOT/vendor/ams" apply --check "$ROOT/env/patches/ams-38325a1c-compat.patch" 2>/dev/null \
    && git -C "$ROOT/vendor/ams" apply "$ROOT/env/patches/ams-38325a1c-compat.patch" \
    || echo "ams patch already applied; skipping"
pip install -e "$ROOT/vendor/ams"

echo
echo "Environment ready. Verify with:"
echo "  source .venv/bin/activate && bash tools/smoke_check.sh"
