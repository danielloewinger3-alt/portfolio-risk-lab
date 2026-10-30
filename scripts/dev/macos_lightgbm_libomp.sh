#!/usr/bin/env bash
# macOS only: LightGBM's wheel links @rpath/libomp.dylib and expects Homebrew's libomp at
# /opt/homebrew/opt/libomp. When that is absent, point lib_lightgbm.dylib at the libomp that
# scikit-learn's macOS wheel already vendors (sklearn/.dylibs/libomp.dylib), then ad-hoc re-sign.
# Touches only files inside .venv; idempotent; a no-op on Linux or when lightgbm already imports.
# The clean alternative is `brew install libomp` with an arm64 Homebrew.
set -euo pipefail
[[ "$(uname -s)" == "Darwin" ]] || exit 0
PY="${1:-.venv/bin/python}"  # optional argument: another venv's python
[[ -x "$PY" ]] || { echo "no .venv yet; run uv sync first" >&2; exit 1; }
if "$PY" -c "import lightgbm" 2>/dev/null; then exit 0; fi
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
OMP="$SITE/sklearn/.dylibs/libomp.dylib"
[[ -f "$LIB" && -f "$OMP" ]] || { echo "lightgbm or sklearn libomp not found" >&2; exit 1; }
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" 2>/dev/null || true
codesign --force --sign - "$LIB" >/dev/null 2>&1
"$PY" -c "import lightgbm" && echo "lightgbm: linked to scikit-learn's vendored libomp"
