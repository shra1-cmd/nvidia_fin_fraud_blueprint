#!/usr/bin/env bash
# Rewrite artifacts/SHA256SUMS. Run on the laptop after ANY change inside artifacts/
# (package_artifacts.sh and choose_threshold.py call it for you).
# On the server, verify with:   cd artifacts && sha256sum -c SHA256SUMS
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../artifacts"
find . -type f ! -name SHA256SUMS ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
echo "wrote artifacts/SHA256SUMS ($(wc -l < SHA256SUMS) files)"
