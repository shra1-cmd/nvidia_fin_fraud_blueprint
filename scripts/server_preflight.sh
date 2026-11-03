#!/usr/bin/env bash
# READ-ONLY checks on the server before building/running anything. Changes nothing, needs no sudo.
#   bash scripts/server_preflight.sh
# Prints the facts CLAUDE.md step 2 decides on: driver -> TORCH_INDEX, free GPU -> GPU_ID, port, disk.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

section() { printf '\n== %s\n' "$1"; }

section "host"
echo "$(hostname)  $(uname -srm)"
grep PRETTY_NAME /etc/os-release 2>/dev/null

section "GPUs (pick ONE idle index for GPU_ID)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,utilization.gpu --format=csv
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "driver: $DRIVER"
  echo "processes on GPUs:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  MAJOR=${DRIVER%%.*}
  if   [ "$MAJOR" -ge 525 ]; then echo "=> TORCH_INDEX=cu126 (driver >= 525)"
  elif [ "$MAJOR" -ge 450 ]; then echo "=> TORCH_INDEX=cu118 (driver $DRIVER < 525; XGBoost will likely run on CPU)"
  else echo "=> driver $DRIVER is too old for any supported build: STOP and tell the user (never upgrade it yourself)"
  fi
else
  echo "nvidia-smi not found: STOP and tell the user"
fi

section "docker"
docker --version 2>&1
docker compose version 2>&1 || echo "no 'docker compose' plugin (use the docker run command in CLAUDE.md)"
docker info 2>/dev/null | grep -iE "runtimes|default runtime" || echo "cannot run 'docker info' (permission? user in docker group?)"
command -v nvidia-ctk >/dev/null && nvidia-ctk --version 2>&1 | head -1 || echo "nvidia-ctk not on PATH (toolkit may still be installed; step 4 tests GPU access)"

section "network (needed to build the image on the server)"
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/ https://registry-1.docker.io/v2/; do
  code=$(curl -sI -o /dev/null -m 8 -w '%{http_code}' "$u" 2>/dev/null) || code="$code (curl failed)"
  echo "$u -> $code"
done

section "port ${HOST_PORT:-8080}"
if command -v ss >/dev/null && ss -ltn | grep -q ":${HOST_PORT:-8080} "; then
  echo "IN USE - pick another HOST_PORT"; ss -ltnp 2>/dev/null | grep ":${HOST_PORT:-8080} "
else
  echo "free"
fi

section "disk (image ~6 GB)"
df -h . /var/lib/docker 2>/dev/null

section "artifacts"
if (cd artifacts && sha256sum -c --quiet SHA256SUMS); then echo "SHA256SUMS OK"; else echo "CHECKSUM MISMATCH: re-copy artifacts/ from the laptop"; fi
python3 -c "import json; m=json.load(open('artifacts/manifest.json')); print('model', m['model_version'], '| threshold', m['threshold'])"
