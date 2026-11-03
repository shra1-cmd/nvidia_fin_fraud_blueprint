# deploy-fin_fraud: runbook for Claude on the GPU server

You are on the GPU server (V100s), working inside this folder. Your job is to **deploy and verify** an
inference-only fraud-scoring API that is already written and tested. Do not redesign or retrain it.

**What it is:** a FastAPI service in Docker. It takes one raw TabFormer card transaction as JSON and
returns `P(fraud)` plus FRAUD/LEGIT. The model pipeline is: clean → encode (fitted encoders) → attach
to a frozen context graph → GNN (2× SAGEConv, hidden 64) → `[38 raw features | 64-d embedding]` →
XGBoost → threshold 0.25.

**Status when it was handed to you:** the code is complete. On the training laptop (RTX 3050), the same
image passed `scripts/smoke_test.py`. Its outputs matched the original notebook scripts to within
1e-5, and it agreed with the offline metrics on the 25,803-transaction test graph. The only unknowns
are server-specific: driver, GPU choice, network, port.

## Hard rules

1. **Never touch the NVIDIA driver, the CUDA toolkit, nvidia-container-toolkit or kernel modules.**
   Don't install, upgrade or downgrade them, don't run `apt` for them, and don't reboot. If the
   driver doesn't fit, adapt the *image* (`TORCH_INDEX`, step 2). If that isn't possible, stop and
   tell the user.
2. **Use exactly one GPU**, pinned with `GPU_ID`. Never take a GPU that another process is using. If
   none is idle, ask the user which one to use.
3. **Don't modify `artifacts/`.** `SHA256SUMS` guards it. Don't retrain, refit encoders or edit
   `manifest.json`. Change the threshold only if the user asks, and then only through
   `FRAUD_THRESHOLD` in `.env`.
4. **Don't change the math in `app/pipeline.py`** (clean / encode / attach / GNN / XGBoost). The
   acceptance criterion is parity with the notebook scripts (`tests/golden/expected.json`). If parity
   fails, investigate and report. Never loosen the tolerance to make it pass.
5. **Keep the API private:** `BIND_ADDR=127.0.0.1`. Ask the user before binding to `0.0.0.0`,
   opening firewall ports, or adding a reverse proxy.
6. **Touch only this folder and the `fin-fraud-api` container/image.** Don't stop, remove or prune
   other people's containers, images or volumes (no `docker system prune`).
7. Ask before anything that needs `sudo`.

## Layout

| Path | What it is |
|---|---|
| `app/pipeline.py` | `FraudScorer`. Loads everything once, runs startup consistency checks and a warm-up, and scores transactions. |
| `app/main.py` | FastAPI app: `/health`, `/ready`, `/v1/model`, `/v1/score`, `/v1/score/batch`. |
| `app/schemas.py` | Request/response models. Raw TabFormer fields in snake_case. |
| `artifacts/` | Mounted **read-only** into the container. Model, encoders, frozen graph, `manifest.json`, `SHA256SUMS`. |
| `artifacts/manifest.json` | Model version, GNN hyperparameters (hidden 64), feature dims, threshold 0.25 and how it was chosen. |
| `scripts/server_preflight.sh` | Read-only facts: driver, GPUs, docker, network, port, disk, checksums. |
| `scripts/smoke_test.py` | Acceptance test (stdlib only): parity, validation, latency. |
| `scripts/choose_threshold.py`, `package_artifacts.sh`, `write_checksums.sh` | Laptop-side tools. **Not needed on the server.** |
| `tests/golden/expected.json` | Reference outputs of the notebook scripts for 8 transactions. |
| `Dockerfile`, `docker-compose.yml`, `.env.example` | The container. Torch's CUDA build is chosen by `TORCH_INDEX`. |

## Runbook

Do the steps in order. Each step has a check, so don't continue past a failed check.

### Step 1: verify the copy
```bash
cd artifacts && sha256sum -c SHA256SUMS && cd ..
```
Every line must say `OK`. If any fails, the copy is incomplete: ask the user to re-sync the folder.

### Step 2: preflight (read-only) and decisions
```bash
bash scripts/server_preflight.sh
```
Decide from its output:

| Fact | Decision |
|---|---|
| Driver ≥ 525.60 | `TORCH_INDEX=cu126` (default) |
| Driver 450.80 – 525 | `TORCH_INDEX=cu118`. XGBoost will probably fall back to CPU automatically, which is fine. |
| Driver < 450, or no `nvidia-smi` | **Stop** and report to the user |
| GPU list | `GPU_ID` = one V100 with ~0 MiB used, 0% utilisation and no processes. None idle → ask the user. |
| Port 8080 in use | Pick another free `HOST_PORT` |
| pypi / download.pytorch.org / docker.io unreachable | Use **Path B** in step 4 |
| `docker info` permission denied | Ask the user to add you to the `docker` group (don't use sudo yourself) |

Never pick `cu128` or newer for a V100. Those wheels dropped sm_70.

### Step 3: configure
```bash
cp .env.example .env
# edit .env: GPU_ID, TORCH_INDEX, HOST_PORT (leave BIND_ADDR=127.0.0.1, FRAUD_THRESHOLD empty)
```

### Step 4: get the image
**Path A (server has internet):**
```bash
docker compose build          # ~5-10 min, downloads torch (~2.5 GB)
```
**Path B (no internet):** ask the user to run this on the laptop, inside `deploy-fin_fraud/`:
```bash
docker compose build   # or: docker build --build-arg TORCH_INDEX=<what you chose> -t fin-fraud-api:1.0 .
docker save fin-fraud-api:1.0 | gzip > fin-fraud-api_1.0.tar.gz
scp fin-fraud-api_1.0.tar.gz <server>:<this folder>/
```
Then load it on the server: `docker load -i fin-fraud-api_1.0.tar.gz`.

**Check that the container can see the GPU, and that torch has kernels for it:**
```bash
docker run --rm --gpus "\"device=$(grep ^GPU_ID .env | cut -d= -f2)\"" fin-fraud-api:1.0 \
  python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```
It must print `Tesla V100…`, `(7, 0)` and an arch list that contains `sm_70`.
- `could not select device driver "nvidia"`: nvidia-container-toolkit is missing or not configured.
  **Stop** and report; it needs an admin.
- `sm_70` is not in the list: wrong `TORCH_INDEX`. Rebuild with cu126 or cu118.

### Step 5: run
```bash
docker compose up -d
docker compose logs -f fraud-api     # wait for "scorer ready" and "Application startup complete"
```
Startup takes roughly 15–60 s. If the container exits, read the log. An `ArtifactError` names exactly
which artifact or setting disagrees. Fix the cause, never the check.

If there's no `docker compose` plugin, the equivalent is:
```bash
docker run -d --name fin-fraud-api --restart unless-stopped --gpus "\"device=$GPU_ID\"" \
  -p 127.0.0.1:$HOST_PORT:8080 -v "$PWD/artifacts:/app/artifacts:ro" \
  -e XGB_DEVICE=auto fin-fraud-api:1.0
```

### Step 6: acceptance
```bash
python3 scripts/smoke_test.py --url http://127.0.0.1:<HOST_PORT> --bench 200
```
It must end with **`ALL CHECKS PASSED`**. Laptop baseline (RTX 3050): p50 ≈ 102 ms, and a V100 should
be similar or faster. Much slower (> 500 ms) usually means the GNN is running on CPU: check
`/v1/model`. Differences around 1e-5 are normal GPU noise. A parity
failure above the 0.002 tolerance means a real mismatch (wrong artifacts or library versions):
report it, don't work around it.

Also confirm the GPU pinning: `nvidia-smi` should show the python process on `GPU_ID` only.

### Step 7: report to the user
Reply with:
- GPU used (index, name), driver version, `TORCH_INDEX`, XGBoost device (from `/v1/model`)
- the full smoke-test output, including the latency p50/p95
- URL / port, and how to reach it from outside (SSH tunnel: `ssh -L 8080:127.0.0.1:<HOST_PORT> <server>`)
- anything that deviated from this runbook, and why

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ArtifactError: … has no kernels for Tesla V100 (sm_70)` | torch wheel without Volta support | Rebuild with `TORCH_INDEX=cu126` (or cu118) |
| `CUDA driver version is insufficient` | Wheel is newer than the driver | `TORCH_INDEX=cu118`. Still failing → stop and report. |
| Log: `XGBoost cannot use the GPU …; falling back to CPU` | xgboost wheel/driver can't use this GPU | Fine: same scores, still fast. Mention it in the report. |
| `ArtifactError: artifacts are inconsistent` | Partial/mixed copy of `artifacts/` | Re-run step 1. Ask the user to re-package on the laptop. |
| `ArtifactError: GNN checkpoint does not match manifest` | Weights and manifest come from different trainings | Ask the user. Don't edit the manifest yourself. |
| HTTP 422 | Bad input (unknown `use_chip`, card ∉ 0..8, non-numeric amount…) | Expected; the detail says which field |
| Port already allocated | Something else is on `HOST_PORT` | Change `HOST_PORT` in `.env`, then `docker compose up -d` |
| `permission denied … docker.sock` | User isn't in the docker group | Ask the user |

## Operating it

```bash
docker compose ps                       # status (healthy = /ready answers)
docker compose logs --tail 100 fraud-api
docker compose restart fraud-api
docker compose down                     # stop and remove the container (keeps image + artifacts)
curl -s 127.0.0.1:<HOST_PORT>/v1/model | python3 -m json.tool
```
- **New model version:** done on the laptop. Retrain, then `scripts/package_artifacts.sh`, then
  `choose_threshold.py --write`, then rsync `artifacts/` here. On the server: step 1, then
  `docker compose restart fraud-api`, then step 6. Code changes need `docker compose up -d --build`.
- **Different threshold (only when the user asks):** set `FRAUD_THRESHOLD=` in `.env`, then
  `docker compose up -d`.

## Deliberate design decisions (don't "improve" these without the user)

- **Frozen context graph.** New transactions attach to the held-out test graph (25,803 tx /
  4,795 cards / 6,773 merchants) for that request only, and are never added to it. Known cards bring
  their history. Unknown cards or merchants become new nodes with no history (`card_known: false`).
- **The graph is rebuilt on the GPU for every request.** This is simple and stateless. It costs
  ~100 ms per request, which the user accepted.
- **No Triton**: a plain API with one uvicorn worker. A lock serialises GPU passes.
- **SHAP only via `/v1/explain`** (added at the user's request, 2026-10-26): exact TreeSHAP of the
  XGBoost stage on a separate CPU booster (4 threads), computed after the unchanged scoring path, so
  `/v1/score` parity is untouched. Startup fails if contributions don't add up to the prediction.
  The dashboard at `/` (`app/static/index.html`, no external assets) calls it on the same origin.
- **Threshold 0.25.** It minimises 10 × missed frauds + false alarms on the test graph: recall 0.993,
  precision 0.853 (at 0.5: recall 0.975, precision 0.925). See `manifest.json` → `threshold_selection`.
- **`/v1/score/batch` scores all its transactions in one graph,** so same-card transactions in one
  batch see each other (the same behaviour as `notebooks/predict_new_tx_np.py`). For fully
  independent scores, call `/v1/score` once per transaction.
