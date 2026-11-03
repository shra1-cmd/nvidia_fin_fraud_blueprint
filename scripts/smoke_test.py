#!/usr/bin/env python3
# Acceptance test for a running fraud API. Standard library only, so it runs with the server's
# system python3:   python3 scripts/smoke_test.py [--url http://127.0.0.1:8080] [--bench 200]
#
# Checks, in order (exit code 1 on any failure):
#   1. /health and /ready answer, /v1/model reports the expected model and threshold
#   2. parity: every golden transaction scores the same P(fraud) as the notebook scripts did on the
#      training laptop (tests/golden/expected.json), single and batch, within the tolerance
#   3. bad input is rejected with 422 (not 500)
#   4. --bench N: N sequential /v1/score calls, prints p50 / p95 / max latency
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8080")
parser.add_argument("--bench", type=int, default=100, help="latency benchmark requests (0 = skip)")
args = parser.parse_args()

with open(os.path.join(ROOT, "tests/golden/expected.json")) as f:
    golden = json.load(f)
with open(os.path.join(ROOT, "artifacts/manifest.json")) as f:
    manifest = json.load(f)
TOL = golden["tolerance"]
failures = []


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(args.url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


print(f"1. service at {args.url}")
try:
    check(call("GET", "/health") == (200, {"status": "ok"}), "/health")
except OSError as e:
    sys.exit(f"  cannot reach {args.url}: {e}")
status, body = call("GET", "/ready")
check(status == 200, f"/ready -> {status} {body}")
status, info = call("GET", "/v1/model")
check(status == 200 and info.get("model_version") == manifest["model_version"],
      f"/v1/model version {info.get('model_version')}")
print(f"       GNN on {info.get('gnn_device')}, XGBoost on {info.get('xgb_device')}, "
      f"threshold {info.get('threshold')}, context graph {info.get('context_graph')}")

print(f"\n2. parity with the notebook scripts (|diff| <= {TOL})")
for case in golden["single"]:
    req, exp = case["request"], case["expected"]
    status, body = call("POST", "/v1/score", req)
    if status != 200:
        check(False, f"{req['transaction_id']}: HTTP {status} {body}")
        continue
    r = body["result"]
    diff = abs(r["p_fraud"] - exp["p_fraud"])
    same_meta = (r["card_known"], r["card_history_tx"], r["unseen"]) == \
                (exp["card_known"], exp["card_history_tx"], exp["unseen"])
    check(diff <= TOL and same_meta and r["transaction_id"] == req["transaction_id"],
          f"{req['transaction_id']:16s} p={r['p_fraud']:.6f} expected {exp['p_fraud']:.6f} "
          f"(diff {diff:.1e}) {r['label']:5s} actual {case['actual_label'] or '-'}  "
          f"card_known={r['card_known']} history={r['card_history_tx']} unseen={r['unseen']}")

status, body = call("POST", "/v1/score/batch", {"transactions": golden["batch"]["requests"]})
if status != 200:
    check(False, f"batch: HTTP {status} {body}")
else:
    got = [r["p_fraud"] for r in body["results"]]
    worst = max(abs(g - e) for g, e in zip(got, golden["batch"]["p_fraud"]))
    check(len(got) == len(golden["batch"]["p_fraud"]) and worst <= TOL,
          f"batch of {len(got)}: max diff {worst:.1e} vs predict_new_tx_np.py, flagged {body['flagged']}, "
          f"{body['latency_ms']} ms")

print("\n3. input validation")
bad = dict(golden["single"][0]["request"], use_chip="Telepathy")
check(call("POST", "/v1/score", bad)[0] == 422, "unknown use_chip -> 422")
bad = dict(golden["single"][0]["request"], card=42)
check(call("POST", "/v1/score", bad)[0] == 422, "card out of range -> 422")
bad = dict(golden["single"][0]["request"], amount="lots")
check(call("POST", "/v1/score", bad)[0] == 422, "non-numeric amount -> 422")
check(call("POST", "/v1/score/batch", {"transactions": []})[0] == 422, "empty batch -> 422")

if args.bench:
    print(f"\n4. latency: {args.bench} sequential /v1/score calls")
    req = golden["single"][0]["request"]
    wall = []
    for _ in range(args.bench):
        t0 = time.perf_counter()
        call("POST", "/v1/score", req)
        wall.append((time.perf_counter() - t0) * 1000)
    wall.sort()
    print(f"       client round trip  p50 {statistics.median(wall):.1f} ms  "
          f"p95 {wall[int(0.95 * (len(wall) - 1))]:.1f} ms  max {wall[-1]:.1f} ms")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
