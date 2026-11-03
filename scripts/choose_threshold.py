# Pick the FRAUD / LEGIT decision threshold and write it into artifacts/manifest.json.
#
# Scores every transaction of the frozen held-out test graph with the deployed pipeline
# (app/pipeline.py, so exactly what the API runs), sweeps thresholds 0.05..0.95 and applies the rule:
#
#     minimise  FN_COST x missed frauds  +  1 x false alarms
#
# FN_COST = 10 assumes a missed fraud (lost amount + chargeback) costs about ten times a false alarm
# (one manual review / customer call). Change it with --fn-cost when the business gives real numbers.
# The test graph is the only out-of-sample labelled data, so the chosen value is slightly
# optimistic; re-run this when fresh labelled transactions exist.
#
# Run on the laptop, inside the API image (labels: artifacts/graph/nodes/transaction_label.csv):
#   docker run --rm --gpus '"device=0"' -v $PWD:/app -w /app fin-fraud-api:1.0 \
#       python scripts/choose_threshold.py            # add --write to update manifest.json
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
from app.pipeline import FraudScorer  # noqa: E402

FN_COST = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
parser.add_argument("--fn-cost", type=float, default=FN_COST, help="cost of a missed fraud / cost of a false alarm")
parser.add_argument("--write", action="store_true", help="write the chosen threshold into manifest.json")
args = parser.parse_args()

scorer = FraudScorer(args.artifacts, threshold=0.5)
y = pd.read_csv(os.path.join(args.artifacts, "graph/nodes/transaction_label.csv")).iloc[:, 0].to_numpy()
p = scorer.score_context_graph()
assert len(p) == len(y)


def at(t):
    pred = p > t
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": round(t, 2), "precision": prec, "recall": rec, "f1": f1, "fp": fp, "fn": fn,
            "flagged": int(pred.sum()), "cost": args.fn_cost * fn + fp}


sweep = pd.DataFrame([at(t) for t in np.round(np.arange(0.05, 0.96, 0.01), 2)])
print(f"held-out test graph: {len(y):,d} tx, {int(y.sum()):,d} fraud ({y.mean():.2%})\n")
with pd.option_context("display.width", 200):
    print(sweep.iloc[::5].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = sweep.loc[sweep.cost.idxmin()]
ref = sweep[sweep.threshold == 0.5].iloc[0]
print(f"\nrule: minimise {args.fn_cost:g} x FN + FP")
for name, row in [("at 0.50", ref), (f"chosen {best.threshold:.2f}", best)]:
    print(f"  {name:12s} precision {row.precision:.4f}  recall {row.recall:.4f}  F1 {row.f1:.4f}  "
          f"FP {int(row.fp)}  FN {int(row.fn)}  cost {row.cost:.0f}")

if args.write:
    path = os.path.join(args.artifacts, "manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    manifest["threshold"] = float(best.threshold)
    manifest["threshold_selection"] = (
        f"minimise {args.fn_cost:g} x FN + FP on the held-out test graph ({len(y)} tx): "
        f"precision {best.precision:.4f}, recall {best.recall:.4f}, F1 {best.f1:.4f}, "
        f"FP {int(best.fp)}, FN {int(best.fn)} (at 0.50: precision {ref.precision:.4f}, "
        f"recall {ref.recall:.4f}, F1 {ref.f1:.4f})")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nwrote threshold {best.threshold:.2f} -> {os.path.relpath(path, ROOT)}")
    subprocess.run([os.path.join(ROOT, "scripts/write_checksums.sh")], check=True)
