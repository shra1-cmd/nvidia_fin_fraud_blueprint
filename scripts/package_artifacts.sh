#!/usr/bin/env bash
# Run on the TRAINING machine (laptop), not on the server.
# Copies the trained model, the fitted encoders and the frozen context graph out of the repo into
# deploy-fin_fraud/artifacts/, then writes artifacts/SHA256SUMS so the server can verify the copy.
#
#   deploy-fin_fraud/scripts/package_artifacts.sh            # repo = parent folder of deploy-fin_fraud
#   REPO=/path/to/financial-fraud-detection deploy-fin_fraud/scripts/package_artifacts.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
SRC_MODEL="$REPO/data/TabFormer/trained_models_np/python_backend_model_repository_np/prediction_and_shapley_np/1"
SRC_ENCODERS="$REPO/data/TabFormer/trained_models_np/encoders_np.pkl"
SRC_GRAPH="$REPO/data/TabFormer/gnn_np/test_gnn"
OUT="$HERE/artifacts"

mkdir -p "$OUT/model" "$OUT/graph/nodes" "$OUT/graph/edges"
cp "$SRC_MODEL"/{model.py,json_loader_writer.py,meta.json,state_dict_gnn_model_np.pth,xgboost_model_np.json} "$OUT/model/"
cp "$SRC_ENCODERS" "$OUT/encoders_np.pkl"
cp "$SRC_GRAPH"/nodes/{transaction,user,merchant,transaction_label}.csv "$OUT/graph/nodes/"
cp "$SRC_GRAPH"/edges/{user_to_transaction,transaction_to_merchant}.csv "$OUT/graph/edges/"
chmod -R a+rX "$OUT"

"$HERE/scripts/write_checksums.sh"
echo "Packaged artifacts from $REPO -> $OUT"
du -sh "$OUT"
