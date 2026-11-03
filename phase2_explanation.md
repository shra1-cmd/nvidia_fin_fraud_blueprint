# fin_fraud (fraud-scoring API)

## Fundamental Tasks 
 - Raw transaction cleaning (TabFormer `preprocess_data()` rules)
 - Feature encoding (fitted lookup tables, never refit)
 - Graph attachment (new transaction → frozen context graph of cards / merchants)
 - Graph embedding (GNN message passing over card and merchant history)
 - Fraud classification (XGBoost on raw features + embedding) and thresholding
 - Explanation (exact TreeSHAP per raw field, optional)

## Models
 - Heterogeneous GNN `HeterogeneousNodePredictor`: 2× SAGEConv (HeteroConv, sum), hidden 64  `artifacts/model/state_dict_gnn_model_np.pth`
 - XGBoost `binary:logistic`: 199 rounds × 50 parallel trees = 9,950 trees, 102 features  `artifacts/model/xgboost_model_np.json`
 - Encoders: binary-code lookup tables + amount scaler  `artifacts/encoders_np.pkl` (~16 MB)
 - Frozen context graph: held-out TabFormer test graph (25,803 tx / 4,795 cards / 6,773 merchants)  `artifacts/graph/` (~3.3 MB)

Model version `gnn-xgb-np-2026-10-07`, threshold **0.25** (`artifacts/manifest.json`).

## Pipeline INP & OUT flow

Notation: B = transactions in the request (1 for `/v1/score`, ≤ 1000 for `/v1/score/batch`), N_tx / N_u / N_m = transaction / card / merchant nodes in the graph.
Worked example ("e.g." values) = README `demo-1` scored on the running V100 service (GPU 6, port 8090): user 1, card 1, $488.00, Chip, Rome, merchant `-7807051024009846392`, MCC 3684.

### Overview

```
POST /v1/score  {raw TabFormer transaction JSON}
  │
  ├─[0] validate (pydantic) ───► fields in snake_case; unknown field / bad value → HTTP 422
  │
  ├─[1] clean ─────────────────► {Card, Merchant, MCC, Errors, Chip, City, Zip, Amount}        e.g. Card = 1×9+1 = 10
  │
  ├─[2] encode (encoders_np.pkl)► tx features [B, 38]  (+ new card [·, 13] / new merchant [·, 24] rows)
  │
  ├─[3] attach to frozen graph ─► x_tx [25,803+B, 38], x_user [4,795(+new), 13], x_merchant [6,773(+new), 24]
  │                               edges user→tx, tx→merchant [2, 25,803+B]  (+ reverse edges)
  │
  ├─[4] GNN, GPU ──────────────► h_tx [N_tx, 64] → cat [x_tx | h_tx] → keep new rows → [B, 102]
  │
  ├─[5] XGBoost, GPU ──────────► P(fraud) [B]                                                   e.g. 0.9654
  │
  ├─[6] decide ────────────────► P(fraud) > 0.25 → FRAUD else LEGIT                              e.g. FRAUD
  │
  └─[7] TreeSHAP (only /v1/explain), CPU
          contributions [B, 103] → summed per raw field → {Zip, Errors, City, Chip, Amount, Graph}
```

Note: nothing is ever written back to the graph. Each request gets a fresh copy of the frozen snapshot, and the whole graph is rebuilt on the GPU for that request.

### [0] Request validation: FastAPI + pydantic, CPU (`app/main.py`, `app/schemas.py`)

Raw TabFormer fields in snake_case. Required: `user`, `card` (0–8), `amount`, `use_chip`, `merchant_name` (send as a string, it is a 64-bit id), `mcc`. Optional: `merchant_city` (default `ONLINE`), `zip`, `errors`, `transaction_id`. `merchant_state`, `year`, `month`, `day`, `time` are accepted but are not model features. Any unknown field → 422.

### [1] Clean: `FraudScorer._clean`, CPU, no model (`app/pipeline.py`)

Same rules as `preprocess_data()` in the training notebooks:

```
Card    = user × 9 + card                                         e.g. 1×9+1 = 10
Merchant= str(merchant_name)                                      e.g. "-7807051024009846392"
City    = merchant_city or "ONLINE" → " ONLINE" (leading space, as in the raw data)
Errors  = errors with commas removed, empty → "XX"                e.g. "XX"
Zip     = float(zip), empty → 0.0
Amount  = "$488.00" → 488.0, empty → training median 41.52, negative = refund
```

### [2] Encode: `encoders_np.pkl` lookup tables, CPU, no model

Each categorical value maps to a fixed **binary code** (ordinal id written in bits). A value never seen in training maps to the all-zeros "unknown" code and is reported in `unseen`. Amount is scaled as `(x − 41.52) / 87.52`.

| Column | Distinct values | Encoding | Width |
|---|---|---|---|
| Zip | 21,870 | binary | 15 |
| Errors | 21 | binary | 5 |
| City | 10,909 | binary | 14 |
| Chip | 3 | one-hot | 3 |
| Amount | — | scaled float | 1 |
| **Transaction node** | | | **38** |
| Card | 6,139 | binary | 13 = **user node** |
| Merchant + MCC | 100,343 / 109 | binary | 17 + 7 = **merchant node 24** |

e.g. City "Rome" → `[0,0,0,0,0,0,0,0,0,0,1,0,1,1]`, Chip → `[1,0,0]`, Amount 488.0 → 5.101. Output: `x_new_tx` float32 [B, 38]. Card / merchant rows are encoded only when the card / merchant is new (step 3).

### [3] Attach to the frozen context graph: CPU, no model

```
frozen graph (artifacts/graph/, loaded once at startup):
  x_tx [25,803, 38]   x_user [4,795, 13]   x_merchant [6,773, 24]
  user→tx [2, 25,803]   tx→merchant [2, 25,803]

per request:
  known card     → edge to its existing user node (brings that card's history)       e.g. card 10: 19 tx
  unknown card   → one new user node (no history, card_known: false)
  known merchant → edge to its existing merchant node                                 e.g. 217 tx
  unknown merch. → one new merchant node (encoded Merchant + MCC)
  append the B new transactions → x_tx [25,803+B, 38], edges [2, 25,803+B]
```

`add_reverse_edges_new` then adds `tx→user (rev_to)` and `merchant→tx (rev_to)`, giving 4 edge types. In a batch, all B transactions share this one graph, so same-card transactions see each other.

### [4] GNN embedding: HeterogeneousNodePredictor, fp32, GPU (`artifacts/model/model.py`)

```
input_proj (Linear per node type):  x_tx [N_tx, 38] → [N_tx, 64]   x_user [N_u, 13] → [N_u, 64]   x_merchant [N_m, 24] → [N_m, 64]
2 × HeteroConv{4 × SAGEConv(64 → 64)}, aggr = sum, ReLU:
    tx  ← mean(card neighbours) + mean(merchant neighbours)
    card / merchant ← mean(their transactions)
  → after 2 hops, a transaction "sees" its card's and merchant's other transactions
cat [x_tx (38) | h_tx (64)] → [N_tx, 102]  →  keep the last B rows → features [B, 102]     e.g. [1, 102]
```

`return_hidden=True`: the GNN's own classifier head is **not** used. Only the 102-d features go to XGBoost. A lock lets only one request use the GPU at a time.

### [5] Classify: XGBoost, GPU (falls back to CPU if needed)

`DMatrix [B, 102]` → 9,950 trees (`binary:logistic`) → `P(fraud)` [B]. e.g. 0.9654. Device from `/v1/model`: `xgb_device: cuda:0`.

### [6] Decide: threshold 0.25

`is_fraud = P(fraud) > 0.25` (strictly greater, as in the notebooks). The threshold minimises 10 × missed frauds + false alarms on the test graph: recall 0.993, precision 0.853 (at 0.50: recall 0.975, precision 0.925). It can be overridden only through `FRAUD_THRESHOLD` in `.env`.

Response, e.g.:
```json
{"p_fraud": 0.9654, "is_fraud": true, "label": "FRAUD", "threshold": 0.25,
 "card_known": true, "merchant_known": true, "card_history_tx": 19, "merchant_history_tx": 217, "unseen": {}}
```

### [7] Explain (optional, `/v1/explain`): TreeSHAP, CPU, 4 threads

A separate CPU copy of the booster runs `pred_contribs=True` on the same features [B, 102] → contributions [B, 103] (102 features + bias), in log-odds. Columns are summed per raw field (`City_3` → City). The 64 GNN dims form one **Graph** group (card, merchant, MCC and their history). `base_value + Σ shap = logit(P(fraud))`, checked at startup.

e.g. demo-1:

| Group | SHAP (log-odds) |
|---|---|
| base_value | −2.418 |
| Graph (card 10 / merchant / MCC 3684) | **+6.029** |
| City (Rome) | +0.167 |
| Zip (none) | +0.096 |
| Errors (none) | +0.004 |
| Amount (488.0) | −0.164 |
| Chip (Chip Transaction) | −0.384 |
| **Sum = logit** | **3.330 → sigmoid = 0.9654** |

### Latency (V100, GPU 6)

`/v1/score` ≈ 120–145 ms (laptop RTX 3050: p50 102 ms). Almost all of it is rebuilding the ~25.8k-node graph on the GPU. `/v1/explain` ≈ 350 ms (SHAP adds ~110 ms+ on CPU). Startup: load, consistency checks, warm-up (including a SHAP additivity check).

### Summary

| Stage | Model | Input | Output |
|---|---|---|---|
| 0 Validate | pydantic | raw JSON | typed fields (or 422) |
| 1 Clean | none | raw fields | Card, Merchant, MCC, Errors, Chip, City, Zip, Amount |
| 2 Encode | lookup tables | cleaned fields | tx [B, 38] (+ new card [·, 13], merchant [·, 24]) |
| 3 Attach | none | frozen graph + new rows | x_tx [25,803+B, 38], x_user [N_u, 13], x_merchant [N_m, 24], 4 × edges [2, 25,803+B] |
| 4 Embed | GNN, 2× SAGEConv, hidden 64 | node features + edges | [x_tx \| h_tx] → [B, 102] |
| 5 Classify | XGBoost, 9,950 trees | [B, 102] | P(fraud) [B] |
| 6 Decide | threshold 0.25 | P(fraud) | FRAUD / LEGIT |
| 7 Explain | TreeSHAP (CPU) | [B, 102] | [B, 103] → 6 groups (log-odds) |
