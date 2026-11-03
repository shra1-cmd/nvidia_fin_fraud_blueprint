# Load the trained node-prediction GNN + XGBoost once and score raw TabFormer transactions.
#
# Same steps, in the same order, as notebooks/single_tx_inference_np.py and predict_new_tx_np.py:
#
#   raw fields -> clean (preprocess_data rules) -> encode (encoders_np.pkl) -> attach to graph
#              -> GNN (2-hop SAGEConv) -> [38 raw tx features | 64-d embedding] -> XGBoost -> P(fraud)
#
# The context graph is a FROZEN snapshot (the held-out test graph): every request attaches its
# transactions to a fresh copy of it, the graph is rebuilt on the GPU for that request, and nothing
# is ever written back. Known cards / merchants link to their existing node, so the GNN sees that
# card's other transactions; unknown ones become new nodes with no history.
import json
import logging
import os
import pickle
import sys
import threading
import time
import types

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

log = logging.getLogger("fraud.pipeline")

NODE_TYPES = ("transaction", "user", "merchant")
U2T = ("user", "to", "transaction")
T2M = ("transaction", "to", "merchant")
SHAP_THREADS = 4  # CPU threads for TreeSHAP (shared server: don't let XGBoost take every core)


class ArtifactError(RuntimeError):
    """Artifacts are missing or disagree with each other; the server must not start."""


def _import_model_code(model_dir):
    """Import the exact model class shipped with the trained weights (artifacts/model/model.py)."""
    # model.py imports Triton/Captum modules that are only needed for Triton serving and SHAP.
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("triton_python_backend_utils"))
    try:
        import captum.attr  # noqa: F401
    except ImportError:
        captum_attr = types.ModuleType("captum.attr")
        captum_attr.ShapleyValueSampling = None
        sys.modules["captum"] = types.ModuleType("captum")
        sys.modules["captum.attr"] = captum_attr
    sys.dont_write_bytecode = True  # artifacts/ is mounted read-only in the container
    sys.path.insert(0, model_dir)
    from model import HeterogeneousNodePredictor, add_reverse_edges_new
    from json_loader_writer import load_meta
    return HeterogeneousNodePredictor, add_reverse_edges_new, load_meta


class FraudScorer:
    def __init__(self, artifacts_dir, device="cuda", xgb_device="auto", threshold=None):
        t0 = time.perf_counter()
        self.dir = os.path.abspath(artifacts_dir)
        with open(self._path("manifest.json")) as f:
            self.manifest = json.load(f)
        self.model_version = self.manifest["model_version"]
        self.threshold = float(self.manifest["threshold"] if threshold is None else threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ArtifactError(f"threshold must be in (0, 1), got {self.threshold}")

        self.device = torch.device(device)
        if self.device.type == "cuda":
            self._check_cuda()
        self._lock = threading.Lock()  # one GPU forward pass at a time

        self._load_encoders()
        self._load_graph()
        self._load_models(xgb_device)
        self._check_consistency()
        self._build_shap_groups()
        self._warm_up()
        log.info("scorer ready in %.1fs: model %s, threshold %.2f, GNN on %s, XGBoost on %s, "
                 "context graph %d tx / %d cards / %d merchants",
                 time.perf_counter() - t0, self.model_version, self.threshold, self.device_name,
                 self.xgb_device, *(len(self.base_x[n]) for n in NODE_TYPES))

    def _path(self, *parts):
        p = os.path.join(self.dir, *parts)
        if not os.path.exists(p):
            raise ArtifactError(f"missing artifact: {p}")
        return p

    # ------------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------------
    def _check_cuda(self):
        if not torch.cuda.is_available():
            raise ArtifactError("DEVICE=cuda but torch sees no GPU (check --gpus / device_ids and the driver)")
        major, minor = torch.cuda.get_device_capability(self.device)
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            raise ArtifactError(
                f"this torch build ({torch.__version__}, CUDA {torch.version.cuda}) has no kernels for "
                f"{torch.cuda.get_device_name(self.device)} ({arch}); supported: {torch.cuda.get_arch_list()}. "
                "Rebuild the image with a TORCH_INDEX whose wheels include this GPU (cu126 or cu118 for V100).")

    @property
    def device_name(self):
        if self.device.type != "cuda":
            return "cpu"
        return f"{torch.cuda.get_device_name(self.device)} ({self.device})"

    def _load_encoders(self):
        with open(self._path("encoders_np.pkl"), "rb") as f:
            enc = pickle.load(f)
        self.tx_tables, self.id_tables = enc["tx_tables"], enc["id_tables"]
        self.tx_columns, self.user_columns, self.merchant_columns = \
            enc["tx_columns"], enc["user_columns"], enc["merchant_columns"]
        self.amount = enc["amount"]
        self.n_card_values = enc["n_card_values"]
        self.card_to_user = {c: i for i, c in enumerate(enc["test_graph_cards"])}
        self.merchant_to_node = {m: i for i, m in enumerate(enc["test_graph_merchants"])}

    def _load_graph(self):
        frames = {n: pd.read_csv(self._path("graph", "nodes", f"{n}.csv")) for n in NODE_TYPES}
        self.graph_tx_columns = list(frames["transaction"].columns)
        self.base_x = {n: np.ascontiguousarray(f.to_numpy(np.float32)) for n, f in frames.items()}
        self.base_u2t = np.ascontiguousarray(
            pd.read_csv(self._path("graph", "edges", "user_to_transaction.csv")).to_numpy(np.int64).T)
        self.base_t2m = np.ascontiguousarray(
            pd.read_csv(self._path("graph", "edges", "transaction_to_merchant.csv")).to_numpy(np.int64).T)
        self.card_history = np.bincount(self.base_u2t[0], minlength=len(self.base_x["user"]))
        self.merchant_history = np.bincount(self.base_t2m[1], minlength=len(self.base_x["merchant"]))

    def _load_models(self, xgb_device):
        model_dir = self._path("model")
        HeterogeneousNodePredictor, self._add_reverse_edges, load_meta = _import_model_code(model_dir)
        g = self.manifest["gnn"]
        self.meta = load_meta(self._path("model", "meta.json"))
        self.gnn = HeterogeneousNodePredictor(
            hidden_channels=g["hidden_channels"], out_channels=g["out_channels"], meta=self.meta,
            node_type_to_predict="transaction", conv_class=g["conv_class"], num_layers=g["num_layers"],
        ).to(self.device)
        state = torch.load(self._path("model", "state_dict_gnn_model_np.pth"), map_location=self.device,
                           weights_only=True)
        try:
            self.gnn.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise ArtifactError(f"GNN checkpoint does not match manifest 'gnn' settings {g}: {e}") from e
        self.gnn.eval()

        self.bst = xgb.Booster()
        self.bst.load_model(self._path("model", "xgboost_model_np.json"))
        self.xgb_device = self._pick_xgb_device(xgb_device)

    def _pick_xgb_device(self, requested):
        if requested not in ("auto", "cuda", "cpu"):
            raise ArtifactError(f"XGB_DEVICE must be auto, cuda or cpu, got {requested!r}")
        if requested == "cpu" or (requested == "auto" and self.device.type != "cuda"):
            self.bst.set_param({"device": "cpu"})
            return "cpu"
        try:
            self.bst.set_param({"device": f"cuda:{self.device.index or 0}"})
            self.bst.predict(xgb.DMatrix(np.zeros((1, self.bst.num_features()), np.float32)))
            return f"cuda:{self.device.index or 0}"
        except Exception as e:  # e.g. the xgboost wheel has no kernels for this GPU or driver
            if requested == "cuda":
                raise ArtifactError(f"XGB_DEVICE=cuda but XGBoost cannot use the GPU: {e}") from e
            log.warning("XGBoost cannot use the GPU (%s); falling back to CPU", str(e).splitlines()[0])
            self.bst.set_param({"device": "cpu"})
            return "cpu"

    def _check_consistency(self):
        """Fail at startup, not with silently wrong probabilities, if the artifacts disagree."""
        problems = []
        dims = self.manifest["feature_dims"]
        for n in NODE_TYPES:
            graph_dim, meta_dim = self.base_x[n].shape[1], self.meta["nodes"][n]["feat_dim"]
            if not graph_dim == meta_dim == dims[n]:
                problems.append(f"{n} feature dim: graph {graph_dim}, meta.json {meta_dim}, manifest {dims[n]}")
        if self.graph_tx_columns != list(self.tx_columns):
            problems.append("graph transaction.csv column order != encoders tx_columns (XGBoost feature order)")
        if len(self.user_columns) != dims["user"] or len(self.merchant_columns) != dims["merchant"]:
            problems.append("encoder user/merchant column counts != manifest feature_dims")
        if len(self.card_to_user) != len(self.base_x["user"]):
            problems.append(f"encoders know {len(self.card_to_user)} graph cards, graph has {len(self.base_x['user'])}")
        if len(self.merchant_to_node) != len(self.base_x["merchant"]):
            problems.append(f"encoders know {len(self.merchant_to_node)} graph merchants, "
                            f"graph has {len(self.base_x['merchant'])}")
        expected = dims["transaction"] + self.manifest["gnn"]["hidden_channels"]
        if self.bst.num_features() != expected or self.manifest["xgb_num_features"] != expected:
            problems.append(f"XGBoost expects {self.bst.num_features()} features, manifest says "
                            f"{self.manifest['xgb_num_features']}, GNN produces {expected}")
        if self.n_card_values != self.manifest["n_card_values"]:
            problems.append(f"n_card_values: encoders {self.n_card_values}, manifest {self.manifest['n_card_values']}")
        if problems:
            raise ArtifactError("artifacts are inconsistent:\n  " + "\n  ".join(problems))

    def _build_shap_groups(self):
        """Map each XGBoost input column to the raw field it encodes (same grouping as the notebook's
        create_feature_mask: 'City_3' -> City). The 64 GNN embedding dims form one 'Graph' group: they
        carry Card, Merchant, MCC and the card/merchant history, which XGBoost never sees directly."""
        groups = [c.split("_")[0] for c in self.tx_columns]
        groups += ["Graph"] * (self.bst.num_features() - len(groups))
        self.shap_group_names = list(dict.fromkeys(groups))
        self.shap_group_index = np.array([self.shap_group_names.index(g) for g in groups])
        # Separate CPU copy for TreeSHAP: for one row it is ~110 ms on 4 threads vs ~540 ms with
        # GPUTreeShap on a V100. The scoring booster (self.bst) stays where it is.
        self.bst_shap = xgb.Booster()
        self.bst_shap.load_model(self._path("model", "xgboost_model_np.json"))
        self.bst_shap.set_param({"device": "cpu", "nthread": SHAP_THREADS})

    def _warm_up(self):
        """Pay CUDA init / kernel load once at startup, and prove the whole path runs on this GPU."""
        warm = [{"user": 0, "card": 0, "amount": "10.00", "merchant_name": next(iter(self.merchant_to_node)),
                 "mcc": 5411, "use_chip": "Chip Transaction", "merchant_city": "ONLINE"}]
        out = self.score(warm, explain=True)[0]
        # TreeSHAP is exact: base value + contributions must rebuild the predicted log-odds.
        logit = np.log(out["p_fraud"] / (1 - out["p_fraud"]))
        rebuilt = out["explanation"]["base_value"] + sum(c["shap"] for c in out["explanation"]["contributions"])
        if abs(rebuilt - logit) > 1e-3:
            raise ArtifactError(f"SHAP contributions do not add up to the prediction ({rebuilt} vs {logit})")

    # ------------------------------------------------------------------------
    # Per request
    # ------------------------------------------------------------------------
    def _clean(self, t):
        """preprocess_data() cleaning rules for one raw transaction (API field names)."""
        if not 0 <= int(t["card"]) < self.n_card_values:
            raise ValueError(f"card must be 0..{self.n_card_values - 1}, got {t['card']}")
        city = t.get("merchant_city") or "ONLINE"
        if city not in self.tx_tables["City"]["table"] and city.strip() == "ONLINE":
            city = " ONLINE"  # the raw data spells it with a leading space
        amount = "" if t.get("amount") is None else str(t["amount"]).replace("$", "").strip()
        zip_code = t.get("zip")
        return {
            "Card": int(t["user"]) * self.n_card_values + int(t["card"]),
            "Merchant": str(t["merchant_name"]).strip(),
            "MCC": int(t["mcc"]),
            "Errors": (t.get("errors") or "").strip().replace(",", "") or "XX",
            "Chip": t["use_chip"],
            "City": city,
            "Zip": float(zip_code) if zip_code not in (None, "") else 0.0,
            "Amount": float(amount) if amount else self.amount["median"],
        }

    @staticmethod
    def _encode(rows, tables, columns, extra=None):
        """Look up each raw value's encoded bits; unseen values get the encoder's 'unknown' code."""
        out = np.empty((len(rows), len(columns)), np.float32)
        unseen = [{} for _ in rows]
        for r, row in enumerate(rows):
            blocks = dict(extra[r]) if extra else {}
            for col, t in tables.items():
                if col not in row or not any(n in columns for n in t["columns"]):
                    continue
                v = row[col]
                if v in t["table"]:
                    blocks.update(zip(t["columns"], t["table"][v]))
                elif t["unknown"] is not None:
                    blocks.update(zip(t["columns"], t["unknown"]))
                    unseen[r][col] = v
                else:
                    raise ValueError(f"unsupported {col} value {v!r}; expected one of {sorted(t['table'])}")
            out[r] = [blocks[n] for n in columns]
        return out, unseen

    def score(self, transactions, explain=False):
        """Score a list of raw transactions (dicts with the API field names). Transactions in the same
        call are attached to the graph together, so same-card transactions see each other as context.
        explain=True adds exact TreeSHAP contributions (log-odds) of the XGBoost stage, per raw field;
        it does not change p_fraud."""
        txs = [self._clean(t) for t in transactions]
        scaled = [{"Amount": (tx["Amount"] - self.amount["center"]) / self.amount["scale"]} for tx in txs]
        x_new_tx, unseen = self._encode(txs, self.tx_tables, self.tx_columns, scaled)

        # Attach: link each transaction to its existing card / merchant node, or create one new node
        # per unseen card / merchant (shared by all transactions of this call that use it).
        n_tx, n_user, n_merchant = (len(self.base_x[n]) for n in NODE_TYPES)
        new_users, new_merchants = {}, {}
        user_nodes, merchant_nodes = [], []
        for tx in txs:
            u = self.card_to_user.get(tx["Card"])
            if u is None:
                u = new_users.setdefault(tx["Card"], n_user + len(new_users))
            m = self.merchant_to_node.get(tx["Merchant"])
            if m is None:
                if tx["Merchant"] not in new_merchants:
                    new_merchants[tx["Merchant"]] = (n_merchant + len(new_merchants), tx["MCC"])
                m = new_merchants[tx["Merchant"]][0]
            user_nodes.append(u)
            merchant_nodes.append(m)

        x_new_user, unseen_user = self._encode([{"Card": c} for c in new_users],
                                               self.id_tables, self.user_columns)
        x_new_merchant, unseen_merchant = self._encode(
            [{"Merchant": k, "MCC": mcc} for k, (_, mcc) in new_merchants.items()],
            self.id_tables, self.merchant_columns)
        for i, (u, m) in enumerate(zip(user_nodes, merchant_nodes)):
            if u >= n_user:
                unseen[i].update(unseen_user[u - n_user])
            if m >= n_merchant:
                unseen[i].update(unseen_merchant[m - n_merchant])

        tx_ids = n_tx + np.arange(len(txs))
        x = {"transaction": np.vstack([self.base_x["transaction"], x_new_tx]),
             "user": np.vstack([self.base_x["user"], x_new_user]),
             "merchant": np.vstack([self.base_x["merchant"], x_new_merchant])}
        u2t = np.hstack([self.base_u2t, np.stack([np.array(user_nodes, np.int64), tx_ids])])
        t2m = np.hstack([self.base_t2m, np.stack([tx_ids, np.array(merchant_nodes, np.int64)])])

        # GNN -> [raw tx features | embedding] -> XGBoost
        with self._lock, torch.no_grad():
            x_t = {k: torch.from_numpy(v).to(self.device) for k, v in x.items()}
            edges = {U2T: torch.from_numpy(u2t).to(self.device), T2M: torch.from_numpy(t2m).to(self.device)}
            emb = self.gnn(x_t, self._add_reverse_edges(edges), node_type="transaction", return_hidden=True)
            features = emb[n_tx:].float().cpu().numpy()
            p_fraud = self.bst.predict(xgb.DMatrix(features))
        contribs = (self.bst_shap.predict(xgb.DMatrix(features, nthread=SHAP_THREADS), pred_contribs=True)
                    if explain else None)

        results = [{
            "p_fraud": float(p),
            "is_fraud": bool(p > self.threshold),  # strictly greater, as in the notebooks
            "card_known": u < n_user,
            "merchant_known": m < n_merchant,
            "card_history_tx": int(self.card_history[u]) if u < n_user else 0,
            "merchant_history_tx": int(self.merchant_history[m]) if m < n_merchant else 0,
            "unseen": {k: str(v) for k, v in un.items()},
        } for p, u, m, un in zip(p_fraud, user_nodes, merchant_nodes, unseen)]
        if explain:
            for r, tx, c in zip(results, txs, contribs):
                per_group = np.bincount(self.shap_group_index, weights=c[:-1],
                                        minlength=len(self.shap_group_names))
                values = {g: tx.get(g) for g in self.shap_group_names}
                values["Errors"] = None if tx["Errors"] == "XX" else tx["Errors"]
                values["Zip"] = None if tx["Zip"] == 0.0 else tx["Zip"]
                values["Graph"] = f"card {tx['Card']} / merchant {tx['Merchant']} / MCC {tx['MCC']}"
                r["explanation"] = {
                    "base_value": float(c[-1]),
                    "contributions": [{"feature": g, "shap": float(s), "value": None if v is None else str(v)}
                                      for g, s, v in zip(self.shap_group_names, per_group, values.values())],
                }
        return results

    def score_context_graph(self):
        """P(fraud) for every transaction already in the frozen graph (used to pick the threshold)."""
        with self._lock, torch.no_grad():
            x_t = {k: torch.from_numpy(v).to(self.device) for k, v in self.base_x.items()}
            edges = {U2T: torch.from_numpy(self.base_u2t).to(self.device),
                     T2M: torch.from_numpy(self.base_t2m).to(self.device)}
            emb = self.gnn(x_t, self._add_reverse_edges(edges), node_type="transaction", return_hidden=True)
            return self.bst.predict(xgb.DMatrix(emb.float().cpu().numpy()))
