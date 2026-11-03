# Fraud-scoring HTTP API (inference only).
#
#   GET  /health           liveness: the process is up
#   GET  /ready            readiness: models loaded, checks passed, warm-up done
#   GET  /v1/model         model version, threshold, devices, context-graph size
#   POST /v1/score         one raw transaction  -> P(fraud) + FRAUD / LEGIT
#   POST /v1/score/batch   up to MAX_BATCH transactions, attached to the graph together
#   POST /v1/explain       one raw transaction  -> /v1/score result + SHAP contributions per field
#   GET  /                 dashboard UI (app/static/index.html), calls /v1/explain on the same origin
#
# Settings (environment variables):
#   ARTIFACTS_DIR   default <this folder>/../artifacts
#   DEVICE          cuda (default) or cpu, for the GNN. Inside the container cuda = the pinned GPU
#   XGB_DEVICE      auto (default: GPU if XGBoost can use it, else CPU), cuda or cpu
#   FRAUD_THRESHOLD optional override of manifest.json "threshold"
#   MAX_BATCH       default 1000
#   LOG_LEVEL       default INFO
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.pipeline import FraudScorer
from app.schemas import (BatchRequest, BatchResponse, ExplainResponse, ExplainResult, ScoreResponse, ScoreResult,
                         Transaction)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fraud.api")

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR",
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "1000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    threshold = os.environ.get("FRAUD_THRESHOLD")
    # Any ArtifactError here stops the process: better no server than wrong probabilities.
    app.state.scorer = FraudScorer(
        ARTIFACTS_DIR,
        device=os.environ.get("DEVICE", "cuda"),
        xgb_device=os.environ.get("XGB_DEVICE", "auto"),
        threshold=float(threshold) if threshold else None,
    )
    yield


app = FastAPI(title="Financial fraud scoring (GNN + XGBoost)", version="1.0.0", lifespan=lifespan)


def _results(transactions, scored):
    return [ScoreResult(transaction_id=t.transaction_id, label="FRAUD" if s["is_fraud"] else "LEGIT",
                        threshold=app.state.scorer.threshold, **s)
            for t, s in zip(transactions, scored)]


def _score(transactions, explain=False):
    try:
        return app.state.scorer.score([t.model_dump() for t in transactions], explain=explain)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    if request.url.path.startswith("/v1/"):
        log.info("%s %s %s %d %.1fms", request_id, request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error, see server log"})


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if getattr(app.state, "scorer", None) is None:
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ready", "model_version": app.state.scorer.model_version}


@app.get("/v1/model")
def model_info():
    s = app.state.scorer
    return {
        "model_version": s.model_version,
        "description": s.manifest["description"],
        "threshold": s.threshold,
        "threshold_selection": s.manifest.get("threshold_selection"),
        "gnn": s.manifest["gnn"],
        "xgb_num_features": s.bst.num_features(),
        "gnn_device": s.device_name,
        "xgb_device": s.xgb_device,
        "context_graph": {n: len(s.base_x[n]) for n in ("transaction", "user", "merchant")},
        "context_graph_mode": "frozen snapshot (scored transactions are not added)",
        "max_batch": MAX_BATCH,
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score(tx: Transaction):
    t0 = time.perf_counter()
    scored = _score([tx])
    return ScoreResponse(model_version=app.state.scorer.model_version,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                         result=_results([tx], scored)[0])


@app.post("/v1/explain", response_model=ExplainResponse)
def explain(tx: Transaction):
    t0 = time.perf_counter()
    s = _score([tx], explain=True)[0]
    return ExplainResponse(model_version=app.state.scorer.model_version,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                           result=ExplainResult(transaction_id=tx.transaction_id,
                                                label="FRAUD" if s["is_fraud"] else "LEGIT",
                                                threshold=app.state.scorer.threshold, **s))


@app.post("/v1/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest):
    if len(req.transactions) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} transactions per batch")
    t0 = time.perf_counter()
    results = _results(req.transactions, _score(req.transactions))
    return BatchResponse(model_version=app.state.scorer.model_version,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                         count=len(results), flagged=sum(r.is_fraud for r in results), results=results)
