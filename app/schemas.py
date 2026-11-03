# Request / response bodies. Field names are the raw TabFormer columns in snake_case:
#   User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?
import math
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "transaction_id": "demo-1", "user": 1, "card": 1, "amount": "$488.00",
        "use_chip": "Chip Transaction", "merchant_name": "-7807051024009846392", "merchant_city": "Rome",
        "merchant_state": "Italy", "zip": None, "mcc": 3684, "errors": None,
        "year": 2019, "month": 5, "day": 11, "time": "14:05"}})

    transaction_id: Optional[str] = Field(None, max_length=128, description="echoed back, not used by the model")
    user: int = Field(ge=0)
    card: int = Field(ge=0, description="card index of that user (0-8)")
    amount: Union[float, str, None] = Field(description="e.g. 488.0 or '$488.00'; negative = refund; "
                                                        "empty = training median")
    use_chip: Literal["Chip Transaction", "Swipe Transaction", "Online Transaction"]
    merchant_name: Union[int, str] = Field(description="anonymised merchant id; send as a string to be safe")
    merchant_city: str = Field("ONLINE", max_length=128)
    zip: Union[float, str, None] = None
    mcc: int = Field(ge=0, le=9999)
    errors: Optional[str] = Field(None, max_length=256, description="e.g. 'Bad PIN' or 'Bad CVV,Insufficient Balance'")
    # Accepted for convenience, not model features
    merchant_state: Optional[str] = Field(None, max_length=128)
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    time: Optional[str] = Field(None, max_length=16)

    @field_validator("amount")
    @classmethod
    def _amount(cls, v):
        if v is None or (isinstance(v, str) and not v.replace("$", "").strip()):
            return None
        value = float(str(v).replace("$", "").strip()) if isinstance(v, str) else float(v)
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return value

    @field_validator("merchant_name")
    @classmethod
    def _merchant(cls, v):
        v = str(v).strip()
        if not v:
            raise ValueError("merchant_name is empty")
        return v

    @field_validator("zip")
    @classmethod
    def _zip(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        value = float(v)
        if not math.isfinite(value):
            raise ValueError("zip must be finite")
        return value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1)


class ScoreResult(BaseModel):
    transaction_id: Optional[str]
    p_fraud: float
    is_fraud: bool
    label: Literal["FRAUD", "LEGIT"]
    threshold: float
    card_known: bool = Field(description="card exists in the context graph (its history is used)")
    merchant_known: bool
    card_history_tx: int = Field(description="other transactions of this card in the context graph")
    merchant_history_tx: int
    unseen: dict[str, str] = Field(description="fields with values never seen in training (encoded as 'unknown')")


class ScoreResponse(BaseModel):
    model_version: str
    latency_ms: float
    result: ScoreResult


class Contribution(BaseModel):
    feature: str = Field(description="raw field (Amount, Chip, City, Zip, Errors) or Graph (GNN embedding)")
    shap: float = Field(description="log-odds contribution; > 0 pushes towards FRAUD")
    value: Optional[str] = Field(description="the cleaned input value of that field")


class Explanation(BaseModel):
    base_value: float = Field(description="XGBoost expected log-odds; base_value + sum(shap) = logit(p_fraud)")
    contributions: list[Contribution]


class ExplainResult(ScoreResult):
    explanation: Explanation


class ExplainResponse(BaseModel):
    model_version: str
    latency_ms: float
    result: ExplainResult


class BatchResponse(BaseModel):
    model_version: str
    latency_ms: float
    count: int
    flagged: int
    results: list[ScoreResult]
