"""
=============================================================================
STAGE 6 — FastAPI Service (JWT-secured fraud scoring API)
=============================================================================

PURPOSE:
    This is the production API server that:
    1. Accepts transaction JSON via POST /score-transaction
    2. Runs preprocessing -> XGBoost predict -> SHAP explain -> tier
    3. Returns {transaction_id, risk_score, tier, top_reasons[]}
    4. Logs every scored transaction to an in-memory store
    5. Serves GET /flagged-transactions for the dashboard to poll
    6. Serves the static dashboard (index.html + main.js)

DESIGN DECISIONS:
    - JWT via python-jose (not fastapi-users): we only need token
      verification, not user registration/management. Lighter dep footprint.
    - In-memory deque(maxlen=1000): no DB needed for buildathon scope.
      deque caps memory usage; comment acknowledges this isn't durable.
    - Model + SHAP loaded once at startup via lifespan handler — avoids
      per-request I/O and keeps latency low (~5-10ms per prediction).
    - preprocess_features() imported from train.train_model — single
      source of truth eliminates train/serve skew.
    - CORS enabled for local dev flexibility.

USAGE:
    cd <project_root>
    python -m uvicorn api.main:app --reload --port 8000
    
    Or:
    python api/main.py

ENDPOINTS:
    POST /score-transaction  (JWT required) — score a single transaction
    GET  /flagged-transactions (JWT required) — return all scored txns
    GET  /                    (no auth)      — serve dashboard HTML
    GET  /api/tiers           (no auth)      — return tier definitions
=============================================================================
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jose import jwt, JWTError
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add project root to path so we can import from train/ and model/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train.train_model import preprocess_features
from model.explain import init_explainer, explain_prediction
from model.tiering import assign_tier, get_tier_info, get_all_tiers
import joblib

# ---------------------------------------------------------------------------
# JWT CONFIGURATION
# ---------------------------------------------------------------------------
# WHY a hardcoded secret for the buildathon?
# This is a demo system, not production auth. A real deployment would:
# 1. Store the secret in an env var or secrets manager (AWS Secrets Manager,
#    HashiCorp Vault)
# 2. Use RS256 (asymmetric) instead of HS256 (symmetric) so the API only
#    needs the public key to verify tokens
# 3. Implement token refresh, revocation, and scope-based access control
#
# For a 12-day buildathon, a hardcoded HS256 secret is acceptable.
# The panel should hear you acknowledge the production gaps.
# ---------------------------------------------------------------------------
JWT_SECRET = "risklens-buildathon-secret-key-2024"
JWT_ALGORITHM = "HS256"
# Token validity: 24 hours — long enough for a full demo session.
JWT_EXPIRATION_HOURS = 24

# Security scheme for Swagger UI / dependency injection.
security = HTTPBearer()

# ---------------------------------------------------------------------------
# IN-MEMORY TRANSACTION LOG
# ---------------------------------------------------------------------------
# WHY deque(maxlen=1000)?
# 1. No database dependency — keeps the stack minimal for a buildathon.
# 2. maxlen=1000 caps memory usage at ~1MB (each record is ~1KB of JSON).
# 3. Oldest entries are automatically evicted when full (FIFO behavior).
# 4. Thread-safe for appends and pops in CPython (GIL protects deque ops).
#
# KNOWN LIMITATION: Data is lost on server restart. In production, you'd
# write to a durable store (PostgreSQL, Redis, or even a CSV file).
# ---------------------------------------------------------------------------
transaction_log: deque = deque(maxlen=1000)

# ---------------------------------------------------------------------------
# MODEL STATE — loaded once at startup
# ---------------------------------------------------------------------------
_model = None
_feature_columns = None


# ---------------------------------------------------------------------------
# PYDANTIC MODELS — request/response schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    """
    Schema for incoming transaction data.
    
    WHY Pydantic (not raw dict)?
    - Automatic type validation and coercion (e.g., string "45000" -> float)
    - Auto-generated OpenAPI docs (Swagger UI)
    - Clear error messages when fields are missing or wrong type
    """
    transaction_id: str = Field(
        default=None,
        description="Unique transaction ID. Auto-generated if not provided."
    )
    merchant_id: str = Field(
        ...,
        description="Merchant identifier"
    )
    payment_method: str = Field(
        ...,
        description="Payment method: upi, card, netbanking, or wallet"
    )
    amount_inr: float = Field(
        ..., gt=0,
        description="Transaction amount in INR (must be positive)"
    )
    hour_of_day: int = Field(
        ..., ge=0, le=23,
        description="Hour of day (0-23)"
    )
    merchant_txn_count_7d: int = Field(
        ..., ge=0,
        description="Merchant's transaction count in the last 7 days"
    )
    merchant_avg_amount_7d: float = Field(
        ..., ge=0,
        description="Merchant's average transaction amount in the last 7 days"
    )
    device_risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Device risk score (0=clean, 1=high risk)"
    )
    is_new_merchant: int = Field(
        ..., ge=0, le=1,
        description="1 if merchant is less than 7 days old, 0 otherwise"
    )


class ScoringResponse(BaseModel):
    """
    Schema for the scoring API response.
    Matches the DATA FLOW spec exactly.
    """
    transaction_id: str
    risk_score: float
    tier: str
    tier_label: str
    tier_color: str
    top_reasons: list[str]
    scored_at: str
    merchant_id: str
    amount_inr: float
    payment_method: str


# ---------------------------------------------------------------------------
# JWT UTILITIES
# ---------------------------------------------------------------------------
def create_token(subject: str = "demo_user") -> str:
    """
    Generate a JWT token for testing/demo purposes.
    
    In production, tokens would be issued by an identity provider (Auth0,
    Cognito, etc.) after proper authentication. Here we self-issue tokens
    with a 24-hour expiry.
    """
    payload = {
        "sub": subject,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    FastAPI dependency that extracts and verifies the JWT from the
    Authorization: Bearer <token> header.
    
    Returns the subject (user ID) from the token payload.
    Raises 401 if the token is missing, expired, or invalid.
    
    WHY a dependency (not middleware)?
    - Dependencies are per-route: we can protect /score-transaction but
      leave / (dashboard) unprotected.
    - Middleware would apply to all routes, requiring exclusion lists.
    - Dependencies are visible in the Swagger UI auto-docs.
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        subject = payload.get("sub")
        if subject is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return subject
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# ---------------------------------------------------------------------------
# LIFESPAN — load model + SHAP at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs once at startup and shutdown.
    
    WHY lifespan (not @app.on_event)?
    on_event("startup") is deprecated in FastAPI >= 0.109. The lifespan
    context manager is the recommended replacement. It also gives us
    clean shutdown behavior (the 'yield' separates startup from shutdown).
    
    We load the model and SHAP explainer here so they're ready before
    the first request arrives. Loading takes ~1-2 seconds; per-request
    prediction takes ~5-10ms.
    """
    global _model, _feature_columns
    
    model_path = os.path.join(PROJECT_ROOT, 'model', 'risk_model.pkl')
    feature_cols_path = os.path.join(PROJECT_ROOT, 'model', 'feature_columns.pkl')
    
    print("Loading model and SHAP explainer...")
    _model = joblib.load(model_path)
    _feature_columns = joblib.load(feature_cols_path)
    print(f"  Model loaded: {model_path}")
    print(f"  Features: {len(_feature_columns)} columns")
    
    # Initialize SHAP TreeExplainer (shared with model.explain module)
    init_explainer()
    
    # Generate and print a dev token for easy testing
    dev_token = create_token("dev_user")
    print(f"\n  DEV TOKEN (valid {JWT_EXPIRATION_HOURS}h):")
    print(f"  {dev_token}\n")
    
    yield  # Server is running — handle requests
    
    # Shutdown (cleanup if needed)
    print("Server shutting down.")


# ---------------------------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RiskLens — AI Risk Manager",
    description="Real-time transaction fraud scoring with XGBoost + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local dev.
# WHY allow all? During development, the dashboard may be opened from
# file://, localhost:3000, or any other origin. In production, you'd
# restrict this to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (dashboard HTML + JS)
static_dir = os.path.join(PROJECT_ROOT, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """
    Serve the dashboard HTML page.
    No authentication required — the dashboard itself handles token
    inclusion in its API calls.
    """
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "Dashboard not found. Create static/index.html (Stage 7)."},
        status_code=404,
    )


@app.post("/score-transaction", response_model=ScoringResponse)
async def score_transaction(
    txn: TransactionInput,
    user: str = Depends(verify_token)
):
    """
    Score a single transaction for fraud risk.
    
    Pipeline: preprocess -> XGBoost predict -> SHAP explain -> tier assign
    
    This endpoint mirrors the DATA FLOW spec exactly:
    transaction JSON -> preprocessing -> predict_proba -> SHAP -> tier -> response
    """
    # Auto-generate transaction ID if not provided
    if txn.transaction_id is None:
        txn.transaction_id = f"txn_{uuid.uuid4().hex[:12]}"
    
    # Validate payment method
    valid_methods = {'upi', 'card', 'netbanking', 'wallet'}
    if txn.payment_method.lower() not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payment_method: '{txn.payment_method}'. "
                   f"Must be one of: {valid_methods}"
        )
    
    # --- Convert to dict for processing ---
    txn_dict = txn.model_dump()
    txn_dict['payment_method'] = txn_dict['payment_method'].lower()
    
    # --- Preprocess features ---
    # Uses the SAME preprocess_features() function as training (Stage 3).
    # This is the train/serve contract that prevents feature skew.
    df = pd.DataFrame([txn_dict])
    features = preprocess_features(df)
    
    # Ensure column order matches training
    for col in _feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[_feature_columns]
    
    # --- Predict fraud probability ---
    # predict_proba returns [[P(legit), P(fraud)]] — we want P(fraud).
    risk_score = float(_model.predict_proba(features)[0][1])
    
    # --- SHAP explanation ---
    # Returns top 3 contributing features with plain-language reasons.
    shap_reasons = explain_prediction(txn_dict, top_n=3)
    top_reasons = [r['reason'] for r in shap_reasons]
    
    # --- Assign confidence tier ---
    tier = assign_tier(risk_score)
    tier_info = get_tier_info(tier)
    
    # --- Build response ---
    scored_at = datetime.now(timezone.utc).isoformat()
    response = ScoringResponse(
        transaction_id=txn.transaction_id,
        risk_score=round(risk_score, 4),
        tier=tier,
        tier_label=tier_info['label'],
        tier_color=tier_info['color'],
        top_reasons=top_reasons,
        scored_at=scored_at,
        merchant_id=txn.merchant_id,
        amount_inr=txn.amount_inr,
        payment_method=txn.payment_method,
    )
    
    # --- Log to in-memory store ---
    # The dashboard polls GET /flagged-transactions to read this log.
    transaction_log.appendleft(response.model_dump())
    
    return response


@app.get("/flagged-transactions")
async def get_flagged_transactions(
    user: str = Depends(verify_token)
):
    """
    Return all scored transactions from the in-memory log.
    The dashboard polls this endpoint every few seconds.
    
    WHY return ALL transactions (not just flagged)?
    The dashboard color-codes by tier, so it needs all tiers to show
    the full picture. Filtering to only flagged (review + block) would
    hide the auto_allow majority, giving a misleading view of the
    system's behavior.
    
    Returns most recent first (deque appendleft gives us this order).
    """
    return {
        "transactions": list(transaction_log),
        "total": len(transaction_log),
    }


@app.get("/api/tiers")
async def get_tiers():
    """
    Return tier definitions for the dashboard legend.
    No auth required — tier metadata is not sensitive.
    """
    return {"tiers": get_all_tiers()}


@app.post("/api/generate-token")
async def generate_token():
    """
    Generate a new JWT token for testing.
    
    WHY no auth on this endpoint?
    This is a demo convenience endpoint — it lets the dashboard and
    curl commands get a valid token without manual JWT crafting.
    In production, this would be behind proper authentication.
    """
    token = create_token("demo_user")
    return {
        "token": token,
        "expires_in_hours": JWT_EXPIRATION_HOURS,
        "usage": f"Authorization: Bearer {token}",
    }


# ---------------------------------------------------------------------------
# ENTRYPOINT — run with: python api/main.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    
    # WHY host="0.0.0.0"? Allows access from other devices on the same
    # network (useful for demoing on a projector from a different machine).
    # Use "127.0.0.1" if you want to restrict to localhost only.
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,         # Auto-reload on code changes during dev
        log_level="info",
    )
