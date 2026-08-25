# RiskLens — AI Risk Manager

Real-time transaction fraud scoring system built for the Razorpay AI Buildathon. Uses XGBoost for prediction, SHAP for explainability, and FastAPI for serving.

---

## Problem Statement

Payment processors handle millions of transactions daily. Each one needs a fraud/no-fraud decision in milliseconds — too slow and the customer bounces, too permissive and fraud losses mount. Rule-based systems can't keep up with evolving fraud patterns, and black-box ML models can't explain their decisions to regulators or ops teams.

**RiskLens** solves both problems: it scores transactions in real-time using a trained XGBoost model, then explains each decision in plain language using SHAP (SHapley Additive exPlanations), and routes transactions into actionable tiers (auto-allow, human-review, auto-block).

---

## Architecture

```
+----------------------------------------------+
|              Frontend (Dashboard)             |
|         static/index.html + main.js           |
|  Polls /flagged-transactions every 3 seconds  |
+---------------------+------------------------+
                      |
                      | HTTP (JWT-secured)
                      v
+----------------------------------------------+
|           FastAPI Service (api/main.py)        |
|                                                |
|  POST /score-transaction                       |
|    1. Validate input (Pydantic)                |
|    2. preprocess_features() [train/serve sync] |
|    3. XGBoost predict_proba()                  |
|    4. SHAP TreeExplainer -> top 3 reasons      |
|    5. assign_tier() -> auto_allow/review/block |
|    6. Log to in-memory deque                   |
|    7. Return JSON response                     |
|                                                |
|  GET /flagged-transactions                     |
|    -> Returns in-memory transaction log        |
+-----+--------------------+-------------------+
      |                    |
      v                    v
+------------+    +------------------+
| risk_model |    | SHAP             |
| .pkl       |    | TreeExplainer    |
| (XGBoost)  |    | (model/explain)  |
+------------+    +------------------+
```

---

## Data Flow

```
Transaction JSON
    |
    v
POST /score-transaction (JWT required)
    |
    v
preprocess_features()  <-- same function used in training
    |
    v
XGBoost.predict_proba() --> risk_score (0-1)
    |
    v
SHAP TreeExplainer --> top 3 contributing features
    |                   translated to plain language
    v
Confidence tiering:
    score < 0.3  --> auto_allow  (green)
    0.3 - 0.7    --> human_review (amber)
    score > 0.7  --> auto_block  (red)
    |
    v
JSON Response:
{
    "transaction_id": "txn_001",
    "risk_score": 0.82,
    "tier": "auto_block",
    "top_reasons": [
        "High transaction count this week increased risk",
        "UPI payment method decreased risk",
        "New merchant increased risk"
    ]
}
    |
    v
Logged to in-memory store --> Dashboard polls and displays
```

---

## Measured Results (Stage 3 — held-out 20% test set)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.7734 |
| **PR-AUC** | 0.3340 |
| **Precision** | 0.3797 |
| **Recall** | 0.4225 |
| **F1-Score** | 0.4000 |

### Tier Distribution on Test Set

| Tier | % of Transactions | Precision | Description |
|------|-------------------|-----------|-------------|
| auto_allow | 89.7% | — | Vast majority pass through |
| human_review | 7.0% | 12.4% | Moderate-risk, routed to manual review |
| auto_block | 3.3% | 51.0% | High confidence — 1 in 2 blocked txns is real fraud |

### Baseline Validation (Stage 2 — Kaggle "Give Me Some Credit")

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8221 |
| PR-AUC | 0.2834 |

This confirms the XGBoost + SMOTE approach generalizes to real-world credit data before applying it to our synthetic fraud dataset.

### Why ROC-AUC is 0.77, Not 0.95+

The synthetic dataset uses **probabilistic fraud labeling** — even when all fraud-pattern conditions are met, there's only a 55-70% chance the transaction is labeled as fraud. This creates an irreducible noise floor that prevents any model from achieving AUC = 1.0. This is intentional: it forces the model to learn soft decision boundaries rather than trivial if-else rules, and it's more representative of real-world fraud data where ground truth is noisy.

---

## Synthetic Data — Fraud Patterns

The synthetic dataset (7,500 transactions) has 4 injected fraud patterns:

| # | Pattern | Trigger Rate | Fraud Rate | Rationale |
|---|---------|-------------|------------|-----------|
| 1 | Late-night high-value UPI + new merchant | 20 txns | 80.0% | UPI instant-settlement + no merchant history |
| 2 | Velocity spike (>5x baseline) | 369 txns | 46.6% | Classic money-laundering signal |
| 3 | Amount z-score outlier (>3 sigma) | 16 txns | 31.2% | Relative anomaly detection |
| 4 | High device risk + new merchant | 70 txns | 55.7% | Device fingerprinting + merchant newness |

Overall fraud rate: **4.76%** (357 / 7,500).

---

## Known Limitations

These are **specific, not generic** — each one has a reason and a mitigation path.

1. **In-memory storage is not durable.** Transaction logs are lost on server restart. A production system would write to PostgreSQL or Redis. Mitigation: add a persistence layer (30 min of work).

2. **JWT secret is hardcoded.** The `JWT_SECRET` in `api/main.py` is a string literal. Production would use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault). Mitigation: read from `os.environ.get('JWT_SECRET')`.

3. **No model retraining pipeline.** The model is trained once and served statically. Concept drift would degrade performance over time. Mitigation: schedule periodic retraining with fresh data, monitor prediction distributions for drift.

4. **Synthetic data doesn't capture all real-world fraud patterns.** We inject 4 known patterns; real fraud includes social engineering, account takeover, refund abuse, etc. Mitigation: train on real labeled data when available.

5. **Single-instance deployment.** No load balancing, no horizontal scaling. The in-memory deque doesn't sync across instances. Mitigation: use Redis for shared state, deploy behind a load balancer.

6. **SHAP explanations are in log-odds space.** TreeExplainer works in XGBoost's raw output space (log-odds), not probability space. SHAP values sum to the log-odds prediction, not the probability. This is mathematically correct but can be confusing when presented to non-technical stakeholders.

7. **Threshold tuning is manual.** The 0.3/0.7 tier thresholds are based on one test-set evaluation. Production would use cost-sensitive optimization (minimize total cost = fraud_loss × FN + review_cost × FP + block_cost × legitimate_blocks).

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Model | XGBoost | 2.x |
| Class balancing | imbalanced-learn (SMOTE) | 0.12+ |
| Explainability | SHAP (TreeExplainer) | 0.45+ |
| API | FastAPI + Uvicorn | 0.110+ |
| Auth | python-jose (JWT HS256) | 3.3+ |
| Data | pandas, numpy, scikit-learn | latest |
| Frontend | Vanilla HTML + JS | — |
| Python | CPython | 3.11 |

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/Hrishikesh-Rondla/RiskLens.git
cd RiskLens

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (macOS/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Data & Train Model

```bash
# Stage 1: Generate synthetic dataset
python data/generate.py

# Stage 2: Validate on Kaggle data (optional — requires cs-training.csv)
# Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place at: data/kaggle/cs-training.csv
python scripts/validate_baseline.py

# Stage 3: Train the production model
python train/train_model.py
```

### Run the Server

```bash
# Start the API + dashboard
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Open the dashboard
# http://127.0.0.1:8000
```

### Test the API

```bash
# Get a JWT token
curl -X POST http://127.0.0.1:8000/api/generate-token

# Score a transaction (replace <TOKEN> with the token from above)
curl -X POST http://127.0.0.1:8000/score-transaction \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "merch_0042",
    "payment_method": "upi",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "merchant_txn_count_7d": 180,
    "merchant_avg_amount_7d": 1200,
    "device_risk_score": 0.92,
    "is_new_merchant": 1
  }'

# View scored transactions
curl -H "Authorization: Bearer <TOKEN>" \
  http://127.0.0.1:8000/flagged-transactions
```

---

## Project Structure

```
RiskLens/
├── data/
│   ├── generate.py                 # Synthetic dataset generator (Stage 1)
│   ├── synthetic_transactions.csv  # Generated dataset (7,500 rows)
│   └── kaggle/
│       └── cs-training.csv         # Kaggle dataset (not in git)
├── scripts/
│   ├── validate_baseline.py        # Kaggle baseline validation (Stage 2)
│   ├── baseline_roc_curve.png      # ROC curve plot
│   └── baseline_pr_curve.png       # PR curve plot
├── train/
│   ├── train_model.py              # Model training pipeline (Stage 3)
│   └── misclassified_cases.csv     # FP/FN cases for analysis
├── model/
│   ├── risk_model.pkl              # Trained XGBoost model
│   ├── feature_columns.pkl         # Feature column ordering
│   ├── explain.py                  # SHAP explainability (Stage 4)
│   └── tiering.py                  # Confidence tier logic (Stage 5)
├── api/
│   └── main.py                     # FastAPI service (Stage 6)
├── static/
│   ├── index.html                  # Dashboard UI (Stage 7)
│   └── main.js                     # Dashboard logic
├── requirements.txt                # Pinned dependencies
├── .gitignore
└── README.md                       # This file (Stage 8)
```

---

## Author

**Hrishikesh Rondla** — Solo build for Razorpay AI Buildathon 2024

---