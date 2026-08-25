"""
=============================================================================
STAGE 4 — SHAP Explainability Module
=============================================================================

PURPOSE:
    Wrap the trained XGBoost model with SHAP's TreeExplainer to provide
    per-prediction explanations in plain language. The API (Stage 6) calls
    explain_prediction() for every scored transaction, and the dashboard
    (Stage 7) displays the top 3 reasons to the user.

WHY SHAP (not LIME, not feature importances)?
    1. TreeExplainer is EXACT for tree models — it computes Shapley values
       in O(TLD) time (T=trees, L=leaves, D=depth) using the tree structure
       directly. LIME and KernelSHAP are approximate and orders of magnitude
       slower.
    2. SHAP values are additive: they sum to (model output - base value),
       which means they have a grounded mathematical interpretation.
       Feature importances (gain/cover) are global averages — they can't
       explain individual predictions.
    3. SHAP values are signed: positive = increases fraud probability,
       negative = decreases it. This lets us say "new merchant INCREASED
       risk" vs "established merchant DECREASED risk" — directional
       explanations are far more useful than just "merchant age matters."

WHY TOP 3 FEATURES (not top 5 or all)?
    - 3 is the UX sweet spot for a dashboard row: enough to explain the
      decision, few enough to scan at a glance.
    - Industry standard: Stripe Radar shows 3 risk factors, PayPal shows
      2-4. More than 5 causes cognitive overload.
    - For a panel demo, 3 reasons per transaction makes the walkthrough
      clean and focused.

PERFORMANCE:
    TreeExplainer is initialized once at module load (or via init_explainer).
    Per-prediction SHAP computation is ~1-5ms for a single instance on a
    300-tree XGBoost model — well within real-time scoring latency budgets.

USAGE:
    from model.explain import init_explainer, explain_prediction
    
    init_explainer()  # call once at startup
    reasons = explain_prediction({
        'amount_inr': 15000,
        'hour_of_day': 2,
        'merchant_txn_count_7d': 150,
        ...
    })
    # Returns: [
    #     "High transaction count this week increased risk",
    #     "Transaction during unusual hours (12am-5am) increased risk",
    #     "New merchant increased risk"
    # ]
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

# Add parent dir to path so we can import from train/
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train.train_model import preprocess_features

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'risk_model.pkl')
FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, 'model', 'feature_columns.pkl')

# ---------------------------------------------------------------------------
# PLAIN-LANGUAGE FEATURE NAME MAPPING
# ---------------------------------------------------------------------------
# WHY a hardcoded dictionary (not LLM-generated)?
# 1. Deterministic — same input always produces the same explanation.
# 2. Zero latency — dictionary lookup is O(1), no API call needed.
# 3. Auditable — the mapping is version-controlled and reviewable.
# 4. No hallucination risk — an LLM might invent plausible-sounding
#    but incorrect explanations.
#
# Each entry maps: raw_feature_name → (human_readable_name, context_note)
# The context_note provides additional detail when the feature has a
# high SHAP value, making the explanation more actionable.
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    'amount_inr': {
        'name': 'Transaction amount',
        'high': 'Unusually high transaction amount increased risk',
        'low': 'Low transaction amount decreased risk',
    },
    'hour_of_day': {
        'name': 'Transaction hour',
        'high': 'Transaction during unusual hours (late night) increased risk',
        'low': 'Transaction during business hours decreased risk',
    },
    'merchant_txn_count_7d': {
        'name': 'Weekly transaction count',
        'high': 'High transaction count this week increased risk',
        'low': 'Normal transaction volume decreased risk',
    },
    'merchant_avg_amount_7d': {
        'name': "Merchant's average transaction amount",
        'high': "Transaction deviates from merchant's typical amount pattern, increasing risk",
        'low': "Transaction aligns with merchant's typical amount pattern, decreasing risk",
    },
    'device_risk_score': {
        'name': 'Device risk score',
        'high': 'High device risk score (possible emulator/VPN) increased risk',
        'low': 'Clean device profile decreased risk',
    },
    'is_new_merchant': {
        'name': 'Merchant age',
        'high': 'New merchant (less than 7 days) increased risk',
        'low': 'Established merchant decreased risk',
    },
    'payment_method_upi': {
        'name': 'Payment method (UPI)',
        'high': 'UPI payment method increased risk',
        'low': 'UPI payment method decreased risk',
    },
    'payment_method_card': {
        'name': 'Payment method (Card)',
        'high': 'Card payment method increased risk',
        'low': 'Card payment method decreased risk',
    },
    'payment_method_netbanking': {
        'name': 'Payment method (Netbanking)',
        'high': 'Netbanking payment method increased risk',
        'low': 'Netbanking payment method decreased risk',
    },
    'payment_method_wallet': {
        'name': 'Payment method (Wallet)',
        'high': 'Wallet payment method increased risk',
        'low': 'Wallet payment method decreased risk',
    },
}

# ---------------------------------------------------------------------------
# MODULE-LEVEL STATE
# ---------------------------------------------------------------------------
# WHY module-level globals instead of a class?
# For a buildathon scope with a single model, module-level state is simpler
# and more readable than wrapping everything in a class. The API loads this
# module once at startup and calls the functions directly. If we ever need
# multiple models or hot-reloading, we'd refactor to a class — but YAGNI.
# ---------------------------------------------------------------------------
_model = None
_explainer = None
_feature_columns = None


def init_explainer():
    """
    Load the model and initialize SHAP's TreeExplainer.
    
    MUST be called once before explain_prediction(). The API (Stage 6)
    calls this in its startup event handler.
    
    WHY separate init from explain?
    - Model loading + TreeExplainer initialization takes ~0.5-1s.
    - Per-prediction SHAP computation takes ~1-5ms.
    - By separating init from prediction, we pay the startup cost once
      and keep per-request latency low.
    """
    global _model, _explainer, _feature_columns
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run 'python train/train_model.py' first."
        )
    
    _model = joblib.load(MODEL_PATH)
    _feature_columns = joblib.load(FEATURE_COLS_PATH)
    
    # WHY TreeExplainer (not KernelExplainer)?
    # TreeExplainer exploits the tree structure to compute EXACT Shapley
    # values in polynomial time. KernelExplainer treats the model as a
    # black box and approximates via sampling — it's 100-1000x slower
    # and introduces approximation error. Since we KNOW our model is
    # tree-based (XGBoost), TreeExplainer is strictly better.
    _explainer = shap.TreeExplainer(_model)
    
    # expected_value may be a scalar, a length-1 array, or a length-2
    # array depending on XGBoost/SHAP version. Normalize to a float.
    base_val = _explainer.expected_value
    if hasattr(base_val, '__len__'):
        # length-2 → [class_0, class_1], take class_1 (fraud)
        # length-1 → single output, take [0]
        base_val = float(base_val[-1])
    print(f"SHAP TreeExplainer initialized.")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Features: {len(_feature_columns)} columns")
    print(f"  Expected base value (avg predicted fraud prob): {base_val:.4f}")


def explain_prediction(
    transaction_dict: dict,
    top_n: int = 3
) -> list[dict]:
    """
    Generate plain-language explanations for a single prediction.
    
    Args:
        transaction_dict: Raw transaction data as a dictionary, with keys
                         matching the CSV columns (e.g., 'amount_inr',
                         'payment_method', etc.).
        top_n: Number of top contributing features to return (default: 3).
    
    Returns:
        List of dicts, each with:
        - 'feature': raw feature name
        - 'shap_value': signed SHAP value (positive = increases fraud risk)
        - 'reason': plain-language explanation string
        
        Sorted by absolute SHAP value (most impactful first).
    
    Example:
        >>> explain_prediction({'amount_inr': 50000, 'hour_of_day': 2, ...})
        [
            {
                'feature': 'merchant_txn_count_7d',
                'shap_value': 1.23,
                'reason': 'High transaction count this week increased risk'
            },
            ...
        ]
    """
    if _explainer is None:
        raise RuntimeError(
            "Explainer not initialized. Call init_explainer() first."
        )
    
    # --- Preprocess the transaction using the SAME function as training ---
    # WHY? This is the train/serve contract. If training one-hot encodes
    # 'payment_method' into 4 columns, inference must do the exact same
    # transformation. preprocess_features() is the single source of truth.
    df = pd.DataFrame([transaction_dict])
    features = preprocess_features(df)
    
    # Ensure columns match training order exactly.
    # WHY? XGBoost (and SHAP) expect features in the same order as training.
    # If the order differs, SHAP values are assigned to the wrong features
    # and explanations become nonsensical.
    # Missing columns (e.g., a payment method not in this transaction) are
    # filled with 0 — correct because one-hot encoding means "not this method."
    for col in _feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[_feature_columns]
    
    # --- Compute SHAP values ---
    # shap_values shape: (1, n_features) for a single instance.
    # Each value represents how much that feature shifts the prediction
    # away from the base value (average model output on training data).
    #
    # Positive SHAP value = feature pushes prediction toward fraud.
    # Negative SHAP value = feature pushes prediction toward legitimate.
    shap_values = _explainer.shap_values(features)
    
    # Handle both array formats — some SHAP versions return a list of arrays
    # (one per class for multi-class), others return a single array.
    if isinstance(shap_values, list):
        # For binary classification, index [1] is the positive class (fraud).
        shap_vals = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        # Shape: (1, n_features, n_classes) — take class 1
        shap_vals = shap_values[0, :, 1]
    else:
        # Shape: (1, n_features) — single output
        shap_vals = shap_values[0]
    
    # --- Build explanations ---
    # Sort features by absolute SHAP value (most impactful first).
    # We use the actual feature values to generate context-aware reasons.
    feature_values = features.iloc[0]
    feature_impacts = []
    for i, col in enumerate(_feature_columns):
        sv = float(shap_vals[i])
        feat_val = float(feature_values[col])
        
        # Skip one-hot payment method columns where the value is 0.
        # WHY? If payment_method_upi=0 (this txn is NOT UPI), the SHAP
        # value for that column represents "the effect of NOT being UPI."
        # Showing "Non-UPI decreased risk" is technically correct but
        # confusing to a human reviewer. Instead, we only show the
        # payment method that IS active (value=1).
        if col.startswith('payment_method_') and feat_val == 0:
            continue
        
        # Look up the plain-language description.
        desc = FEATURE_DESCRIPTIONS.get(col, {
            'name': col,
            'high': f'{col} increased risk',
            'low': f'{col} decreased risk',
        })
        
        # WHY directional phrasing?
        # "New merchant increased risk" is more actionable than just
        # "New merchant". The direction tells the reviewer whether to
        # focus on this feature or dismiss it.
        if sv > 0:
            reason = desc['high']
        else:
            reason = desc['low']
        
        feature_impacts.append({
            'feature': col,
            'shap_value': round(sv, 4),
            'reason': reason,
        })
    
    # Sort by absolute SHAP value descending — the most impactful
    # features come first, regardless of direction.
    feature_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return feature_impacts[:top_n]


def get_base_value() -> float:
    """
    Return the explainer's expected (base) value.
    
    This is the average model output across the training set — it's the
    prediction you'd get with zero information about a specific transaction.
    Useful for the dashboard to show "model starts at X, then adjusts
    based on these features."
    """
    if _explainer is None:
        raise RuntimeError(
            "Explainer not initialized. Call init_explainer() first."
        )
    base_val = _explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(base_val[-1])
    return float(base_val)


# ---------------------------------------------------------------------------
# STANDALONE TEST — run this file directly to verify SHAP works
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json
    
    print("=" * 65)
    print("STAGE 4 — SHAP Explainability Module Test")
    print("=" * 65)
    
    # Initialize
    init_explainer()
    
    # Test with a suspicious transaction (should trigger multiple patterns)
    suspicious_txn = {
        'transaction_id': 'test_001',
        'merchant_id': 'merch_0042',
        'payment_method': 'upi',
        'amount_inr': 45000.0,
        'hour_of_day': 2,
        'merchant_txn_count_7d': 180,
        'merchant_avg_amount_7d': 1200.0,
        'device_risk_score': 0.92,
        'is_new_merchant': 1,
    }
    
    # Test with a normal transaction (should be low risk)
    normal_txn = {
        'transaction_id': 'test_002',
        'merchant_id': 'merch_0100',
        'payment_method': 'card',
        'amount_inr': 500.0,
        'hour_of_day': 14,
        'merchant_txn_count_7d': 25,
        'merchant_avg_amount_7d': 600.0,
        'device_risk_score': 0.05,
        'is_new_merchant': 0,
    }
    
    for label, txn in [("SUSPICIOUS", suspicious_txn), ("NORMAL", normal_txn)]:
        print(f"\n--- {label} TRANSACTION ---")
        print(f"Input: {json.dumps(txn, indent=2)}")
        
        # Get model prediction
        df = pd.DataFrame([txn])
        features = preprocess_features(df)
        for col in _feature_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[_feature_columns]
        prob = _model.predict_proba(features)[0][1]
        print(f"Fraud probability: {prob:.4f}")
        
        # Get SHAP explanations
        reasons = explain_prediction(txn, top_n=3)
        print(f"Top 3 reasons:")
        for i, r in enumerate(reasons, 1):
            direction = "+" if r['shap_value'] > 0 else ""
            print(f"  {i}. {r['reason']} "
                  f"(SHAP: {direction}{r['shap_value']:.4f})")
    
    print("\n" + "=" * 65)
    print("STAGE 4 TEST COMPLETE")
    print("=" * 65)
