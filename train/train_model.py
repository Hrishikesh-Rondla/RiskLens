"""
=============================================================================
STAGE 3 — Production Model Training (XGBoost + SMOTE)
=============================================================================

PURPOSE:
    Train the fraud detection model on our synthetic_transactions.csv dataset.
    This IS the production model — unlike Stage 2, the artifacts from this
    script are used downstream by the API (Stage 6).

OUTPUTS:
    1. model/risk_model.pkl        — serialized XGBoost model (joblib)
    2. model/feature_columns.pkl   — ordered list of feature column names
                                     (ensures train/serve feature alignment)
    3. train/misclassified_cases.csv — FP/FN cases with all feature values
                                       for panel walkthroughs
    4. stdout: precision, recall, F1, flag rate, confusion matrix

DESIGN DECISIONS:
    - SMOTE applied ONLY to training fold (same discipline as Stage 2).
    - scale_pos_weight left at 1 because SMOTE handles imbalance.
    - Feature preprocessing is extracted into a reusable function
      (preprocess_features) that Stage 6's API will import directly.
      This eliminates train/serve skew — the #1 cause of ML bugs in prod.
    - We save feature_columns.pkl alongside the model so the API can
      guarantee column ordering matches training, even if the input JSON
      has keys in a different order.

USAGE:
    python train/train_model.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# PATHS — resolved relative to project root, not script location,
# so the script works whether called from project root or train/ dir.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'synthetic_transactions.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_model.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')
MISCLASSIFIED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'misclassified_cases.csv'
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# These are the raw feature columns from the CSV.
# transaction_id and merchant_id are identifiers, NOT features.
# WHY exclude merchant_id? Including it would let the model memorize
# which merchants are fraudulent in our synthetic data — that won't
# generalize to new merchants in production. The model should learn
# behavioral patterns (velocity, amount, device risk), not identities.
ID_COLUMNS = ['transaction_id', 'merchant_id']
TARGET_COLUMN = 'is_fraud'

# Confidence tier thresholds — defined here so the training report
# can compute flag rates at these boundaries. These same thresholds
# are used in Stage 5 (tiering.py).
AUTO_ALLOW_CEILING = 0.3    # score < 0.3 → auto_allow
AUTO_BLOCK_FLOOR = 0.7      # score > 0.7 → auto_block


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw transaction data into model-ready features.
    
    THIS FUNCTION IS THE CONTRACT BETWEEN TRAINING AND SERVING.
    The API in Stage 6 will import and call this exact function.
    If you change anything here, the API automatically picks it up —
    no manual sync needed.
    
    WHY one-hot encode payment_method?
    XGBoost CAN handle categoricals natively (via enable_categorical=True),
    but one-hot encoding is more portable — it works with SHAP's
    TreeExplainer without extra configuration, and makes feature names
    interpretable in SHAP plots (e.g., "payment_method_upi" is clearer
    than "category index 2").
    
    WHY not scale/normalize numerical features?
    XGBoost is tree-based — it splits on feature values using inequality
    comparisons (e.g., amount > 5000), so it's invariant to monotonic
    transformations like scaling. Normalizing would add complexity
    without any benefit for tree models. (This would be different for
    logistic regression or neural nets.)
    
    Returns:
        DataFrame with model-ready features, no ID columns, no target.
    """
    # Make a copy to avoid mutating the original DataFrame.
    features = df.copy()
    
    # Drop ID columns if present (they're identifiers, not features).
    for col in ID_COLUMNS:
        if col in features.columns:
            features = features.drop(columns=[col])
    
    # Drop target column if present (during training it's separated;
    # during inference it won't exist in the input).
    if TARGET_COLUMN in features.columns:
        features = features.drop(columns=[TARGET_COLUMN])
    
    # One-hot encode payment_method.
    # WHY drop_first=False? With drop_first=True, we'd have k-1 dummies
    # and the "reference" category would be implicit (all zeros). This
    # makes SHAP explanations confusing — "payment_method_upi has high
    # SHAP value" is clear, but "all payment dummies are zero" is not.
    # The minor multicollinearity cost is irrelevant for tree models.
    if 'payment_method' in features.columns:
        features = pd.get_dummies(
            features,
            columns=['payment_method'],
            prefix='payment_method',
            drop_first=False,
            dtype=int  # XGBoost wants numeric, not bool
        )
    
    # Ensure is_new_merchant is int (0/1), not bool.
    # WHY? XGBoost handles both, but SHAP sometimes has issues with
    # boolean columns in older versions. Explicit int is safer.
    if 'is_new_merchant' in features.columns:
        features['is_new_merchant'] = features['is_new_merchant'].astype(int)
    
    return features


def train_model():
    """
    Full training pipeline: load → preprocess → split → SMOTE → train →
    evaluate → export.
    """
    
    # --- Load data ---
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run 'python data/generate.py' first to create the synthetic dataset.")
        sys.exit(1)
    
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET_COLUMN].mean()*100:.2f}%")
    
    # --- Separate features and target ---
    y = df[TARGET_COLUMN]
    X = preprocess_features(df)
    
    # Save feature column order — the API needs this to guarantee
    # the same column ordering during inference.
    feature_columns = list(X.columns)
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    
    # --- Stratified train/test split ---
    # WHY stratified? With ~4.76% fraud rate and 7,500 rows, a random
    # split could give a test set with anywhere from 3% to 7% fraud.
    # Stratification guarantees the test set mirrors the overall rate,
    # making metrics reproducible and comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Keep the original DataFrame rows for misclassification analysis later.
    # We need the raw features (including IDs) for interpretability.
    test_indices = X_test.index
    df_test = df.loc[test_indices].copy()
    
    print(f"\nTrain set: {len(X_train):,} rows "
          f"(fraud: {y_train.sum()}, {y_train.mean()*100:.2f}%)")
    print(f"Test set:  {len(X_test):,} rows "
          f"(fraud: {y_test.sum()}, {y_test.mean()*100:.2f}%)")
    
    # --- SMOTE: oversample minority class in training set ONLY ---
    # CRITICAL: SMOTE MUST happen AFTER the split, not before.
    # If SMOTE is applied before splitting, synthetic fraud samples
    # can leak into the test set, inflating metrics. This is the most
    # common ML pipeline mistake and a guaranteed panel question.
    print("\nApplying SMOTE to training set only...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_res):,} rows "
          f"(fraud: {y_train_res.sum()}, {y_train_res.mean()*100:.1f}%)")
    
    # --- Train XGBoost ---
    # WHY these hyperparameters?
    # - n_estimators=300: More trees than Stage 2 because our dataset is
    #   smaller (7.5K vs 150K) — we need more iterations to learn subtle
    #   patterns without overfitting (controlled by max_depth + lr).
    # - max_depth=5: Slightly shallower than default (6) to reduce
    #   overfitting risk on 7.5K rows. Depth 5 means at most 2^5=32
    #   leaf nodes per tree — enough to capture our 4 fraud patterns
    #   without memorizing noise.
    # - learning_rate=0.05: Lower than default (0.1) to require more
    #   trees for convergence, which acts as regularization.
    #   Rule of thumb: lower lr + more trees = better generalization.
    # - min_child_weight=5: Minimum 5 samples per leaf node. Prevents
    #   the model from creating leaves that memorize single fraud cases.
    # - subsample=0.8: Each tree sees 80% of the data. Injects randomness
    #   that reduces overfitting (similar to bagging in Random Forest).
    # - colsample_bytree=0.8: Each tree uses 80% of features. Further
    #   regularization + ensures no single feature dominates.
    # - scale_pos_weight=1: NOT adjusted because SMOTE already balances
    #   the classes. Using BOTH would double-correct the imbalance.
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train_res, y_train_res)
    print("Training complete.")
    
    # --- Predict on test set ---
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of fraud
    y_pred = (y_prob >= 0.5).astype(int)         # hard labels at 0.5
    
    # --- Metrics ---
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # --- Flag rate: what fraction of test transactions would be flagged? ---
    # This is an OPERATIONAL metric, not a model metric. If the flag rate
    # is too high (e.g., 30%), the ops team drowns in manual reviews.
    # If too low (e.g., 0.5%), we're missing fraud. A good flag rate for
    # payment fraud is typically 3-8%.
    flag_rate_auto_block = (y_prob > AUTO_BLOCK_FLOOR).mean() * 100
    flag_rate_human_review = (
        (y_prob >= AUTO_ALLOW_CEILING) & (y_prob <= AUTO_BLOCK_FLOOR)
    ).mean() * 100
    flag_rate_auto_allow = (y_prob < AUTO_ALLOW_CEILING).mean() * 100
    
    print("\n" + "=" * 65)
    print("STAGE 3 — MODEL EVALUATION RESULTS (on held-out 20% test set)")
    print("=" * 65)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print("-" * 65)
    print(f"Classification Report (threshold=0.5):")
    print(classification_report(
        y_test, y_pred, target_names=['Legitimate', 'Fraud']
    ))
    print("-" * 65)
    print("OPERATIONAL METRICS — Tier Distribution on Test Set:")
    print(f"  auto_allow  (score < {AUTO_ALLOW_CEILING}): "
          f"{flag_rate_auto_allow:.1f}%")
    print(f"  human_review ({AUTO_ALLOW_CEILING} <= score <= {AUTO_BLOCK_FLOOR}): "
          f"{flag_rate_human_review:.1f}%")
    print(f"  auto_block  (score > {AUTO_BLOCK_FLOOR}): "
          f"{flag_rate_auto_block:.1f}%")
    
    # --- Precision/Recall at tier boundaries ---
    # This tells us: of the transactions we'd auto_block, what fraction
    # are actually fraud? And of all fraud, what fraction do we catch
    # in the auto_block tier?
    auto_block_mask = y_prob > AUTO_BLOCK_FLOOR
    if auto_block_mask.sum() > 0:
        block_precision = y_test[auto_block_mask].mean()
        block_recall = y_test[auto_block_mask].sum() / y_test.sum()
        print(f"\n  auto_block precision: {block_precision:.4f} "
              f"({y_test[auto_block_mask].sum()}/{auto_block_mask.sum()} "
              f"are actual fraud)")
        print(f"  auto_block recall:    {block_recall:.4f} "
              f"(catches {block_recall*100:.1f}% of all fraud)")
    
    human_review_mask = (y_prob >= AUTO_ALLOW_CEILING) & (y_prob <= AUTO_BLOCK_FLOOR)
    if human_review_mask.sum() > 0:
        review_precision = y_test[human_review_mask].mean()
        print(f"  human_review precision: {review_precision:.4f} "
              f"({y_test[human_review_mask].sum()}/{human_review_mask.sum()} "
              f"are actual fraud)")
    
    print("=" * 65)
    
    # --- Feature importance (built-in, not SHAP — that's Stage 4) ---
    # WHY show this in addition to SHAP?
    # Built-in importance is fast and gives a quick sanity check.
    # If a feature the model thinks is important doesn't match our
    # injected patterns, something's wrong with the data generation.
    print("\nFeature Importance (XGBoost gain):")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.iterrows():
        bar = '#' * int(row['importance'] * 50)
        print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # --- Export misclassified cases ---
    # WHY export these?
    # On a panel, you'll be asked "show me a false positive" or "why did
    # the model miss this fraud case?" Having a CSV of misclassified cases
    # with all feature values lets you walk through specific examples
    # instead of giving vague answers.
    print(f"\nExporting misclassified cases...")
    df_test = df_test.copy()
    df_test['predicted_prob'] = y_prob
    df_test['predicted_label'] = y_pred
    df_test['actual_label'] = y_test.values
    
    # Misclassified = prediction != actual
    misclassified = df_test[df_test['predicted_label'] != df_test['actual_label']]
    
    # Add error type for quick filtering
    misclassified = misclassified.copy()
    misclassified['error_type'] = misclassified.apply(
        lambda row: 'false_positive' if row['predicted_label'] == 1 
                    else 'false_negative',
        axis=1
    )
    
    misclassified.to_csv(MISCLASSIFIED_PATH, index=False)
    
    n_fp = (misclassified['error_type'] == 'false_positive').sum()
    n_fn = (misclassified['error_type'] == 'false_negative').sum()
    print(f"  Total misclassified: {len(misclassified)} "
          f"({n_fp} false positives, {n_fn} false negatives)")
    print(f"  Saved to: {MISCLASSIFIED_PATH}")
    
    # --- Save model and feature columns ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLS_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURE_COLS_PATH}")
    
    # --- Final summary ---
    print("\n" + "-" * 65)
    print("STAGE 3 COMPLETE")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Features: {len(feature_columns)} columns")
    print(f"  Misclassified cases: {MISCLASSIFIED_PATH}")
    print(f"  ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, "
          f"auto_block rate={flag_rate_auto_block:.1f}%")
    print("-" * 65)
    
    return model, feature_columns


if __name__ == '__main__':
    train_model()
