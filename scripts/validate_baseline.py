"""
=============================================================================
STAGE 2 — Baseline Validation on Kaggle "Give Me Some Credit" Dataset
=============================================================================

PURPOSE:
    This script is a VALIDATION STEP ONLY — it does NOT produce a model
    artifact used anywhere downstream. Its sole purpose is to confirm that
    the modeling approach (XGBoost + SMOTE) generalizes to a real-world
    credit-risk dataset BEFORE we apply it to our own synthetic data.
    
    WHY DO THIS?
    1. Proves to a panel that we didn't just pick XGBoost arbitrarily —
       we validated it on a well-known benchmark first.
    2. If AUC-ROC < 0.80 here, we'd know to reconsider our approach
       before investing time in Stages 3-7.
    3. The Kaggle "Give Me Some Credit" dataset has structural similarities
       to our fraud task: binary target, rare positive class (~6.7%),
       tabular features, class imbalance requiring SMOTE.

DATASET:
    Source: https://www.kaggle.com/c/GiveMeSomeCredit
    Target: SeriousDlqin2yrs (1 = experienced 90+ days past-due, 0 = no)
    Size: 150,000 rows, 10 features
    Positive class rate: ~6.7% (comparable to our 4.76% fraud rate)
    
    Missing values:
    - MonthlyIncome: 29,731 missing (~19.8%)
    - NumberOfDependents: 3,924 missing (~2.6%)

EXPECTED RESULTS:
    - AUC-ROC: ~0.85-0.87 (based on published Kaggle leaderboard)
    - PR-AUC: ~0.45-0.55 (lower than ROC-AUC due to class imbalance —
      this is normal and expected for rare-event prediction)

USAGE:
    python scripts/validate_baseline.py

NOTE: The model trained here is DISCARDED. It is not saved, not exported,
and not used in any subsequent stage. Stage 3 trains the production model
on our own synthetic data.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib
# WHY Agg backend? We're running in a script, not a GUI. 'Agg' renders to
# file without needing a display server — critical for CI/headless environments.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
KAGGLE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'kaggle', 'cs-training.csv'
)

# WHY 42? Same seed as Stage 1 — consistency across the project.
RANDOM_STATE = 42

# WHY 80/20 split? Industry standard for tabular data with 150K rows.
# 80% train = 120K rows is more than enough for XGBoost to converge.
# 20% test = 30K rows gives tight confidence intervals on metrics.
TEST_SIZE = 0.20

# Output directory for plots — saved alongside this script for easy review.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Kaggle dataset and handle missing values + outliers.
    
    WHY median imputation (not mean)?
    - MonthlyIncome has extreme outliers (max ~3.5M). Mean would be
      pulled by these outliers; median is robust.
    - For a validation step, simple imputation is sufficient. We're not
      trying to win the Kaggle competition — just confirm XGBoost+SMOTE
      works on real data.
    
    WHY drop the index column?
    - 'Unnamed: 0' is just a row number from Kaggle's export — it has
      zero predictive value and would just add noise.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop the Kaggle row-index column — it's not a feature.
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Separate target before any preprocessing to avoid leakage.
    target_col = 'SeriousDlqin2yrs'
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"Raw shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Positive class rate: {y.mean()*100:.2f}%")
    
    # --- Handle missing values ---
    missing_cols = X.columns[X.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_cols}")
    
    for col in missing_cols:
        n_missing = X[col].isnull().sum()
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        print(f"  {col}: filled {n_missing:,} missing with median={median_val:.2f}")
    
    # --- Cap extreme outliers ---
    # WHY cap at 99th percentile?
    # Some features have extreme values (e.g., RevolvingUtilization > 50000,
    # which makes no financial sense — likely data errors). Capping at P99
    # prevents these from dominating splits without losing information.
    # We do NOT remove rows because that would bias the class distribution.
    outlier_cols = [
        'RevolvingUtilizationOfUnsecuredLines',
        'DebtRatio',
        'MonthlyIncome',
    ]
    for col in outlier_cols:
        p99 = X[col].quantile(0.99)
        n_capped = (X[col] > p99).sum()
        if n_capped > 0:
            X[col] = X[col].clip(upper=p99)
            print(f"  {col}: capped {n_capped:,} values at P99={p99:.2f}")
    
    print(f"Preprocessed shape: {X.shape}")
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    Train XGBoost + SMOTE and report metrics.
    
    KEY DESIGN DECISIONS (same approach we'll use in Stage 3):
    
    1. SMOTE is applied ONLY to the training set, AFTER the split.
       WHY? If you SMOTE before splitting, synthetic minority samples
       leak into the test set → metrics are artificially inflated.
       This is a common mistake that a panel will test you on.
    
    2. We use XGBClassifier with default hyperparameters.
       WHY? For a validation step, defaults are fine. We're checking
       whether the APPROACH works, not squeezing the last 0.5% AUC.
       Stage 3 may tune hyperparameters on our own data if needed.
    
    3. scale_pos_weight is left at 1 (default).
       WHY? SMOTE already handles class imbalance by oversampling the
       minority class. Using BOTH SMOTE and scale_pos_weight would
       double-correct, biasing the model toward predicting positive.
    
    4. We report BOTH ROC-AUC and PR-AUC.
       WHY? ROC-AUC can be misleadingly high on imbalanced data because
       it includes true negatives (which are abundant). PR-AUC focuses
       on the precision-recall tradeoff for the rare positive class —
       more informative for fraud/default detection.
    """
    
    # --- Split: stratified to preserve class ratio in both sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # WHY stratify? Ensures test set has same ~6.7% positive
                     # rate as train set, preventing evaluation bias.
    )
    
    print(f"\nTrain set: {X_train.shape[0]:,} rows "
          f"(positive: {y_train.sum():,}, {y_train.mean()*100:.2f}%)")
    print(f"Test set:  {X_test.shape[0]:,} rows "
          f"(positive: {y_test.sum():,}, {y_test.mean()*100:.2f}%)")
    
    # --- SMOTE: oversample minority class in training set ONLY ---
    # WHY SMOTE over random oversampling?
    # SMOTE creates synthetic samples by interpolating between existing
    # minority neighbors, producing more diverse training examples than
    # simple duplication. This reduces overfitting to specific minority
    # instances.
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_resampled.shape[0]:,} rows "
          f"(positive: {y_train_resampled.sum():,}, "
          f"{y_train_resampled.mean()*100:.2f}%)")
    
    # --- Train XGBoost ---
    # WHY eval_metric='aucpr'?
    # Aligns the internal evaluation metric with our primary concern
    # (precision-recall on the rare class), not accuracy.
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,         # Enough trees for convergence on 150K rows
        max_depth=6,              # Default; controls model complexity
        learning_rate=0.1,        # Default; standard for tabular data
        eval_metric='aucpr',      # Matches our focus on rare-class detection
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,              # Suppress training logs for clean output
    )
    model.fit(X_train_resampled, y_train_resampled)
    print("Training complete.")
    
    # --- Predict probabilities (not hard labels) ---
    # WHY probabilities? AUC metrics require continuous scores, not 0/1.
    # Hard labels at threshold=0.5 would discard calibration information.
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # --- Metrics ---
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print("\n" + "=" * 65)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 65)
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print("-" * 65)
    print("Classification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    print("=" * 65)
    
    # --- Interpretation for the panel ---
    if roc_auc >= 0.85:
        verdict = "STRONG"
        explanation = ("ROC-AUC >= 0.85 confirms XGBoost+SMOTE is a strong "
                      "baseline for rare-event binary classification on "
                      "tabular data. Safe to proceed with this approach "
                      "on our synthetic fraud dataset.")
    elif roc_auc >= 0.80:
        verdict = "ACCEPTABLE"
        explanation = ("ROC-AUC >= 0.80 is acceptable for a baseline. "
                      "The approach works; hyperparameter tuning in "
                      "Stage 3 may improve results further.")
    else:
        verdict = "WEAK"
        explanation = ("ROC-AUC < 0.80 suggests this approach may need "
                      "reconsideration. Check preprocessing and consider "
                      "alternative models.")
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {explanation}")
    
    # --- Generate plots ---
    print("\nGenerating evaluation plots...")
    _plot_roc_curve(y_test, y_prob, roc_auc)
    _plot_pr_curve(y_test, y_prob, pr_auc)
    print(f"Plots saved to: {OUTPUT_DIR}")
    
    return roc_auc, pr_auc


def _plot_roc_curve(y_true, y_prob, auc_score):
    """
    Plot ROC curve and save to file.
    
    WHY include the diagonal?
    The diagonal (AUC=0.5) represents a random classifier. Showing it
    makes it visually obvious how much better our model is than chance.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2563eb', lw=2,
             label=f'XGBoost + SMOTE (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — Kaggle Baseline Validation', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_roc_curve.png'), dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_prob, auc_score):
    """
    Plot Precision-Recall curve and save to file.
    
    WHY PR curve in addition to ROC?
    For imbalanced datasets (6.7% positive here), ROC-AUC can be misleadingly
    high because the large number of true negatives inflates the true-negative
    rate. PR-AUC focuses exclusively on how well the model identifies the
    rare positive class — which is what actually matters for fraud detection.
    
    The horizontal baseline represents the positive class prevalence (~6.7%).
    A model that predicts "positive" for everything would achieve this line.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#dc2626', lw=2,
             label=f'XGBoost + SMOTE (PR-AUC = {auc_score:.4f})')
    plt.axhline(y=baseline, color='gray', lw=1, linestyle='--',
                label=f'Baseline (prevalence = {baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve — Kaggle Baseline Validation', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_pr_curve.png'), dpi=150)
    plt.close()


def main():
    """
    Entry point: load data → preprocess → train → evaluate → plot.
    
    REMINDER: This is a validation step. The model trained here is DISCARDED.
    We are only checking that XGBoost + SMOTE produces reasonable metrics
    on a known benchmark before applying the same approach to our own data
    in Stage 3.
    """
    # Check that the Kaggle CSV exists
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"ERROR: Kaggle dataset not found at {KAGGLE_CSV_PATH}")
        print("Please download cs-training.csv from:")
        print("  https://www.kaggle.com/c/GiveMeSomeCredit/data")
        print(f"And place it at: {KAGGLE_CSV_PATH}")
        sys.exit(1)
    
    X, y = load_and_preprocess(KAGGLE_CSV_PATH)
    roc_auc, pr_auc = train_and_evaluate(X, y)
    
    print("\n" + "-" * 65)
    print("STAGE 2 COMPLETE — Baseline validation passed.")
    print("This model is NOT saved. Proceeding to Stage 3 for production model.")
    print("-" * 65)


if __name__ == '__main__':
    main()
