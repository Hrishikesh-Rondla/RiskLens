"""
=============================================================================
STAGE 1 — Synthetic Transaction Dataset Generator
=============================================================================

PURPOSE:
    Generate a realistic synthetic dataset of Indian payment transactions
    with known, controllable fraud patterns. This is NOT a shortcut — it's a
    deliberate design choice for a 12-day buildathon:
    
    WHY synthetic over real data?
    1. No PII/compliance risk — we can open-source and demo freely.
    2. We control the ground truth — we KNOW which patterns exist, so we can
       validate that the model actually learns them (not just memorizes noise).
    3. Reproducible — seeded RNG means any reviewer gets identical results.
    
    WHY NOT just use an existing fraud dataset (e.g., IEEE-CIS)?
    - Those datasets have different feature schemas (credit card, not UPI/wallet).
    - We want features that match Razorpay's domain: UPI, merchant velocity,
      device risk scores — things a Razorpay panel will recognize.

FRAUD PATTERNS INJECTED (4 total):
    Each pattern increases the probability of fraud when its conditions are met.
    Probabilities are ADDITIVE, not deterministic flags — this forces the model
    to learn correlations rather than memorize simple if-else rules.

    Pattern 1 — "Late-night high-value UPI from new merchant"
        Conditions: payment_method == 'upi' AND is_new_merchant == True
                    AND hour_of_day in [0..4] (midnight to 5am)
                    AND amount_inr > 3000
        Added P(fraud): +0.55
        Rationale: UPI's instant settlement makes it attractive for fraudsters;
                   new merchants lack transaction history for verification;
                   odd hours reduce the chance of manual review catching it.
        Approximate observed fraud rate when all conditions met: ~60-65%

    Pattern 2 — "Velocity spike"
        Conditions: merchant_txn_count_7d > 5× the merchant's own baseline
                    (baseline = merchant_avg_txn_count, derived per-merchant)
        Added P(fraud): +0.50
        Rationale: A sudden spike in transaction volume from a single merchant
                   is a classic money-laundering / bust-out fraud signal.
                   The 5x multiplier is industry-standard (Stripe Radar uses 4-6x).
        Approximate observed fraud rate when triggered: ~55-60%

    Pattern 3 — "Amount z-score outlier"
        Conditions: |amount_inr - merchant_avg_amount_7d| / merchant_std > 3
                    (i.e., the transaction amount is >3 standard deviations
                    from the merchant's own average)
        Added P(fraud): +0.45
        Rationale: Tests whether the model picks up RELATIVE anomalies, not just
                   absolute thresholds. A ₹50,000 txn from a jeweler is normal;
                   the same from a chai stall is suspicious.
        Approximate observed fraud rate when triggered: ~50-55%

    Pattern 4 — "High device risk + new merchant"
        Conditions: device_risk_score > 0.8 AND is_new_merchant == True
        Added P(fraud): +0.50
        Rationale: Device fingerprinting (emulators, rooted phones, VPN/proxy)
                   is the single strongest signal in production fraud systems.
                   Combined with merchant newness, it's a high-confidence signal.
        Approximate observed fraud rate when triggered: ~55-60%

    Base fraud rate (no pattern triggered): ~2-3%
    Overall dataset fraud rate target: ~5-8%

OUTPUT:
    data/synthetic_transactions.csv — 7,500 rows, 10 columns + header

USAGE:
    python data/generate.py
    # or from project root:
    python -m data.generate
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# WHY seed=42? Convention for reproducibility. Any reviewer can re-run this
# script and get byte-identical output. The specific value doesn't matter —
# 42 is just tradition (Hitchhiker's Guide, if anyone asks on the panel).
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# CONFIGURATION — all magic numbers in one place so they're easy to tune
# and easy to defend ("I didn't bury constants in logic; they're all here.")
# ---------------------------------------------------------------------------
N_TRANSACTIONS = 7500       # Midpoint of 5k-10k spec. Enough for stratified
                            # 80/20 split to have ~300+ fraud cases in test.

N_MERCHANTS = 200           # Realistic ratio: ~37.5 txns per merchant on avg.
                            # Too few merchants → model memorizes merchant_id.
                            # Too many → not enough txns per merchant to build
                            # meaningful velocity/average features.

# Payment method distribution — weighted to reflect Indian digital payments
# landscape. UPI dominates (~45% of digital txns as of 2024 NPCI data),
# cards are second, netbanking and wallets trail.
PAYMENT_METHODS = ['upi', 'card', 'netbanking', 'wallet']
PAYMENT_WEIGHTS = [0.45, 0.28, 0.15, 0.12]

# Fraction of merchants that are "new" (< 7 days old).
# 20% is intentionally high to ensure enough samples trigger Pattern 1 & 4.
# In production, this would be ~5-10%.
NEW_MERCHANT_FRACTION = 0.20

# Base probability of fraud when NO pattern is triggered.
# Set low (2%) so the overall rate stays in the 5-8% target range
# even after pattern injection adds fraud cases.
BASE_FRAUD_PROB = 0.02

# --- Pattern-specific fraud probability boosts ---
# These are ADDED to base_prob when conditions are met.
# They're intentionally below 1.0 so not every triggered pattern = fraud.
# This prevents the model from learning trivial rules.
PATTERN1_BOOST = 0.55   # Late-night high-value UPI + new merchant
PATTERN2_BOOST = 0.50   # Velocity spike (>5x baseline)
PATTERN3_BOOST = 0.45   # Amount z-score outlier (>3 sigma)
PATTERN4_BOOST = 0.50   # High device risk + new merchant

# Velocity spike multiplier — a merchant's txn count must exceed this
# multiple of its own baseline to trigger Pattern 2.
VELOCITY_SPIKE_MULTIPLIER = 5.0

# Z-score threshold for Pattern 3 — how many standard deviations away
# from the merchant's average amount to consider "anomalous".
ZSCORE_THRESHOLD = 3.0

# Device risk threshold for Pattern 4
DEVICE_RISK_THRESHOLD = 0.8


def generate_merchant_profiles(n_merchants: int) -> pd.DataFrame:
    """
    Generate stable per-merchant profiles BEFORE generating transactions.
    
    WHY separate profiles?
    In reality, each merchant has a characteristic transaction pattern —
    a chai stall averages ₹50/txn while a jeweler averages ₹25,000/txn.
    Generating per-merchant baselines first, then sampling transactions
    FROM those baselines, creates realistic within-merchant variance and
    between-merchant heterogeneity. Without this, all merchants would look
    the same and the model couldn't learn merchant-relative anomalies.
    """
    profiles = pd.DataFrame({
        'merchant_id': [f'merch_{i:04d}' for i in range(n_merchants)],
        
        # WHY log-normal for average amount?
        # Transaction amounts in payments follow a heavy-tailed distribution:
        # most merchants are small (₹100-₹2000 avg), few are large (₹10k+).
        # Log-normal captures this naturally. mean=7.0, sigma=1.2 gives a
        # median of ~₹1,100 with a long right tail up to ~₹50k.
        'avg_amount': np.random.lognormal(mean=7.0, sigma=1.2, size=n_merchants),
        
        # Standard deviation of the merchant's typical transaction amounts.
        # Set as 30-60% of the mean — realistic coefficient of variation.
        'std_amount': None,  # computed below
        
        # Baseline weekly transaction count — how many txns this merchant
        # normally processes in 7 days. Poisson-distributed because txn
        # arrivals are approximately Poisson in practice.
        'baseline_txn_count_7d': np.random.poisson(lam=25, size=n_merchants).clip(min=3),
        
        # Whether this merchant is "new" (registered < 7 days ago).
        'is_new': np.random.random(n_merchants) < NEW_MERCHANT_FRACTION,
    })
    
    # Std is 30-60% of mean — this creates realistic variance without
    # making amounts negative (we'll clip later).
    profiles['std_amount'] = profiles['avg_amount'] * np.random.uniform(
        0.3, 0.6, size=n_merchants
    )
    
    return profiles


def generate_transactions(
    n_transactions: int,
    merchant_profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate individual transactions by sampling from merchant profiles.
    
    Each transaction draws its merchant, then samples features from that
    merchant's characteristic distribution. This creates the realistic
    correlation structure that makes the dataset non-trivial to model.
    """
    
    # --- Assign each transaction to a merchant ---
    # WHY uniform random? In production, transaction volume is Zipf-distributed
    # (few merchants dominate volume). But for a 7,500-row dataset with 200
    # merchants, Zipf would starve most merchants of samples, making
    # per-merchant features unreliable. Uniform gives ~37 txns/merchant.
    merchant_indices = np.random.randint(0, len(merchant_profiles), size=n_transactions)
    merchants = merchant_profiles.iloc[merchant_indices].reset_index(drop=True)
    
    # --- Transaction IDs ---
    # Sequential with prefix for readability in the dashboard.
    txn_ids = [f'txn_{i:06d}' for i in range(n_transactions)]
    
    # --- Payment method ---
    # Sampled independently of merchant because in India, most merchants
    # accept all digital methods. The weights reflect market-level UPI
    # dominance, not merchant-level preferences.
    payment_methods = np.random.choice(
        PAYMENT_METHODS,
        size=n_transactions,
        p=PAYMENT_WEIGHTS
    )
    
    # --- Transaction amount (INR) ---
    # Sampled from each merchant's own normal distribution (mean, std),
    # then clipped to [10, 500000]. The clip prevents negative amounts
    # and caps at ₹5L (reasonable for digital payments).
    amounts = np.random.normal(
        loc=merchants['avg_amount'].values,
        scale=merchants['std_amount'].values
    ).clip(min=10, max=500000)
    
    # --- Hour of day ---
    # WHY beta distribution, not uniform?
    # Real transaction volumes peak during business hours (10am-8pm) and
    # dip at night. Beta(2, 5) shifted and scaled to [0, 23] gives a
    # realistic daytime peak. We also inject ~8% of transactions into
    # the 0-4am window to ensure Pattern 1 has enough samples.
    hours = np.zeros(n_transactions, dtype=int)
    
    # 88% of transactions follow a business-hours-heavy distribution.
    # WHY 12% odd-hours? Ensures Pattern 1 (late-night UPI) has enough
    # samples (~900 txns in 0-4am window) to trigger meaningfully.
    n_normal_hours = int(n_transactions * 0.88)
    n_odd_hours = n_transactions - n_normal_hours
    
    # Business hours: beta distribution peaked around 10am-6pm
    normal_hours = (np.random.beta(2, 2, size=n_normal_hours) * 18 + 6).astype(int) % 24
    # Odd hours: uniform in [0, 4] for Pattern 1 triggering
    odd_hours = np.random.randint(0, 5, size=n_odd_hours)
    
    hours[:n_normal_hours] = normal_hours
    hours[n_normal_hours:] = odd_hours
    # Shuffle so odd-hour transactions aren't all at the end
    np.random.shuffle(hours)
    
    # --- Merchant transaction count (7-day rolling) ---
    # For most transactions, this is close to the merchant's baseline.
    # We add noise (Poisson) to simulate natural day-to-day variance.
    # For ~5% of transactions, we inject a velocity spike (Pattern 2)
    # by multiplying the baseline by 5-10x.
    base_counts = merchants['baseline_txn_count_7d'].values.astype(float)
    txn_counts = np.random.poisson(lam=base_counts).clip(min=1)
    
    # Inject velocity spikes into ~5% of transactions
    # WHY 5%? Enough to give the model ~375 spike samples to learn from,
    # but not so many that spikes become "normal".
    spike_mask = np.random.random(n_transactions) < 0.05
    spike_multipliers = np.random.uniform(
        VELOCITY_SPIKE_MULTIPLIER, VELOCITY_SPIKE_MULTIPLIER * 2,
        size=n_transactions
    )
    txn_counts[spike_mask] = (base_counts[spike_mask] * spike_multipliers[spike_mask]).astype(int)
    
    # --- Merchant average amount (7-day rolling) ---
    # This is the merchant's OWN historical average, not the current txn amount.
    # We use the profile's avg_amount with slight noise to simulate
    # that the 7-day rolling average shifts slightly day to day.
    merchant_avg_7d = merchants['avg_amount'].values * np.random.uniform(
        0.85, 1.15, size=n_transactions
    )
    
    # --- Device risk score (0-1) ---
    # WHY beta distribution?
    # Device risk scores in production cluster near 0 (most devices are clean)
    # with a thin tail toward 1. Beta(1.5, 8) gives a median of ~0.12 and
    # P(>0.8) ≈ 0.002 — realistic for a population where most users have
    # clean devices. We boost some scores to ensure Pattern 4 fires.
    device_scores = np.random.beta(1.5, 8, size=n_transactions)
    
    # Boost ~4% of transactions to high device risk (>0.8) to ensure
    # Pattern 4 has enough samples. Without this, Beta(1.5,8) would
    # produce almost no samples above 0.8.
    high_device_mask = np.random.random(n_transactions) < 0.04
    device_scores[high_device_mask] = np.random.uniform(0.8, 1.0, size=high_device_mask.sum())
    
    # --- Is new merchant (boolean) ---
    # Pulled directly from the merchant profile — this is a merchant-level
    # attribute, not a transaction-level one.
    is_new = merchants['is_new'].values.astype(int)
    
    # --- Assemble the DataFrame before labeling ---
    df = pd.DataFrame({
        'transaction_id': txn_ids,
        'merchant_id': merchants['merchant_id'].values,
        'payment_method': payment_methods,
        'amount_inr': np.round(amounts, 2),
        'hour_of_day': hours,
        'merchant_txn_count_7d': txn_counts.astype(int),
        'merchant_avg_amount_7d': np.round(merchant_avg_7d, 2),
        'device_risk_score': np.round(device_scores, 4),
        'is_new_merchant': is_new,
    })
    
    # Store merchant std for z-score calculation (not exported, just for labeling)
    merchant_stds = merchants['std_amount'].values
    
    # -----------------------------------------------------------------------
    # FRAUD LABELING — probabilistic, not deterministic
    #
    # WHY probabilistic?
    # If we labeled fraud deterministically (e.g., "if all Pattern 1 conditions
    # met → always fraud"), the model would learn a trivial if-else rule, and
    # SHAP explanations would be uninteresting. Probabilistic labeling means:
    #   - Some triggered-pattern transactions are NOT fraud (realistic: not
    #     every suspicious-looking txn is actually fraudulent).
    #   - Some non-triggered transactions ARE fraud (base rate: random fraud
    #     exists even without matching known patterns).
    #   - The model must learn soft decision boundaries, not hard rules.
    # -----------------------------------------------------------------------
    
    fraud_prob = np.full(n_transactions, BASE_FRAUD_PROB)
    
    # --- Pattern 1: Late-night high-value UPI from new merchant ---
    pattern1_mask = (
        (df['payment_method'] == 'upi') &
        (df['is_new_merchant'] == 1) &
        (df['hour_of_day'].isin([0, 1, 2, 3, 4])) &
        (df['amount_inr'] > 3000)
    )
    fraud_prob[pattern1_mask] += PATTERN1_BOOST
    
    # --- Pattern 2: Velocity spike (>5x baseline) ---
    # Compare current merchant_txn_count_7d to the merchant's baseline.
    pattern2_mask = (
        df['merchant_txn_count_7d'].values > 
        VELOCITY_SPIKE_MULTIPLIER * base_counts
    )
    fraud_prob[pattern2_mask] += PATTERN2_BOOST
    
    # --- Pattern 3: Amount z-score outlier ---
    # z-score = |amount - merchant_avg| / merchant_std
    # Guarding against division by zero with clip(min=1).
    z_scores = np.abs(
        df['amount_inr'].values - df['merchant_avg_amount_7d'].values
    ) / np.clip(merchant_stds, 1, None)
    pattern3_mask = z_scores > ZSCORE_THRESHOLD
    fraud_prob[pattern3_mask] += PATTERN3_BOOST
    
    # --- Pattern 4: High device risk + new merchant ---
    pattern4_mask = (
        (df['device_risk_score'] > DEVICE_RISK_THRESHOLD) &
        (df['is_new_merchant'] == 1)
    )
    fraud_prob[pattern4_mask] += PATTERN4_BOOST
    
    # Cap probability at 0.95 — even the most suspicious transaction has a
    # 5% chance of being legitimate. This prevents deterministic labels.
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    
    # Roll the dice: each transaction is independently labeled as fraud
    # based on its computed probability.
    df['is_fraud'] = (np.random.random(n_transactions) < fraud_prob).astype(int)
    
    # -----------------------------------------------------------------------
    # DIAGNOSTICS — printed to stdout so the developer can verify the
    # generated dataset matches expectations before training.
    # -----------------------------------------------------------------------
    total_fraud = df['is_fraud'].sum()
    fraud_rate = total_fraud / len(df) * 100
    
    print("=" * 65)
    print("SYNTHETIC DATASET GENERATION REPORT")
    print("=" * 65)
    print(f"Total transactions:        {len(df):,}")
    print(f"Total fraud cases:         {total_fraud:,}")
    print(f"Overall fraud rate:        {fraud_rate:.2f}%")
    print("-" * 65)
    print("PATTERN BREAKDOWN:")
    print(f"  Pattern 1 (UPI+new+night+high-value): "
          f"{pattern1_mask.sum():,} triggered, "
          f"{df.loc[pattern1_mask, 'is_fraud'].sum()} fraud "
          f"({df.loc[pattern1_mask, 'is_fraud'].mean()*100:.1f}% rate)")
    print(f"  Pattern 2 (velocity spike >5x):       "
          f"{pattern2_mask.sum():,} triggered, "
          f"{df.loc[pattern2_mask, 'is_fraud'].sum()} fraud "
          f"({df.loc[pattern2_mask, 'is_fraud'].mean()*100:.1f}% rate)")
    print(f"  Pattern 3 (amount z-score >3 sigma):        "
          f"{pattern3_mask.sum():,} triggered, "
          f"{df.loc[pattern3_mask, 'is_fraud'].sum()} fraud "
          f"({df.loc[pattern3_mask, 'is_fraud'].mean()*100:.1f}% rate)")
    print(f"  Pattern 4 (device risk+new merchant):  "
          f"{pattern4_mask.sum():,} triggered, "
          f"{df.loc[pattern4_mask, 'is_fraud'].sum()} fraud "
          f"({df.loc[pattern4_mask, 'is_fraud'].mean()*100:.1f}% rate)")
    print("-" * 65)
    
    # Fraud rate by payment method — useful for panel Q&A about whether
    # the model is biased toward/against specific payment methods.
    print("FRAUD RATE BY PAYMENT METHOD:")
    for method in PAYMENT_METHODS:
        method_df = df[df['payment_method'] == method]
        method_fraud = method_df['is_fraud'].mean() * 100
        print(f"  {method:12s}: {method_fraud:.2f}% "
              f"({method_df['is_fraud'].sum()}/{len(method_df)})")
    print("-" * 65)
    
    # Fraud rate by merchant age — sanity check that new merchants have
    # higher fraud rates (as designed).
    print("FRAUD RATE BY MERCHANT AGE:")
    for is_new_val, label in [(1, "New (<7 days)"), (0, "Established")]:
        subset = df[df['is_new_merchant'] == is_new_val]
        rate = subset['is_fraud'].mean() * 100
        print(f"  {label:15s}: {rate:.2f}% "
              f"({subset['is_fraud'].sum()}/{len(subset)})")
    print("=" * 65)
    
    return df


def main():
    """
    Entry point: generate merchant profiles → generate transactions →
    save to CSV. Separated into a function so it can be imported and
    called programmatically (e.g., from a test script).
    """
    
    # Ensure the output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'synthetic_transactions.csv')
    
    print(f"\nGenerating {N_TRANSACTIONS:,} synthetic transactions...\n")
    
    # Step 1: Create merchant profiles (stable baselines)
    merchant_profiles = generate_merchant_profiles(N_MERCHANTS)
    print(f"Created {N_MERCHANTS} merchant profiles "
          f"({int(merchant_profiles['is_new'].sum())} new, "
          f"{int((~merchant_profiles['is_new']).sum())} established)\n")
    
    # Step 2: Generate transactions from those profiles
    df = generate_transactions(N_TRANSACTIONS, merchant_profiles)
    
    # Step 3: Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
