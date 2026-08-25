"""
=============================================================================
STAGE 5 — Confidence Tier Logic
=============================================================================

PURPOSE:
    Map a continuous risk score (0-1 probability from XGBoost) to one of
    three discrete action tiers: auto_allow, human_review, auto_block.
    
    WHY TIERS (not just a single threshold)?
    A single threshold (e.g., "block if score > 0.5") forces a binary
    decision: block or allow. In reality, there's a middle zone where the
    model is genuinely uncertain and a human reviewer can add value.
    Three tiers capture this:
    
    - auto_allow:   Model is confident the txn is legitimate → let it through
    - human_review: Model is uncertain → route to a human for judgment
    - auto_block:   Model is confident the txn is fraud → block it
    
    This maps directly to how payment processors operate in production:
    Razorpay, Stripe, and PayPal all use tiered decisioning (sometimes
    with more than 3 tiers — e.g., Stripe Radar has 5 risk levels).

THRESHOLD JUSTIFICATION (from Stage 3 evaluation):
    These thresholds are NOT arbitrary round numbers. They were chosen
    based on the precision/recall tradeoffs observed on the held-out test
    set in Stage 3:
    
    auto_block (score > 0.7):
        - 3.3% of test transactions fell in this tier
        - Precision: 51.0% (1 in 2 blocked txns is actual fraud)
        - Recall: 35.2% (catches ~1/3 of all fraud automatically)
        - WHY 0.7? Below this, precision drops sharply — at 0.5, precision
          is only 38%, meaning we'd block too many legitimate transactions.
          0.7 keeps auto-block actions high-confidence.
    
    human_review (0.3 <= score <= 0.7):
        - 7.0% of test transactions fell in this tier
        - Precision: 12.4% (1 in 8 reviewed txns is fraud)
        - This is the "uncertain" band — the model sees some risk signals
          but not enough to act automatically. Human review catches fraud
          the model alone would miss.
        - WHY 0.3 as the lower bound? Below 0.3, the fraud rate drops to
          near the base rate (~2-3%), making review not cost-effective.
    
    auto_allow (score < 0.3):
        - 89.7% of test transactions fell in this tier
        - The vast majority of transactions are legitimate and pass through
          without delay — critical for payment UX.
    
    OPERATIONAL TRADEOFF:
        Higher auto_block threshold → fewer false blocks, but more fraud
        gets through to human_review (or worse, to auto_allow).
        Lower auto_block threshold → catches more fraud, but blocks more
        legitimate transactions (higher ops cost, worse merchant UX).
        
        The 0.3/0.7 split balances these tensions for a typical payment
        processor. In production, these would be tuned per merchant segment
        (e.g., high-risk MCCs might use 0.2/0.6).

USAGE:
    from model.tiering import assign_tier, AUTO_ALLOW_CEILING, AUTO_BLOCK_FLOOR
    
    tier = assign_tier(0.82)  # returns 'auto_block'
    tier = assign_tier(0.45)  # returns 'human_review'
    tier = assign_tier(0.15)  # returns 'auto_allow'
=============================================================================
"""

# ---------------------------------------------------------------------------
# THRESHOLD CONSTANTS — named and top-level for easy tuning during demos.
#
# WHY named constants (not function parameters with defaults)?
# 1. Constants are discoverable — a reviewer can grep for them instantly.
# 2. Constants are consistent — the API (Stage 6) and training (Stage 3)
#    both import from here, so there's a single source of truth.
# 3. Constants are tunable — during a live demo, you can change these
#    two numbers and restart the server to see immediate impact on
#    tier distribution.
# ---------------------------------------------------------------------------

# Transactions with risk score BELOW this value are automatically allowed.
# Based on Stage 3: below 0.3, fraud rate is near base rate (~2-3%),
# making manual review not cost-effective.
AUTO_ALLOW_CEILING = 0.3

# Transactions with risk score ABOVE this value are automatically blocked.
# Based on Stage 3: above 0.7, precision is ~51% — high enough to justify
# automatic action without human intervention.
AUTO_BLOCK_FLOOR = 0.7

# Tier name constants — avoids string typos across the codebase.
TIER_AUTO_ALLOW = 'auto_allow'
TIER_HUMAN_REVIEW = 'human_review'
TIER_AUTO_BLOCK = 'auto_block'

# Color codes for dashboard display — defined here so the frontend and
# any other consumer use consistent colors.
TIER_COLORS = {
    TIER_AUTO_ALLOW: '#22c55e',     # Green — safe, let it through
    TIER_HUMAN_REVIEW: '#f59e0b',   # Amber — needs attention
    TIER_AUTO_BLOCK: '#ef4444',     # Red — blocked
}

# Human-readable labels for display.
TIER_LABELS = {
    TIER_AUTO_ALLOW: 'Auto Allow',
    TIER_HUMAN_REVIEW: 'Human Review',
    TIER_AUTO_BLOCK: 'Auto Block',
}


def assign_tier(risk_score: float) -> str:
    """
    Map a risk score to an action tier.
    
    Args:
        risk_score: Float in [0, 1] — probability of fraud from the model.
        
    Returns:
        One of: 'auto_allow', 'human_review', 'auto_block'
    
    Boundary behavior:
        - score == AUTO_ALLOW_CEILING (0.3) → human_review (conservative:
          at the boundary, we prefer review over auto-allow)
        - score == AUTO_BLOCK_FLOOR (0.7) → human_review (conservative:
          at the boundary, we prefer review over auto-block)
        
    WHY conservative boundaries?
        False blocks (blocking legitimate transactions) damage merchant
        trust and revenue. False allows (letting fraud through) cause
        financial loss. At the exact boundary, both risks are present,
        so routing to human review is the safest default.
    """
    if risk_score < AUTO_ALLOW_CEILING:
        return TIER_AUTO_ALLOW
    elif risk_score > AUTO_BLOCK_FLOOR:
        return TIER_AUTO_BLOCK
    else:
        return TIER_HUMAN_REVIEW


def get_tier_info(tier: str) -> dict:
    """
    Return display metadata for a tier.
    
    Args:
        tier: One of 'auto_allow', 'human_review', 'auto_block'
    
    Returns:
        Dict with 'tier', 'label', 'color' keys.
        
    Used by the API response to include display info that the dashboard
    can use directly without maintaining its own mapping.
    """
    return {
        'tier': tier,
        'label': TIER_LABELS.get(tier, tier),
        'color': TIER_COLORS.get(tier, '#6b7280'),  # gray fallback
    }


def get_all_tiers() -> list[dict]:
    """
    Return metadata for all tiers, useful for the dashboard legend.
    """
    return [
        {
            'tier': TIER_AUTO_ALLOW,
            'label': TIER_LABELS[TIER_AUTO_ALLOW],
            'color': TIER_COLORS[TIER_AUTO_ALLOW],
            'threshold': f'score < {AUTO_ALLOW_CEILING}',
        },
        {
            'tier': TIER_HUMAN_REVIEW,
            'label': TIER_LABELS[TIER_HUMAN_REVIEW],
            'color': TIER_COLORS[TIER_HUMAN_REVIEW],
            'threshold': f'{AUTO_ALLOW_CEILING} <= score <= {AUTO_BLOCK_FLOOR}',
        },
        {
            'tier': TIER_AUTO_BLOCK,
            'label': TIER_LABELS[TIER_AUTO_BLOCK],
            'color': TIER_COLORS[TIER_AUTO_BLOCK],
            'threshold': f'score > {AUTO_BLOCK_FLOOR}',
        },
    ]


# ---------------------------------------------------------------------------
# STANDALONE TEST — boundary value testing
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 55)
    print("STAGE 5 — Confidence Tier Logic Test")
    print("=" * 55)
    print(f"AUTO_ALLOW_CEILING: {AUTO_ALLOW_CEILING}")
    print(f"AUTO_BLOCK_FLOOR:   {AUTO_BLOCK_FLOOR}")
    print("-" * 55)
    
    # Boundary value tests — the most important cases to verify.
    test_cases = [
        (0.00, TIER_AUTO_ALLOW,   "Minimum score"),
        (0.15, TIER_AUTO_ALLOW,   "Mid auto_allow"),
        (0.29, TIER_AUTO_ALLOW,   "Just below ceiling"),
        (0.30, TIER_HUMAN_REVIEW, "Exactly at ceiling (conservative -> review)"),
        (0.50, TIER_HUMAN_REVIEW, "Mid human_review"),
        (0.70, TIER_HUMAN_REVIEW, "Exactly at floor (conservative -> review)"),
        (0.71, TIER_AUTO_BLOCK,   "Just above floor"),
        (0.85, TIER_AUTO_BLOCK,   "Mid auto_block"),
        (1.00, TIER_AUTO_BLOCK,   "Maximum score"),
    ]
    
    all_passed = True
    for score, expected_tier, description in test_cases:
        actual_tier = assign_tier(score)
        status = "PASS" if actual_tier == expected_tier else "FAIL"
        if status == "FAIL":
            all_passed = False
        
        info = get_tier_info(actual_tier)
        print(f"  {status} | score={score:.2f} | "
              f"tier={actual_tier:14s} | {description}")
    
    print("-" * 55)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    print("\nTier metadata:")
    for tier_info in get_all_tiers():
        print(f"  {tier_info['label']:14s} | "
              f"{tier_info['threshold']:24s} | "
              f"color={tier_info['color']}")
    
    print("=" * 55)
