/**
 * ==========================================================================
 * STAGE 7 — Dashboard JavaScript (Vanilla JS, no framework)
 * ==========================================================================
 *
 * PURPOSE:
 *   Polls GET /flagged-transactions every 3 seconds and renders scored
 *   transactions in a color-coded table. Also provides a "Simulate
 *   Transaction" button that generates random transactions via POST
 *   /score-transaction.
 *
 * WHY VANILLA JS (no React/Vue/Svelte)?
 *   Spec says "dependency-light." For a single-page polling dashboard,
 *   vanilla JS with DOM manipulation is simpler, faster to load, and
 *   has zero build step. The entire frontend is two files (HTML + JS).
 *
 * WHY POLLING (not WebSocket)?
 *   1. Simpler to implement — no connection lifecycle management.
 *   2. Easier to debug — you can see the requests in browser DevTools.
 *   3. 3-second interval is fine for a demo dashboard.
 *   4. WebSocket would be better at scale (10,000+ concurrent dashboards),
 *      but for a buildathon demo with 1-5 viewers, polling is sufficient.
 *
 * AUTH:
 *   Uses a hardcoded dev token obtained from POST /api/generate-token
 *   at page load. This is a demo tool — production would use proper
 *   auth UI with login/logout flows.
 * ==========================================================================
 */

// ---------------------------------------------------------------------------
// STATE
// ---------------------------------------------------------------------------
let authToken = null;          // JWT token, fetched on page load
let allTransactions = [];      // Full transaction list from API
let currentFilter = 'all';    // Active tier filter
let pollInterval = null;       // setInterval reference
const POLL_MS = 3000;          // Poll every 3 seconds

// ---------------------------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    await initAuth();
    startPolling();
});

/**
 * Fetch a JWT token from the /api/generate-token endpoint.
 * 
 * WHY auto-fetch instead of hardcoding?
 * Tokens expire (24h). Auto-fetching on page load guarantees a fresh
 * token every time the dashboard is opened. The token endpoint has no
 * auth (it's a demo convenience) — see Stage 6 comments for the
 * production alternative.
 */
async function initAuth() {
    try {
        const resp = await fetch('/api/generate-token', { method: 'POST' });
        const data = await resp.json();
        authToken = data.token;
        updateStatus(true);
    } catch (err) {
        console.error('Auth initialization failed:', err);
        updateStatus(false);
    }
}

// ---------------------------------------------------------------------------
// POLLING
// ---------------------------------------------------------------------------
function startPolling() {
    // Immediate first fetch, then every POLL_MS
    fetchTransactions();
    pollInterval = setInterval(fetchTransactions, POLL_MS);
}

async function fetchTransactions() {
    if (!authToken) return;
    
    try {
        const resp = await fetch('/flagged-transactions', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        if (resp.status === 401) {
            // Token expired — get a new one
            await initAuth();
            return;
        }
        
        const data = await resp.json();
        allTransactions = data.transactions || [];
        updateStatus(true);
        renderAll();
    } catch (err) {
        console.error('Fetch failed:', err);
        updateStatus(false);
    }
}

// ---------------------------------------------------------------------------
// RENDERING
// ---------------------------------------------------------------------------

/**
 * Re-render everything: stats cards, filtered table, and counts.
 * Called after every poll and after filter changes.
 */
function renderAll() {
    renderStats();
    renderTable();
}

function renderStats() {
    const total = allTransactions.length;
    const allow = allTransactions.filter(t => t.tier === 'auto_allow').length;
    const review = allTransactions.filter(t => t.tier === 'human_review').length;
    const block = allTransactions.filter(t => t.tier === 'auto_block').length;
    
    document.getElementById('statTotal').textContent = total;
    document.getElementById('statAllow').textContent = allow;
    document.getElementById('statReview').textContent = review;
    document.getElementById('statBlock').textContent = block;
    
    // Percentages
    const pct = (n) => total > 0 ? ((n / total) * 100).toFixed(1) + '% of total' : '0% of total';
    document.getElementById('statAllowPct').textContent = pct(allow);
    document.getElementById('statReviewPct').textContent = pct(review);
    document.getElementById('statBlockPct').textContent = pct(block);
}

function renderTable() {
    const tbody = document.getElementById('txnTableBody');
    const emptyState = document.getElementById('emptyState');
    const tableCount = document.getElementById('tableCount');
    
    // Apply filter
    const filtered = currentFilter === 'all'
        ? allTransactions
        : allTransactions.filter(t => t.tier === currentFilter);
    
    tableCount.textContent = `${filtered.length} transaction${filtered.length !== 1 ? 's' : ''}`;
    
    if (filtered.length === 0) {
        tbody.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }
    
    emptyState.style.display = 'none';
    
    // Build table rows
    // WHY innerHTML instead of DOM API?
    // For a list of <100 items, innerHTML is simpler and fast enough.
    // React-style virtual DOM diffing would be overkill here.
    tbody.innerHTML = filtered.map(txn => {
        const tierClass = `tier-${txn.tier}`;
        const tierLabel = txn.tier_label || txn.tier.replace('_', ' ');
        
        // Format amount with Indian locale (e.g., 1,00,000)
        const amountFormatted = new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(txn.amount_inr);
        
        // Risk score with color gradient: green (0) -> yellow (0.5) -> red (1)
        const scoreColor = getScoreColor(txn.risk_score);
        
        // Show first reason as the primary one
        const topReason = txn.top_reasons && txn.top_reasons.length > 0
            ? txn.top_reasons[0]
            : 'N/A';
        
        return `
            <tr>
                <td><span class="txn-id">${escapeHtml(txn.transaction_id)}</span></td>
                <td><span class="merchant-id">${escapeHtml(txn.merchant_id)}</span></td>
                <td>${escapeHtml(txn.payment_method.toUpperCase())}</td>
                <td><span class="amount">${amountFormatted}</span></td>
                <td><span class="score" style="color:${scoreColor}">${txn.risk_score.toFixed(4)}</span></td>
                <td><span class="tier-badge ${tierClass}"><span class="dot"></span>${tierLabel}</span></td>
                <td><span class="reason-pill">${escapeHtml(topReason)}</span></td>
            </tr>
        `;
    }).join('');
}

// ---------------------------------------------------------------------------
// SIMULATION
// ---------------------------------------------------------------------------

/**
 * Generate and score a random transaction via POST /score-transaction.
 *
 * WHY not just add to the local table?
 * The transaction must go through the full pipeline (preprocess -> predict
 * -> SHAP -> tier) on the backend. Generating locally would skip the model
 * and produce fake results. The simulate button calls the real API endpoint.
 */
async function simulateTransaction() {
    if (!authToken) {
        await initAuth();
        if (!authToken) return;
    }
    
    const btn = document.getElementById('simulateBtn');
    btn.disabled = true;
    btn.textContent = 'Scoring...';
    
    // Generate random transaction data
    const txn = generateRandomTransaction();
    
    try {
        const resp = await fetch('/score-transaction', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(txn),
        });
        
        if (!resp.ok) {
            const err = await resp.json();
            console.error('Scoring failed:', err);
        }
        
        // Immediately fetch to show the new transaction
        await fetchTransactions();
    } catch (err) {
        console.error('Simulation failed:', err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Simulate Transaction';
    }
}

/**
 * Generate a random transaction with realistic distributions.
 * 
 * Occasionally generates "suspicious" transactions (high velocity,
 * late-night UPI, etc.) to demonstrate the fraud detection patterns.
 * This makes the demo more interesting than all-green auto_allow rows.
 */
function generateRandomTransaction() {
    const methods = ['upi', 'card', 'netbanking', 'wallet'];
    const isSuspicious = Math.random() < 0.35; // 35% chance of suspicious txn
    
    if (isSuspicious) {
        // Generate a transaction that's likely to trigger fraud patterns
        const pattern = Math.floor(Math.random() * 4);
        
        switch (pattern) {
            case 0: // Pattern 1: Late-night high-value UPI + new merchant
                return {
                    merchant_id: `merch_${String(Math.floor(Math.random() * 200)).padStart(4, '0')}`,
                    payment_method: 'upi',
                    amount_inr: 5000 + Math.random() * 45000,
                    hour_of_day: Math.floor(Math.random() * 5), // 0-4am
                    merchant_txn_count_7d: Math.floor(Math.random() * 30) + 5,
                    merchant_avg_amount_7d: 500 + Math.random() * 2000,
                    device_risk_score: 0.3 + Math.random() * 0.7,
                    is_new_merchant: 1,
                };
            case 1: // Pattern 2: Velocity spike
                return {
                    merchant_id: `merch_${String(Math.floor(Math.random() * 200)).padStart(4, '0')}`,
                    payment_method: methods[Math.floor(Math.random() * 4)],
                    amount_inr: 100 + Math.random() * 10000,
                    hour_of_day: Math.floor(Math.random() * 24),
                    merchant_txn_count_7d: 150 + Math.floor(Math.random() * 200), // Very high
                    merchant_avg_amount_7d: 500 + Math.random() * 3000,
                    device_risk_score: Math.random() * 0.5,
                    is_new_merchant: Math.random() < 0.3 ? 1 : 0,
                };
            case 2: // Pattern 3: Amount outlier
                return {
                    merchant_id: `merch_${String(Math.floor(Math.random() * 200)).padStart(4, '0')}`,
                    payment_method: methods[Math.floor(Math.random() * 4)],
                    amount_inr: 50000 + Math.random() * 200000, // Way above average
                    hour_of_day: Math.floor(Math.random() * 24),
                    merchant_txn_count_7d: Math.floor(Math.random() * 50) + 5,
                    merchant_avg_amount_7d: 500 + Math.random() * 1500, // Low average
                    device_risk_score: Math.random() * 0.6,
                    is_new_merchant: Math.random() < 0.4 ? 1 : 0,
                };
            case 3: // Pattern 4: High device risk + new merchant
                return {
                    merchant_id: `merch_${String(Math.floor(Math.random() * 200)).padStart(4, '0')}`,
                    payment_method: methods[Math.floor(Math.random() * 4)],
                    amount_inr: 500 + Math.random() * 15000,
                    hour_of_day: Math.floor(Math.random() * 24),
                    merchant_txn_count_7d: Math.floor(Math.random() * 40) + 5,
                    merchant_avg_amount_7d: 500 + Math.random() * 3000,
                    device_risk_score: 0.82 + Math.random() * 0.18, // High risk
                    is_new_merchant: 1,
                };
        }
    }
    
    // Normal (likely legitimate) transaction
    return {
        merchant_id: `merch_${String(Math.floor(Math.random() * 200)).padStart(4, '0')}`,
        payment_method: methods[Math.floor(Math.random() * 4)],
        amount_inr: Math.round(100 + Math.random() * 5000),
        hour_of_day: 8 + Math.floor(Math.random() * 12), // Business hours
        merchant_txn_count_7d: Math.floor(Math.random() * 40) + 5,
        merchant_avg_amount_7d: Math.round(300 + Math.random() * 3000),
        device_risk_score: parseFloat((Math.random() * 0.3).toFixed(2)),
        is_new_merchant: Math.random() < 0.15 ? 1 : 0,
    };
}

// ---------------------------------------------------------------------------
// FILTER
// ---------------------------------------------------------------------------
function setFilter(filter) {
    currentFilter = filter;
    
    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });
    
    renderTable();
}

// ---------------------------------------------------------------------------
// UTILITIES
// ---------------------------------------------------------------------------

/**
 * Map risk score (0-1) to a color: green -> yellow -> red.
 * Uses HSL color space for a smooth gradient.
 */
function getScoreColor(score) {
    // Hue: 120 (green) at score=0, 0 (red) at score=1
    const hue = (1 - score) * 120;
    return `hsl(${hue}, 80%, 55%)`;
}

/**
 * Update the connection status indicator.
 */
function updateStatus(connected) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    
    if (connected) {
        dot.classList.remove('disconnected');
        text.textContent = `Live - polling every ${POLL_MS / 1000}s`;
    } else {
        dot.classList.add('disconnected');
        text.textContent = 'Disconnected';
    }
}

/**
 * Escape HTML to prevent XSS.
 * 
 * WHY manual escaping (not a library)?
 * We're inserting user-controlled data (merchant_id, transaction_id) into
 * innerHTML. Without escaping, a crafted merchant_id like
 * "<script>alert('xss')</script>" would execute. This 4-line function
 * handles the standard HTML entities.
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
