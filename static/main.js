/**
 * main.js - RiskLens Fraud Dashboard Logic
 * Explanations are provided inline for clarity.
 */

// State
let transactions = [];
let jwtToken = null;
let currentFilter = 'all';
let selectedTxnId = null;

// DOM Elements
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const txnTableBody = document.getElementById('txnTableBody');
const modalOverlay = document.getElementById('modalOverlay');
const modalPanel = document.getElementById('modalPanel');
const modalClose = document.getElementById('modalClose');

// Init
async function init() {
    await fetchToken();
    if (jwtToken) {
        setConnected(true);
        pollTransactions();
        simulateThresholds(); // Initial call
        
        // Polling loop (every 3 seconds)
        setInterval(pollTransactions, 3000);
    } else {
        setConnected(false);
    }
}

// XSS mitigation
function escapeHtml(unsafe) {
    if (unsafe == null) return '';
    return unsafe
         .toString()
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function setConnected(isLive) {
    if (isLive) {
        statusBadge.className = 'status-badge live';
        statusText.textContent = 'Live';
    } else {
        statusBadge.className = 'status-badge disconnected';
        statusText.textContent = 'Disconnected';
    }
}

// Fetch JWT
async function fetchToken() {
    try {
        const res = await fetch('/api/generate-token', { method: 'POST' });
        const data = await res.json();
        jwtToken = data.token;
    } catch (err) {
        console.error("Token fetch failed", err);
    }
}

// Poll Transactions
async function pollTransactions() {
    if (!jwtToken) return;
    try {
        const res = await fetch('/flagged-transactions', {
            headers: { 'Authorization': `Bearer ${jwtToken}` }
        });
        if (res.ok) {
            const data = await res.json();
            transactions = data.transactions || [];
            updateStats();
            renderTable();
            renderHistogram();
            setConnected(true);
        } else if (res.status === 401) {
            await fetchToken(); // Re-fetch on expiry
        }
    } catch (err) {
        console.error("Polling failed", err);
        setConnected(false);
    }
}

// Update Stats Cards
function updateStats() {
    const total = transactions.length;
    let allow = 0, review = 0, block = 0;
    
    transactions.forEach(t => {
        if (t.tier === 'auto_allow') allow++;
        else if (t.tier === 'human_review') review++;
        else if (t.tier === 'auto_block') block++;
    });

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statAllow').textContent = allow;
    document.getElementById('statReview').textContent = review;
    document.getElementById('statBlock').textContent = block;

    const getPct = (val) => total > 0 ? ((val / total) * 100).toFixed(1) + '%' : '0%';
    document.getElementById('statAllowPct').textContent = getPct(allow);
    document.getElementById('statReviewPct').textContent = getPct(review);
    document.getElementById('statBlockPct').textContent = getPct(block);
}

// Map score to color (green -> red)
function getScoreColor(score) {
    const hue = (1 - score) * 120; // 0 = red, 120 = green. If score=1, hue=0 (red).
    return `hsl(${hue}, 80%, 50%)`;
}

// Render Table
function renderTable() {
    txnTableBody.innerHTML = '';
    
    const filtered = currentFilter === 'all' 
        ? transactions 
        : transactions.filter(t => t.tier === currentFilter);
        
    // Sort descending by time (assuming ID roughly correlates or just list order)
    filtered.forEach(txn => {
        const tr = document.createElement('tr');
        tr.onclick = () => openModal(txn);
        
        const date = new Date(txn.scored_at).toLocaleTimeString();
        const scoreColor = getScoreColor(txn.risk_score);
        
        // Reasons
        const reasonsHtml = (txn.top_reasons || []).map(r => 
            `<span class="pill">${escapeHtml(r)}</span>`
        ).join('');
        
        tr.innerHTML = `
            <td>${escapeHtml(date)}</td>
            <td class="mono">${escapeHtml(txn.transaction_id)}</td>
            <td class="mono">₹${txn.amount_inr.toLocaleString()}</td>
            <td class="mono" style="color: ${scoreColor}">${txn.risk_score.toFixed(3)}</td>
            <td>
                <span class="tier-badge" style="background-color: ${escapeHtml(txn.tier_color)}20; color: ${escapeHtml(txn.tier_color)}; border: 1px solid ${escapeHtml(txn.tier_color)}40">
                    ${escapeHtml(txn.tier_label)}
                </span>
            </td>
            <td>${reasonsHtml}</td>
        `;
        txnTableBody.appendChild(tr);
    });
}

// Detail Modal / SHAP Waterfall
function openModal(txn) {
    selectedTxnId = txn.transaction_id;
    document.getElementById('modalTitle').textContent = `Txn: ${txn.transaction_id}`;
    
    const scoreEl = document.getElementById('modalScore');
    scoreEl.textContent = txn.risk_score.toFixed(3);
    scoreEl.style.color = getScoreColor(txn.risk_score);
    
    document.getElementById('modalTier').innerHTML = `
        <span class="tier-badge" style="background-color: ${escapeHtml(txn.tier_color)}20; color: ${escapeHtml(txn.tier_color)}; border: 1px solid ${escapeHtml(txn.tier_color)}40; font-size: 1rem;">
            ${escapeHtml(txn.tier_label)}
        </span>
    `;
    
    renderWaterfall(txn);
    renderRawFeatures(txn);
    
    modalOverlay.classList.add('active');
    modalPanel.classList.add('active');
    renderHistogram(); // Update histogram to show marker
}

function closeModal() {
    modalOverlay.classList.remove('active');
    modalPanel.classList.remove('active');
    selectedTxnId = null;
    renderHistogram(); // Remove marker
}

modalClose.onclick = closeModal;
modalOverlay.onclick = closeModal;

// Pure CSS Waterfall Chart
function renderWaterfall(txn) {
    const container = document.getElementById('waterfallContainer');
    if (!txn.shap_details || txn.shap_details.length === 0) {
        container.innerHTML = '<p>No SHAP details available.</p>';
        return;
    }
    
    // Sort by absolute shap value
    const details = [...txn.shap_details].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    
    // Find max absolute value to scale bars
    const maxVal = Math.max(...details.map(d => Math.abs(d.shap_value)));
    
    let html = '';
    
    // 50% is center line.
    details.forEach(d => {
        const isPositive = d.shap_value > 0;
        const widthPct = maxVal > 0 ? (Math.abs(d.shap_value) / maxVal) * 50 : 0;
        
        let barStyle = `width: ${widthPct}%;`;
        if (isPositive) {
            barStyle += `left: 50%; background-color: rgba(239, 68, 68, 0.8);`; // red increases risk
        } else {
            barStyle += `right: 50%; background-color: rgba(59, 130, 246, 0.8);`; // blue decreases risk
        }
        
        html += `
            <div class="waterfall-row" title="${escapeHtml(d.reason)}">
                <div class="wf-label">${escapeHtml(d.feature)}</div>
                <div class="wf-chart-area">
                    <div class="wf-bar ${isPositive ? 'positive' : 'negative'}" style="${barStyle}">
                        ${isPositive ? '+' : ''}${d.shap_value.toFixed(2)}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `
        <div class="prediction-summary">
            <div>Base: ${txn.base_value.toFixed(2)}</div>
            <div>Score: ${txn.risk_score.toFixed(3)}</div>
        </div>
    `;
    
    container.innerHTML = html;
}

function renderRawFeatures(txn) {
    const tbody = document.getElementById('rawFeaturesTable');
    let html = '';
    
    const exclude = ['transaction_id', 'tier', 'tier_label', 'tier_color', 'top_reasons', 'shap_details'];
    
    for (const [key, value] of Object.entries(txn)) {
        if (!exclude.includes(key)) {
            html += `
                <tr>
                    <td style="color: var(--text-secondary); padding: 0.25rem 0;">${escapeHtml(key)}</td>
                    <td class="mono" style="text-align: right; padding: 0.25rem 0;">${escapeHtml(value)}</td>
                </tr>
            `;
        }
    }
    tbody.innerHTML = html;
}

// Histogram (Inline SVG)
function renderHistogram() {
    const container = document.getElementById('histogram-container');
    if (!transactions || transactions.length === 0) {
        container.innerHTML = '<div style="display:flex; height:100%; align-items:center; justify-content:center; color: var(--text-secondary)">No data</div>';
        return;
    }
    
    const bins = 20;
    const counts = new Array(bins).fill(0);
    let maxCount = 0;
    
    transactions.forEach(t => {
        let bin = Math.floor(t.risk_score * bins);
        if (bin >= bins) bin = bins - 1; // edge case for score=1.0
        counts[bin]++;
        if (counts[bin] > maxCount) maxCount = counts[bin];
    });
    
    const allowThreshold = parseFloat(document.getElementById('allowSlider').value);
    const blockThreshold = parseFloat(document.getElementById('blockSlider').value);
    
    let svg = `<svg viewBox="0 0 100 100" preserveAspectRatio="none">`;
    
    // Draw Bars
    counts.forEach((count, i) => {
        if (count === 0) return;
        const x = (i / bins) * 100;
        const width = 100 / bins - 1; // -1 for gap
        const height = (count / maxCount) * 90; // leave 10 for margin
        const y = 100 - height;
        
        const binMid = (i + 0.5) / bins;
        let color = 'var(--color-review)';
        if (binMid < allowThreshold) color = 'var(--color-allow)';
        else if (binMid > blockThreshold) color = 'var(--color-block)';
        
        svg += `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${color}" opacity="0.8" rx="1" />`;
    });
    
    // Selected Transaction Marker
    if (selectedTxnId) {
        const txn = transactions.find(t => t.transaction_id === selectedTxnId);
        if (txn) {
            const x = txn.risk_score * 100;
            svg += `
                <line x1="${x}" y1="0" x2="${x}" y2="100" stroke="#fff" stroke-width="0.5" stroke-dasharray="2" />
                <circle cx="${x}" cy="5" r="2" fill="#fff" />
            `;
        }
    }
    
    // Threshold Lines
    const ax = allowThreshold * 100;
    const bx = blockThreshold * 100;
    
    svg += `
        <line x1="${ax}" y1="0" x2="${ax}" y2="100" stroke="var(--color-allow)" stroke-width="0.5" stroke-dasharray="2" />
        <line x1="${bx}" y1="0" x2="${bx}" y2="100" stroke="var(--color-block)" stroke-width="0.5" stroke-dasharray="2" />
    `;
    
    svg += `</svg>`;
    container.innerHTML = svg;
}

// Threshold Tuning
const allowSlider = document.getElementById('allowSlider');
const blockSlider = document.getElementById('blockSlider');
const allowVal = document.getElementById('allowCeilVal');
const blockVal = document.getElementById('blockFloorVal');

function handleSliderChange() {
    let a = parseFloat(allowSlider.value);
    let b = parseFloat(blockSlider.value);
    
    // Enforce logic: allow cannot be >= block
    if (a >= b) {
        if (this === allowSlider) {
            a = b - 0.01;
            allowSlider.value = a;
        } else {
            b = a + 0.01;
            blockSlider.value = b;
        }
    }
    
    allowVal.textContent = a.toFixed(2);
    blockVal.textContent = b.toFixed(2);
    
    renderHistogram(); // Instant update of colors
    debouncedSimulate(); // API call
}

allowSlider.addEventListener('input', handleSliderChange);
blockSlider.addEventListener('input', handleSliderChange);

// Debounce helper
let simulateTimer;
function debouncedSimulate() {
    clearTimeout(simulateTimer);
    simulateTimer = setTimeout(simulateThresholds, 300); // 300ms debounce
}

// Call API to simulate
async function simulateThresholds() {
    if (!jwtToken) return;
    try {
        const res = await fetch('/api/simulate-thresholds', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${jwtToken}`
            },
            body: JSON.stringify({
                allow_ceiling: parseFloat(allowSlider.value),
                block_floor: parseFloat(blockSlider.value)
            })
        });
        
        if (res.ok) {
            const data = await res.json();
            renderSimulation(data);
        }
    } catch (err) {
        console.error("Simulation failed", err);
    }
}

function renderSimulation(data) {
    const container = document.getElementById('simBarContainer');
    const statsText = document.getElementById('simStatsText');
    
    if (data.total === 0) {
        container.innerHTML = '';
        statsText.innerHTML = 'No data';
        return;
    }
    
    const apct = data.auto_allow.pct;
    const rpct = data.human_review.pct;
    const bpct = data.auto_block.pct;
    
    container.innerHTML = `
        <div class="sim-bar" style="width: ${apct}%; background-color: var(--color-allow)" title="Allow: ${data.auto_allow.count}">
            ${apct > 10 ? apct.toFixed(0) + '%' : ''}
        </div>
        <div class="sim-bar" style="width: ${rpct}%; background-color: var(--color-review)" title="Review: ${data.human_review.count}">
            ${rpct > 10 ? rpct.toFixed(0) + '%' : ''}
        </div>
        <div class="sim-bar" style="width: ${bpct}%; background-color: var(--color-block)" title="Block: ${data.auto_block.count}">
            ${bpct > 10 ? bpct.toFixed(0) + '%' : ''}
        </div>
    `;
    
    statsText.innerHTML = `
        <span style="color: var(--color-allow)">Allow: ${data.auto_allow.count}</span>
        <span style="color: var(--color-review)">Review: ${data.human_review.count}</span>
        <span style="color: var(--color-block)">Block: ${data.auto_block.count}</span>
    `;
}

// Filter buttons
document.getElementById('tierFilters').addEventListener('click', e => {
    if (e.target.tagName === 'BUTTON') {
        document.querySelectorAll('#tierFilters button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentFilter = e.target.dataset.tier;
        renderTable();
    }
});

// Simulate Button
document.getElementById('btnSimulate').addEventListener('click', async (e) => {
    const isBatch = e.shiftKey;
    const count = isBatch ? 10 : 1;
    
    const btn = e.target;
    btn.disabled = true;
    btn.textContent = 'Simulating...';
    
    try {
        for(let i=0; i<count; i++) {
            // Wait for 100ms between calls
            if(i>0) await new Promise(r => setTimeout(r, 100));
            
            await fetch('/score-transaction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${jwtToken}`
                },
                body: JSON.stringify({}) // Assuming backend generates transaction context
            });
        }
        pollTransactions(); // Refresh immediately after simulation
    } catch(err) {
        console.error("Simulation generation failed", err);
    }
    
    btn.disabled = false;
    btn.textContent = 'Simulate Transaction';
});

// Run
init();
