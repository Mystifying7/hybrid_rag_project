// static/app.js

// State management
let currentStats = null;
let isSearching = false;
let isSpeaking = false;
let currentView = 'intro'; // 'intro' or 'search'
let searchHistory = JSON.parse(localStorage.getItem('rag_search_history') || '[]');

// DOM Elements
const introView = document.getElementById('introView');
const searchView = document.getElementById('searchView');
const tabIntro = document.getElementById('tabIntro');
const tabSearch = document.getElementById('tabSearch');

const queryInput = document.getElementById('queryInput');
const btnSearch = document.getElementById('btnSearch');
const btnClearQuery = document.getElementById('btnClearQuery');
const resultsArea = document.getElementById('resultsArea');
const emptyStateArea = document.getElementById('emptyStateArea');
const loadingSkeleton = document.getElementById('loadingSkeleton');
const pipelineFlowCard = document.getElementById('pipelineFlowCard');
const answerContent = document.getElementById('answerContent');
const sourcesContainer = document.getElementById('sourcesContainer');
const contextContent = document.getElementById('contextContent');
const initialKSlider = document.getElementById('initialKSlider');
const finalKSlider = document.getElementById('finalKSlider');
const initialKVal = document.getElementById('initialKVal');
const finalKVal = document.getElementById('finalKVal');
const totalLatencyTag = document.getElementById('totalLatencyTag');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const btnReindex = document.getElementById('btnReindex');
const docsModal = document.getElementById('docsModal');
const toastContainer = document.getElementById('toastContainer');
const historyChipsContainer = document.getElementById('historyChipsContainer');
const quickChipsContainer = document.getElementById('quickChipsContainer');

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    setupEventListeners();
    renderSearchHistory();
    setupMarkdownConfig();
});

// Switch Between Intro Pitch Showcase and Live Search Engine
function switchView(viewName) {
    currentView = viewName;
    if (viewName === 'intro') {
        introView.style.display = 'block';
        searchView.style.display = 'none';
        tabIntro.classList.add('active');
        tabSearch.classList.remove('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
        introView.style.display = 'none';
        searchView.style.display = 'block';
        tabIntro.classList.remove('active');
        tabSearch.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
        setTimeout(() => {
            if (queryInput) queryInput.focus();
        }, 100);
    }
}

// Smooth scroll to a section inside the intro page
function scrollToSection(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Configure Marked.js for safe and highlighted code rendering
function setupMarkdownConfig() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            gfm: true,
            breaks: true,
            highlight: function (code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return typeof hljs !== 'undefined' ? hljs.highlightAuto(code).value : code;
            }
        });
    }
}

// Fetch System Stats & Index Info
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('Failed to load system stats');
        currentStats = await res.json();
        
        if (currentStats.status === 'ready') {
            statusBadge.style.display = 'inline-flex';
            statusText.textContent = `Ready • ${currentStats.chunks_count} Chunks • ${currentStats.docs_count} Doc(s)`;
        }
        
        // Render sample query chips if available
        if (currentStats.sample_queries && quickChipsContainer) {
            renderSampleChips(currentStats.sample_queries);
        }
    } catch (err) {
        console.error('Stats load error:', err);
        statusText.textContent = 'Pipeline Offline';
    }
}

// Render Sample Chips
function renderSampleChips(queries) {
    quickChipsContainer.innerHTML = '';
    queries.forEach((q) => {
        const isGuardrail = q.toLowerCase().includes('redis') || q.toLowerCase().includes('out of scope');
        const chip = document.createElement('div');
        chip.className = `query-chip ${isGuardrail ? 'guardrail-chip' : ''}`;
        chip.innerHTML = `${isGuardrail ? '🛡️' : '⚡'} ${escapeHtml(q)}`;
        chip.onclick = () => {
            switchView('search');
            queryInput.value = q;
            updateInputState();
            executeSearch();
        };
        quickChipsContainer.appendChild(chip);
    });
}

// Setup Event Listeners
function setupEventListeners() {
    // Search input enter key
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            executeSearch();
        }
    });

    queryInput.addEventListener('input', updateInputState);

    btnClearQuery.addEventListener('click', () => {
        queryInput.value = '';
        updateInputState();
        queryInput.focus();
    });

    // Keyboard shortcut '/' or 'Ctrl+K'
    document.addEventListener('keydown', (e) => {
        if ((e.key === '/' && document.activeElement !== queryInput) || ((e.ctrlKey || e.metaKey) && e.key === 'k')) {
            e.preventDefault();
            switchView('search');
            queryInput.focus();
            queryInput.select();
        }
    });

    // Hyperparameter sliders
    initialKSlider.addEventListener('input', (e) => {
        initialKVal.textContent = e.target.value;
    });

    finalKSlider.addEventListener('input', (e) => {
        finalKVal.textContent = e.target.value;
    });

    // Reindex button
    btnReindex.addEventListener('click', handleReindex);
}

function updateInputState() {
    if (queryInput.value.trim().length > 0) {
        btnClearQuery.style.display = 'flex';
    } else {
        btnClearQuery.style.display = 'none';
    }
}

// Toggle Parameter Drawer
function toggleParamsDrawer() {
    const panel = document.getElementById('paramsPanel');
    const icon = document.getElementById('paramsChevronIcon');
    if (panel.style.display === 'grid') {
        panel.style.display = 'none';
        icon.style.transform = 'rotate(0deg)';
    } else {
        panel.style.display = 'grid';
        icon.style.transform = 'rotate(180deg)';
    }
}

// Search Execution
async function executeSearch() {
    const query = queryInput.value.trim();
    if (!query || isSearching) return;

    isSearching = true;
    btnSearch.disabled = true;
    btnSearch.innerHTML = `<span class="pulse-dot" style="background:#fff; box-shadow:none;"></span> Synthesizing...`;

    // Save to history
    addToHistory(query);

    // Show loading state
    emptyStateArea.style.display = 'none';
    resultsArea.style.display = 'none';
    loadingSkeleton.style.display = 'flex';
    pipelineFlowCard.style.display = 'block';

    // Animate Pipeline Stages
    animatePipelineStart();

    const initialK = parseInt(initialKSlider.value, 10);
    const finalK = parseInt(finalKSlider.value, 10);

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                initial_k: initialK,
                final_k: finalK
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Search pipeline failed');

        // Complete Pipeline Stepper Animation
        animatePipelineComplete(data.metrics);

        // Render Results
        renderResults(data);

    } catch (err) {
        showToast(`❌ Error: ${err.message}`, 4000);
        loadingSkeleton.style.display = 'none';
        emptyStateArea.style.display = 'block';
    } finally {
        isSearching = false;
        btnSearch.disabled = false;
        btnSearch.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Synthesize`;
    }
}

// Pipeline Stage Animation Helpers
function animatePipelineStart() {
    const steps = ['step-bm25', 'step-faiss', 'step-rrf', 'step-rerank', 'step-llm'];
    steps.forEach((id, idx) => {
        const el = document.getElementById(id);
        if (el) {
            el.className = 'pipeline-step';
            if (idx === 0) el.classList.add('active');
        }
    });
    totalLatencyTag.textContent = 'Processing...';
}

function animatePipelineComplete(metrics) {
    const steps = ['step-bm25', 'step-faiss', 'step-rrf', 'step-rerank', 'step-llm'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = 'pipeline-step completed';
    });

    if (metrics) {
        totalLatencyTag.textContent = `⚡ Total: ${metrics.total_latency_ms}ms (Retrieval: ${metrics.retrieval_latency_ms}ms | LLM: ${metrics.generation_latency_ms}ms)`;
    }
}

// Render Results
function renderResults(data) {
    loadingSkeleton.style.display = 'none';
    resultsArea.style.display = 'grid';

    // 1. Render Markdown Answer
    let rawAnswer = data.answer || '';
    let renderedHtml = typeof marked !== 'undefined' ? marked.parse(rawAnswer) : rawAnswer.replace(/\n/g, '<br>');
    answerContent.innerHTML = renderedHtml;

    // Wrap pre blocks for copy button integration
    document.querySelectorAll('#answerContent pre').forEach(pre => {
        if (!pre.parentElement.classList.contains('code-block-wrapper')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            
            const header = document.createElement('div');
            header.className = 'code-block-header';
            header.innerHTML = `<span>CODE SNIPPET</span><button class="btn-copy-code" onclick="copySnippet(this)">📋 Copy Code</button>`;
            
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(header);
            wrapper.appendChild(pre);
        }
    });

    // 2. Render Cited Sources
    sourcesContainer.innerHTML = '';
    if (data.sources && data.sources.length > 0) {
        let citationsHtml = `
            <div class="citations-wrapper">
                <div class="citations-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
                    Verified Reference Sources (${data.sources.length})
                </div>
                <div class="citation-pills-list">
        `;
        data.sources.forEach(src => {
            citationsHtml += `
                <div class="citation-card">
                    <div class="citation-meta">
                        <span class="citation-icon">📄</span>
                        <div>
                            <div class="citation-filename">${escapeHtml(src)}</div>
                        </div>
                    </div>
                    <span class="panel-badge badge-grounded">Grounding Verified</span>
                </div>
            `;
        });
        citationsHtml += `</div></div>`;
        sourcesContainer.innerHTML = citationsHtml;
    }

    // 3. Render Context Inspector Cards
    renderContextCards(data.retrieved_context);
}

// Render Context Inspector Cards
function renderContextCards(chunks) {
    contextContent.innerHTML = '';
    if (!chunks || chunks.length === 0) {
        contextContent.innerHTML = `<p style="color: var(--text-muted);">No candidate context chunks retrieved.</p>`;
        return;
    }

    let html = '<div class="context-cards-list">';
    chunks.forEach((ctx, index) => {
        const isTop = index === 0;
        const normalizedPct = Math.min(Math.max((ctx.cross_encoder_score + 10) / 20 * 100, 5), 100);
        
        html += `
            <div class="chunk-inspect-card ${isTop ? 'rank-top' : ''}">
                <div class="chunk-card-header">
                    <span class="rank-badge-pill">${isTop ? '🏆' : '🎯'} Rank #${ctx.rank || index + 1}</span>
                    <div class="score-metrics-group">
                        <span class="score-tag highlight-score">Logit: ${ctx.cross_encoder_score}</span>
                        <span class="score-tag">RRF: ${ctx.rrf_score}</span>
                    </div>
                </div>

                <div class="score-gauge-container">
                    <div class="gauge-label-row">
                        <span>Cross-Encoder Relevance</span>
                        <span>${normalizedPct.toFixed(0)}% Confidence</span>
                    </div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-fill" style="width: ${normalizedPct}%;"></div>
                    </div>
                </div>

                <div class="chunk-source-breadcrumb">
                    <span>📁 <strong>${escapeHtml(ctx.source)}</strong></span>
                    <span>›</span>
                    <span>📑 ${escapeHtml(ctx.section)}</span>
                </div>

                <div class="chunk-content-preview">${escapeHtml(ctx.content)}</div>
            </div>
        `;
    });
    html += '</div>';
    contextContent.innerHTML = html;
}

// Re-index Trigger
async function handleReindex() {
    btnReindex.disabled = true;
    btnReindex.innerHTML = `🔄 Reindexing...`;
    
    try {
        const res = await fetch('/api/reindex', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Reindexing failed');
        
        showToast(`✅ ${data.message} (${data.duration_ms}ms)`, 3500);
        await fetchStats();
    } catch (err) {
        showToast(`❌ Reindex Error: ${err.message}`, 4000);
    } finally {
        btnReindex.disabled = false;
        btnReindex.innerHTML = `🔄 Re-index`;
    }
}

// Copy Answer to Clipboard
function copyAnswer() {
    const rawAnswer = answerContent.innerText;
    if (!rawAnswer) return;
    navigator.clipboard.writeText(rawAnswer).then(() => {
        showToast('📋 Answer copied to clipboard!', 2000);
    }).catch(() => {
        showToast('Failed to copy', 2000);
    });
}

// Copy Code Snippet
function copySnippet(btn) {
    const pre = btn.closest('.code-block-wrapper').querySelector('pre');
    if (pre) {
        navigator.clipboard.writeText(pre.innerText).then(() => {
            const orig = btn.innerHTML;
            btn.innerHTML = `✅ Copied!`;
            setTimeout(() => { btn.innerHTML = orig; }, 1800);
        });
    }
}

// Text-to-Speech (Read Aloud)
function toggleSpeech() {
    if (!('speechSynthesis' in window)) {
        showToast('Speech synthesis not supported on this browser', 2500);
        return;
    }

    if (isSpeaking) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
        document.getElementById('btnTtsIcon').textContent = '🔊';
        return;
    }

    const textToSpeak = answerContent.innerText;
    if (!textToSpeak) return;

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    utterance.onend = () => {
        isSpeaking = false;
        document.getElementById('btnTtsIcon').textContent = '🔊';
    };

    utterance.onerror = () => {
        isSpeaking = false;
        document.getElementById('btnTtsIcon').textContent = '🔊';
    };

    window.speechSynthesis.speak(utterance);
    isSpeaking = true;
    document.getElementById('btnTtsIcon').textContent = '⏹️';
    showToast('🔊 Reading answer aloud...', 2000);
}

// Search History
function addToHistory(query) {
    searchHistory = searchHistory.filter(q => q.toLowerCase() !== query.toLowerCase());
    searchHistory.unshift(query);
    if (searchHistory.length > 5) searchHistory.pop();
    localStorage.setItem('rag_search_history', JSON.stringify(searchHistory));
    renderSearchHistory();
}

function renderSearchHistory() {
    if (!historyChipsContainer) return;
    historyChipsContainer.innerHTML = '';
    if (searchHistory.length === 0) {
        historyChipsContainer.parentElement.style.display = 'none';
        return;
    }
    historyChipsContainer.parentElement.style.display = 'block';
    searchHistory.forEach(q => {
        const chip = document.createElement('div');
        chip.className = 'query-chip';
        chip.innerHTML = `🕒 ${escapeHtml(q)}`;
        chip.onclick = () => {
            switchView('search');
            queryInput.value = q;
            updateInputState();
            executeSearch();
        };
        historyChipsContainer.appendChild(chip);
    });
}

function clearSearchHistory() {
    searchHistory = [];
    localStorage.removeItem('rag_search_history');
    renderSearchHistory();
    showToast('History cleared', 1500);
}

// Modals Management
function openDocsModal() {
    if (currentStats && currentStats.documents) {
        const listEl = document.getElementById('indexedDocsList');
        let docsHtml = '';
        currentStats.documents.forEach(doc => {
            docsHtml += `
                <div style="background: rgba(15,23,42,0.6); border: 1px solid var(--border-subtle); padding: 14px; border-radius: var(--radius-md); margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #38bdf8; font-size: 14px;">📄 ${escapeHtml(doc.filename)}</strong>
                        <span class="score-tag">${doc.chunks_count} Chunks</span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 12px; margin-top: 4px;">
                        Path: <code>${escapeHtml(doc.path)}</code> • Size: ${(doc.size_bytes / 1024).toFixed(1)} KB
                    </div>
                </div>
            `;
        });
        listEl.innerHTML = docsHtml;
    }
    docsModal.style.display = 'flex';
}

function closeDocsModal() {
    docsModal.style.display = 'none';
}

// Toast Helper
function showToast(msg, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = msg;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Utility: Escape HTML
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function (m) {
        return ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[m];
    });
}
