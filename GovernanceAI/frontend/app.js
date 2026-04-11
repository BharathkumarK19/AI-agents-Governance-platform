/* ── Governance AI — Frontend App Logic ────────────────────────────────────
   A ChatGPT-like interface for the multi-agent governance platform.
   Connects to the FastAPI backend at /api/*
── */

const API = window.location.origin;

// ══════════════════════════════════════════════════════════════════════════════
//  ONBOARDING FLOW  (Login → Model Selection → Chat)
// ══════════════════════════════════════════════════════════════════════════════

const DEMO_EMAIL    = 'demo@governance.ai';
const DEMO_PASSWORD = 'demo123';

// Available governance analysis modes
const AVAILABLE_MODELS = [
  {
    id: 'business-analyzer',
    name: 'Business Analyzer',
    provider: 'Governance AI · Business Intelligence',
    icon: '📈',
    desc: 'Deep analysis of market trends, corporate strategy, financial data, and business governance frameworks.',
    badge: 'Enterprise',
    badgeClass: 'green',
    color: '#22c55e',
  },
  {
    id: 'analyzer',
    name: 'Analyzer',
    provider: 'Governance AI · Multi-Domain Analysis',
    icon: '🔍',
    desc: 'General purpose analyzer for policy, regulation, and cross-disciplinary governance research.',
    badge: 'Standard',
    badgeClass: '',
    color: '#6366f1',
  }
];

let selectedModel = null;

// ── Render model grid ──────────────────────────────────────────────────────
function renderModelGrid() {
  const grid = document.getElementById('modelGrid');
  if (!grid) return;

  grid.innerHTML = AVAILABLE_MODELS.map(m => `
    <button type="button" class="model-option" data-model-id="${m.id}"
            style="--m-clr: ${m.color}"
            title="Select ${m.name}" aria-label="Select ${m.name}">
      <div class="model-option-header">
        <span class="model-icon">${m.icon}</span>
        <div class="model-check">
          <svg viewBox="0 0 10 10" fill="none" stroke="white" stroke-width="2">
            <path d="M1.5 5L4 7.5L8.5 2.5"/>
          </svg>
        </div>
      </div>
      <div class="model-name">${m.name}</div>
      <div class="model-provider">${m.provider}</div>
      <div class="model-desc">${m.desc}</div>
      <span class="model-badge ${m.badgeClass}">${m.badge}</span>
    </button>
  `).join('');

  // Bind selection
  grid.querySelectorAll('.model-option').forEach(btn => {
    btn.addEventListener('click', () => {
      grid.querySelectorAll('.model-option').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      selectedModel = btn.dataset.modelId;
      document.getElementById('btnConfirmModel').disabled = false;
    });
  });
}

// ── Screen transition helpers ──────────────────────────────────────────────
function fadeOutScreen(id, callback) {
  const el = document.getElementById(id);
  if (!el) { callback && callback(); return; }
  el.classList.add('fade-out');
  setTimeout(() => {
    el.classList.add('hidden');
    el.classList.remove('fade-out');
    callback && callback();
  }, 500);
}

function showScreen(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('hidden');
  // Re-trigger card animation
  const card = el.querySelector('.onboard-card');
  if (card) {
    card.style.animation = 'none';
    card.offsetHeight; // reflow
    card.style.animation = '';
  }
}

// ── Step 1 → Step 2: Login ─────────────────────────────────────────────────
function initLogin() {
  const form     = document.getElementById('loginForm');
  const emailEl  = document.getElementById('loginEmail');
  const passEl   = document.getElementById('loginPassword');
  const errorEl  = document.getElementById('loginError');
  const btnLogin = document.getElementById('btnLogin');
  const btnEye   = document.getElementById('btnTogglePass');

  if (!btnLogin || !emailEl || !passEl) {
    console.error('[Login] Critical UI elements missing!');
    return;
  }

  console.log('[Login] Ready.');

  // Password visibility
  btnEye && btnEye.addEventListener('click', () => {
    const isText = passEl.type === 'text';
    passEl.type = isText ? 'password' : 'text';
    btnEye.querySelector('svg').innerHTML = isText
      ? '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>'
      : '<path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19M1 1l22 22"/>';
  });

  const handleAuth = (e) => {
    if (e) e.preventDefault();
    console.log('[Login] Submitting...');
    
    const email = emailEl.value.trim().toLowerCase();
    const pass  = passEl.value;

    errorEl.classList.add('hidden');
    
    const isValid = (email === DEMO_EMAIL && pass === DEMO_PASSWORD) || 
                   (email.includes('@') && email.includes('.') && pass.length >= 4);

    if (!isValid) {
      errorEl.classList.remove('hidden');
      passEl.value = '';
      passEl.focus();
      return;
    }

    sessionStorage.setItem('gov_ai_user', email);
    btnLogin.classList.add('loading');
    btnLogin.disabled = true;

    setTimeout(() => {
      fadeOutScreen('loginScreen', () => {
        renderModelGrid();
        showScreen('modelScreen');
      });
    }, 400);
  };

  if (form) form.addEventListener('submit', handleAuth);
  btnLogin.addEventListener('click', handleAuth);
}

function initModelSelect() {
  const btnConfirm = document.getElementById('btnConfirmModel');
  if (!btnConfirm) return;

  btnConfirm.addEventListener('click', () => {
    if (!selectedModel) return;
    sessionStorage.setItem('gov_ai_model', selectedModel);
    btnConfirm.classList.add('loading');
    btnConfirm.disabled = true;
    setTimeout(() => {
      fadeOutScreen('modelScreen', () => initApp());
    }, 400);
  });
}

// ── Called once onboarding is complete ────────────────────────────────────
async function initApp() {
  bindEvents();

  try {
    await apiFetch('/api/health');
  } catch (e) {
    showToast('⚠️ Backend unreachable. Start server on port 8090.', 'error');
  }

  await Promise.all([loadConversations(), loadAgents()]);

  // Personalize Interface based on chosen model
  personalizeInterface();
}

function personalizeInterface() {
  const modelId = sessionStorage.getItem('gov_ai_model');
  const model = AVAILABLE_MODELS.find(m => m.id === modelId) || AVAILABLE_MODELS[0];
  
  // Update header and icons
  if (els.chatTitle) els.chatTitle.textContent = model.name;
  
  const subtitle = document.querySelector('.chat-subtitle');
  if (subtitle) {
    subtitle.textContent = `${model.provider} Workspace`;
  }

  // Update welcome screen personalization
  const welcomeTitle = document.querySelector('#welcomeScreen h2');
  if (welcomeTitle) {
    welcomeTitle.textContent = `How can your ${model.name} assist you today?`;
  }
}

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  currentView: 'chat',
  currentConvId: null,
  conversations: [],
  agents: [],
  isLoading: false,
  sidebarCollapsed: false,
  traceOpen: false,
  lastTrace: null,
  sessionStats: {
    totalConfidence: 0,
    messageCount: 0,
  },
  profile: {
    name: 'John Doe',
    role: 'Policy Analyst',
    dept: 'Governance',
    email: '',
  }
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const els = {
  sidebar:          $('sidebar'),
  sidebarToggle:    $('sidebarToggle'),
  mobileMenu:       $('mobileMenu'),
  btnNewChat:       $('btnNewChat'),
  convList:         $('conversationList'),
  agentListSidebar: $('agentListSidebar'),
  btnAgents:        $('btnAgents'),
  btnSettings:      $('btnSettings'),

  chatView:         $('chatView'),
  agentsView:       $('agentsView'),
  settingsView:     $('settingsView'),
  profileView:      $('profileView'),
  
  sideUserName:     $('sideUserName'),
  sideUserRole:     $('sideUserRole'),

  chatTitle:        $('chatTitle'),
  welcomeScreen:    $('welcomeScreen'),
  messagesList:     $('messagesList'),
  messagesContainer:$('messagesContainer'),

  tracePanel:       $('tracePanel'),
  traceContent:     $('traceContent'),
  btnToggleTrace:   $('btnToggleTrace'),
  btnCloseTrace:    $('btnCloseTrace'),

  chatInput:        $('chatInput'),
  btnSend:          $('btnSend'),
  btnBackToModels:  $('btnBackToModels'),
  charCount:        $('charCount'),

  agentsGrid:       $('agentsGrid'),

  btnProfile:       $('btnProfile'),
  btnLogout:        $('btnLogout'),
  profileForm:      $('profileForm'),

  loadingOverlay:   $('loadingOverlay'),
  loadingText:      $('loadingText'),

  pipelineSteps: {
    research: $('pipelineResearch'),
    analysis: $('pipelineAnalysis'),
    summary:  $('pipelineSummary'),
    eval:     $('pipelineEval'),
  },

  // Settings
  openrouterKey:    $('openrouterKey'),
  tavilyKey:        $('tavilyKey'),
  btnSaveOpenrouter:$('btnSaveOpenrouter'),
  btnSaveTavily:    $('btnSaveTavily'),
  modelSelect:      $('modelSelect'),
  tempSlider:       $('tempSlider'),
  tempValue:        $('tempValue'),
  btnClearHistory:  $('btnClearHistory'),
  // Sidebar Metrics
  sideStats:        $('sidebarMetrics'),
  sideAccProgress:  $('sideAccuracyProgress'),
  sideAccValue:     $('sideAccuracyValue'),
  sideAccCircle:   $('sideAccuracyCircle'),
  sideMetricTabs:   $('sideMetricTabs'),
};

// ── Markdown renderer (lightweight) ───────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  let html = String(text);

  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${escHtml(code.trim())}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`);

  // Bold & italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

  // HR
  html = html.replace(/^---+$/gm, '<hr>');

  // Blockquote
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered lists
  html = html.replace(/((?:^[-*] .+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^[-*] /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });

  // Ordered lists
  html = html.replace(/((?:^\d+\. .+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  // Paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;

  // Clean up empty paragraphs
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<p>(<(?:ul|ol|pre|h[1-6]|hr|blockquote))/g, '$1');
  html = html.replace(/(<\/(?:ul|ol|pre|h[1-6]|hr|blockquote)>)<\/p>/g, '$1');

  return html;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Time formatting ────────────────────────────────────────────────────────
function formatTime(isoStr) {
  if (!isoStr) return '';
  try {
    return new Date(isoStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function timeAgo(isoStr) {
  if (!isoStr) return '';
  const diff = Date.now() - new Date(isoStr).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// ── API helpers ────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Load conversations ─────────────────────────────────────────────────────
async function loadConversations() {
  try {
    state.conversations = await apiFetch('/api/conversations');
    renderConversationList();
  } catch (e) {
    console.warn('Could not load conversations:', e.message);
  }
}

function renderConversationList() {
  const convs = state.conversations;
  if (!convs || convs.length === 0) {
    els.convList.innerHTML = '<div class="conv-empty">No conversations yet</div>';
    return;
  }
  els.convList.innerHTML = convs.map(c => `
    <div class="conv-item ${c.id === state.currentConvId ? 'active' : ''}" data-id="${c.id}">
      <div class="conv-icon">💬</div>
      <div class="conv-info">
        <div class="conv-title">${escHtml(c.title || 'Conversation')}</div>
        <div class="conv-time">${timeAgo(c.updated_at)}</div>
      </div>
      <button class="conv-delete btn-icon" data-del="${c.id}" title="Delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 6h18M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6M9 6V4h6v2"/>
        </svg>
      </button>
    </div>
  `).join('');

  // Events
  els.convList.querySelectorAll('.conv-item').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('[data-del]')) return;
      loadConversation(el.dataset.id);
    });
  });
  els.convList.querySelectorAll('[data-del]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(btn.dataset.del);
    });
  });
}

async function loadConversation(convId) {
  state.currentConvId = convId;
  resetSidebarMetrics(); // Clear old session stats
  try {
    const data = await apiFetch(`/api/conversations/${convId}`);
    els.welcomeScreen.classList.add('hidden');
    els.messagesList.innerHTML = '';
    
    // Recalculate metrics for all messages in history
    (data.messages || []).forEach(msg => appendMessage(msg));
    
    scrollToBottom();
    renderConversationList();
    // Update title
    const conv = state.conversations.find(c => c.id === convId);
    if (els.chatTitle) els.chatTitle.textContent = conv ? conv.title : 'Conversation';
  } catch (e) {
    showToast('Failed to load conversation', 'error');
  }
}

async function deleteConversation(convId) {
  try {
    await apiFetch(`/api/conversations/${convId}`, { method: 'DELETE' });
    if (state.currentConvId === convId) {
      startNewChat();
    }
    await loadConversations();
  } catch (e) {
    showToast('Failed to delete', 'error');
  }
}

// ── Load agents ────────────────────────────────────────────────────────────
async function loadAgents() {
  try {
    state.agents = await apiFetch('/api/agents');
    renderAgentsSidebar();
    renderAgentsGrid();
  } catch (e) {
    console.warn('Could not load agents:', e.message);
    // Fallback agents
    state.agents = [
      { id: 'research', name: 'Research Agent', icon: '🔍', status: 'active', model: 'openrouter/auto', tasks_completed: 127, accuracy: 0.96, description: 'Gathers real-time data from the web.', color: '#6366f1' },
      { id: 'analysis', name: 'Analysis Agent', icon: '📊', status: 'active', model: 'openrouter/auto', tasks_completed: 119, accuracy: 0.93, description: 'Extracts patterns and insights.', color: '#8b5cf6' },
      { id: 'summary', name: 'Summary Agent', icon: '📝', status: 'active', model: 'openrouter/auto', tasks_completed: 115, accuracy: 0.95, description: 'Creates governance reports.', color: '#a855f7' },
      { id: 'evaluation', name: 'Evaluation Agent', icon: '✅', status: 'active', model: 'openrouter/auto', tasks_completed: 112, accuracy: 0.98, description: 'Detects hallucinations.', color: '#ec4899' },
    ];
    renderAgentsSidebar();
    renderAgentsGrid();
  }
}

function renderAgentsSidebar() {
  els.agentListSidebar.innerHTML = state.agents.map(a => `
    <div class="agent-pill">
      <span class="agent-pill-icon">${a.icon}</span>
      <div class="agent-pill-info">
        <div class="agent-pill-name">${a.name}</div>
        <div class="agent-pill-status"><span class="dot-status"></span> Online</div>
      </div>
    </div>
  `).join('');
}

function renderAgentsGrid() {
  els.agentsGrid.innerHTML = state.agents.map(a => `
    <div class="agent-card" style="--accent-clr: ${a.color || '#6366f1'}">
      <div class="agent-card-header">
        <div class="agent-card-icon">${a.icon}</div>
        <div class="agent-card-info">
          <div class="agent-card-name">${a.name}</div>
          <div class="agent-card-status"><span class="dot-status"></span> Active</div>
          <div class="agent-card-model">${a.model}</div>
        </div>
      </div>
      <div class="agent-card-desc">${a.description}</div>
      <div class="agent-card-stats">
        <div class="agent-stat">
          <div class="agent-stat-value">${a.tasks_completed}</div>
          <div class="agent-stat-label">Tasks Done</div>
        </div>
        <div class="agent-stat">
          <div class="agent-stat-value">${Math.round(a.accuracy * 100)}%</div>
          <div class="agent-stat-label">Accuracy</div>
        </div>
      </div>
    </div>
  `).join('');
}

// ── Chat functions ─────────────────────────────────────────────────────────
function startNewChat() {
  state.currentConvId = null;
  els.chatTitle.textContent = 'Governance AI';
  els.welcomeScreen.classList.remove('hidden');
  els.messagesList.innerHTML = '';
  resetSidebarMetrics();
  renderConversationList();
  showView('chat');
}

async function sendMessage() {
  const text = els.chatInput.value.trim();
  if (!text || state.isLoading) return;

  state.isLoading = true;
  els.chatInput.value = '';
  updateCharCount();
  els.btnSend.disabled = true;

  // Hide welcome, show messages
  els.welcomeScreen.classList.add('hidden');

  // Append user message immediately
  const userMsg = {
    id: 'tmp-' + Date.now(),
    role: 'user',
    content: text,
    timestamp: new Date().toISOString(),
  };
  appendMessage(userMsg);
  scrollToBottom();

  // Show typing indicator
  showTypingIndicator();

  // Show loading overlay with pipeline animation
  showLoading();

  try {
    const data = await apiFetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: state.currentConvId,
        message: text,
      }),
    });

    state.currentConvId = data.conversation_id;

    // Remove typing indicator
    removeTypingIndicator();
    hideLoading();

    // Append assistant message
    if (data.message) {
      appendMessage(data.message);
      state.lastTrace = data.message.trace;
      if (state.traceOpen && data.message.trace) {
        renderTrace(data.message.trace);
      }
    }

    scrollToBottom();

    // Reload conversations
    await loadConversations();

  } catch (e) {
    removeTypingIndicator();
    hideLoading();
    appendMessage({
      id: 'err-' + Date.now(),
      role: 'assistant',
      content: `⚠️ **Error**: ${e.message}\n\nPlease check if the backend server is running on port 8080.`,
      timestamp: new Date().toISOString(),
    });
    scrollToBottom();
  }

  state.isLoading = false;
  els.btnSend.disabled = els.chatInput.value.trim().length === 0;
  els.chatInput.focus();
}

function appendMessage(msg) {
  const isUser = msg.role === 'user';
  const div = document.createElement('div');
  div.className = `message ${isUser ? 'user' : 'assistant'}`;
  div.dataset.msgId = msg.id;

  const avatar = isUser ? '👤' : '🤖';
  const name   = isUser ? (state.profile.name || 'User') : (sessionStorage.getItem('gov_ai_model_name') || 'Assistant');

  const contentHtml = isUser ? escHtml(msg.content) : renderMarkdown(msg.content);

  if (!isUser) updateSidebarMetrics(msg);

  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-header">
        <span class="message-author">${name}</span>
        <span class="message-time">${formatTime(msg.timestamp)}</span>
      </div>
      <div class="message-body">${contentHtml}</div>
    </div>
  `;

  if (msg.trace) div._trace = msg.trace;
  els.messagesList.appendChild(div);
  scrollToBottom();
}

function updateSidebarMetrics(msg) {
  if (!msg.metrics && !msg.hallucination_report) return;

  const hr = msg.hallucination_report || {};
  const m = msg.metrics || {};
  const currentConf = hr.confidence_score != null ? Math.round(hr.confidence_score * 100) : null;

  // 1. Calculate combined accuracy
  if (currentConf !== null) {
    state.sessionStats.totalConfidence += currentConf;
    state.sessionStats.messageCount++;
  }

  const avgAcc = state.sessionStats.messageCount > 0
    ? Math.round(state.sessionStats.totalConfidence / state.sessionStats.messageCount)
    : 0;

  // 2. Update Sidebar Circle
  if (els.sideAccValue) {
    els.sideAccValue.innerHTML = `${avgAcc}%<span>Accuracy</span>`;
  }

  if (els.sideAccProgress) {
    const offset = 440 - (440 * (avgAcc / 100));
    els.sideAccProgress.style.strokeDashoffset = offset;
  }

  // Update circle color based on average
  if (els.sideAccCircle) {
    els.sideAccCircle.classList.remove('status-safe', 'status-warn', 'status-danger');
    const colorClass = avgAcc >= 85 ? 'status-safe' : avgAcc >= 65 ? 'status-warn' : 'status-danger';
    els.sideAccCircle.classList.add(colorClass);
  }

  // 3. Update Sidebar Tabs (latest message)
  if (els.sideMetricTabs) {
    let tabsHtml = '';
    if (currentConf !== null) {
      const cls = currentConf >= 85 ? 'safe' : currentConf >= 65 ? 'warn' : 'danger';
      tabsHtml += `<span class="metric-badge ${cls}">🎯 ${currentConf}% Confidence</span>`;
    }
    if (hr.verdict) {
      const cls = hr.verdict === 'SAFE' || hr.verdict === 'Reliable' ? 'safe' : 'warn';
      tabsHtml += `<span class="metric-badge ${cls}">${hr.verdict === 'SAFE' || hr.verdict === 'Reliable' ? '✅' : '⚠️'} ${hr.verdict}</span>`;
    }
    if (hr.hallucination_detected) {
      tabsHtml += `<span class="metric-badge danger">🚨 Hallucination detected</span>`;
    }
    if (m.total_execution_time) {
      tabsHtml += `<span class="metric-badge">⏱ ${m.total_execution_time}s latency</span>`;
    }
    if (m.number_of_steps_executed) {
      tabsHtml += `<span class="metric-badge">🤖 ${m.number_of_steps_executed} agents</span>`;
    }
    if (msg.trace) {
      tabsHtml += `<button type="button" class="btn-trace" onclick="openTrace('${msg.id}')">View execution trace →</button>`;
    }
    els.sideMetricTabs.innerHTML = tabsHtml;
  }
}

function resetSidebarMetrics() {
  state.sessionStats = { totalConfidence: 0, messageCount: 0 };
  if (els.sideAccValue) els.sideAccValue.innerHTML = `--%<span>Accuracy</span>`;
  if (els.sideAccProgress) els.sideAccProgress.style.strokeDashoffset = 440;
  if (els.sideMetricTabs) els.sideMetricTabs.innerHTML = '';
  if (els.sideAccCircle) els.sideAccCircle.classList.remove('status-safe', 'status-warn', 'status-danger');
}

function openTrace(msgId) {
  const msgEl = els.messagesList.querySelector(`[data-msg-id="${msgId}"]`);
  if (msgEl && msgEl._trace) {
    renderTrace(msgEl._trace);
    state.traceOpen = true;
    els.tracePanel.classList.add('open');
  }
}

window.openTrace = openTrace;

function toggleSidebarSection(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.toggle('section-collapsed');
  }
}

window.toggleSidebarSection = toggleSidebarSection;

function initCustomSelects() {
  const selects = document.querySelectorAll('.custom-select');
  selects.forEach(select => {
    const trigger = select.querySelector('.select-trigger');
    const options = select.querySelectorAll('.option');
    const valueEl = select.querySelector('.select-value');
    const hiddenInput = select.parentElement.querySelector('input[type="hidden"]');

    // Sync initial state
    if (hiddenInput && hiddenInput.value) {
      options.forEach(opt => {
        if (opt.dataset.value === hiddenInput.value) {
          opt.classList.add('selected');
          valueEl.textContent = opt.textContent;
        }
      });
    }

    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = select.classList.contains('open');
      // Close all first
      document.querySelectorAll('.custom-select').forEach(s => s.classList.remove('open'));
      if (!isOpen) select.classList.add('open');
    });

    options.forEach(opt => {
      opt.addEventListener('click', (e) => {
        e.stopPropagation();
        const val = opt.dataset.value;
        const text = opt.textContent;
        
        valueEl.textContent = text;
        if (hiddenInput) {
          hiddenInput.value = val;
          hiddenInput.dispatchEvent(new Event('change'));
        }

        options.forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        select.classList.remove('open');
      });
    });
  });

  document.addEventListener('click', () => {
    document.querySelectorAll('.custom-select').forEach(s => s.classList.remove('open'));
  });
}

function scrollToBottom() {
  els.messagesContainer.scrollTo({ top: els.messagesContainer.scrollHeight, behavior: 'smooth' });
}

function showTypingIndicator() {
  const el = document.createElement('div');
  el.className = 'typing-indicator';
  el.id = 'typingIndicator';
  el.innerHTML = `
    <div class="message-avatar">
      <svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="14" fill="url(#tGrad)"/><path d="M10 16L16 10L22 16L16 22Z" fill="white" opacity="0.9"/><circle cx="16" cy="16" r="3" fill="white"/><defs><linearGradient id="tGrad" x1="0" y1="0" x2="32" y2="32"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#a855f7"/></linearGradient></defs></svg>
    </div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  els.messagesList.appendChild(el);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

// ── Loading overlay pipeline ───────────────────────────────────────────────
const pipelineStages = [
  { key: 'research', label: 'Research Agent is gathering data...' },
  { key: 'analysis', label: 'Analysis Agent is extracting insights...' },
  { key: 'summary', label: 'Summary Agent is generating report...' },
  { key: 'eval',    label: 'Evaluation Agent is checking facts...' },
];

let loadingTimer = null;

function showLoading() {
  // Reset pipeline
  Object.values(els.pipelineSteps).forEach(el => {
    el.classList.remove('active', 'done');
    el.querySelector('.pipeline-fill').style.width = '0';
  });

  els.loadingOverlay.classList.remove('hidden');
  let stageIdx = 0;

  function nextStage() {
    if (stageIdx > 0) {
      const prev = pipelineStages[stageIdx - 1];
      els.pipelineSteps[prev.key].classList.remove('active');
      els.pipelineSteps[prev.key].classList.add('done');
      els.pipelineSteps[prev.key].querySelector('.pipeline-fill').style.width = '100%';
    }
    if (stageIdx < pipelineStages.length) {
      const stage = pipelineStages[stageIdx];
      els.pipelineSteps[stage.key].classList.add('active');
      els.loadingText.textContent = stage.label;
      stageIdx++;
      loadingTimer = setTimeout(nextStage, 3000);
    }
  }

  nextStage();
}

function hideLoading() {
  clearTimeout(loadingTimer);
  els.loadingOverlay.classList.add('hidden');
  // Mark all done
  Object.values(els.pipelineSteps).forEach(el => {
    el.classList.remove('active');
    el.classList.add('done');
    el.querySelector('.pipeline-fill').style.width = '100%';
  });
}

// ── Trace panel ────────────────────────────────────────────────────────────
function renderTrace(trace) {
  if (!trace || !trace.steps || trace.steps.length === 0) {
    els.traceContent.innerHTML = '<div class="trace-empty">No trace data available.</div>';
    return;
  }

  const stepsHtml = trace.steps.map(step => {
    const icons = {
      'Research Agent': '🔍', 'Research': '🔍',
      'Analysis Agent': '📊', 'Analysis': '📊',
      'Summary Agent': '📝', 'Summary': '📝',
      'Evaluation Agent': '✅', 'Evaluation': '✅',
      'Mock Research Component': '🔍', 'Mock Synthesizer Unit': '📝',
    };
    const icon = icons[step.step_name] || icons[step.agent_responsible] || '🤖';
    const status = step.status || 'success';
    const duration = step.duration ? `${step.duration.toFixed(2)}s` : '';
    const halluc = step.hallucination_flag ? '<span style="color: var(--red)">⚠ Hallucination</span>' : '';

    return `
      <div class="trace-step ${status}">
        <div class="trace-step-icon">${icon}</div>
        <div class="trace-step-body">
          <div class="trace-step-name">${step.step_name || 'Step'}</div>
          <div class="trace-step-agent">${step.agent_responsible || ''}</div>
          <div class="trace-step-duration">${duration}</div>
          <div class="trace-step-status">${status} ${halluc}</div>
        </div>
      </div>
    `;
  }).join('');

  els.traceContent.innerHTML = `
    <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:10px; font-family:var(--mono)">
      Trace ID: ${trace.trace_id ? trace.trace_id.slice(0, 8) + '...' : 'N/A'}
    </div>
    ${stepsHtml}
  `;
}

// ── View switching ─────────────────────────────────────────────────────────
function showView(view) {
  state.currentView = view;
  els.chatView.classList.toggle('hidden', view !== 'chat');
  els.agentsView.classList.toggle('hidden', view !== 'agents');
  els.settingsView.classList.toggle('hidden', view !== 'settings');
  els.profileView.classList.toggle('hidden', view !== 'profile');
  
  els.btnAgents.classList.toggle('active', view === 'agents');
  els.btnSettings.classList.toggle('active', view === 'settings');
  els.btnProfile.classList.toggle('active', view === 'profile');
  
  if (view === 'profile') renderProfile();
}

function renderProfile() {
  const p = state.profile;
  const f = els.profileForm;
  f.profName.value = p.name;
  f.profRole.value = p.role;
  f.profEmail.value = sessionStorage.getItem('gov_ai_user') || '';
  f.profDept.value = p.dept;
}

function handleLogout() {
  if (!confirm('Are you sure you want to sign out?')) return;
  console.log('[App] Logging out...');
  sessionStorage.clear();
  location.reload(); // Hard reset for security
}

function handleBackToModels() {
  console.log('[App] Returning to model selection...');
  sessionStorage.removeItem('gov_ai_model');
  sessionStorage.removeItem('gov_ai_model_name');
  fadeOutScreen('main', () => {
    // Show model screen again
    document.getElementById('modelScreen').classList.remove('hidden');
    renderModelGrid();
    showScreen('modelScreen');
  });
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:90px; right:24px; z-index:9999;
    padding:10px 18px; border-radius:10px; font-size:0.85rem; font-weight:500;
    background:${type === 'error' ? 'rgba(239,68,68,0.95)' : 'rgba(99,102,241,0.95)'};
    color:white; box-shadow:0 4px 20px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── Auto-resize textarea ───────────────────────────────────────────────────
function autoResize() {
  const ta = els.chatInput;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}

function updateCharCount() {
  const len = els.chatInput.value.length;
  els.charCount.textContent = `${len} / 4000`;
  els.btnSend.disabled = len === 0 || state.isLoading;
}

// ── Event Listeners ────────────────────────────────────────────────────────
function bindEvents() {
  // Init custom UI
  initCustomSelects();

  // Sidebar toggle
  els.sidebarToggle.addEventListener('click', () => {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    els.sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  });

  // Mobile menu
  els.mobileMenu.addEventListener('click', () => {
    els.sidebar.classList.toggle('mobile-open');
  });

  // New chat
  els.btnNewChat.addEventListener('click', startNewChat);

  // Nav buttons
  els.btnAgents.addEventListener('click', () => showView('agents'));
  els.btnSettings.addEventListener('click', () => showView('settings'));
  els.btnProfile.addEventListener('click', () => showView('profile'));
  els.btnLogout.addEventListener('click', handleLogout);
  els.btnBackToModels.addEventListener('click', handleBackToModels);

  // Profile form
  els.profileForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const f = els.profileForm;
    state.profile = {
      name: f.profName.value,
      role: f.profRole.value,
      dept: f.profDept.value,
    };
    // Update sidebar
    els.sideUserName.textContent = state.profile.name;
    els.sideUserRole.textContent = state.profile.role;
    document.querySelector('.account-avatar').textContent = 
      state.profile.name.split(' ').map(n=>n[0]).join('').toUpperCase().slice(0,2);
    
    showToast('Profile updated successfully');
  });

  // Chat input
  els.chatInput.addEventListener('input', () => {
    autoResize();
    updateCharCount();
  });
  els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Send button
  els.btnSend.addEventListener('click', sendMessage);

  // Suggestion cards
  document.querySelectorAll('.suggestion-card').forEach(card => {
    card.addEventListener('click', () => {
      const text = card.dataset.text;
      if (text) {
        els.chatInput.value = text;
        updateCharCount();
        autoResize();
        sendMessage();
      }
    });
  });

  // Trace panel
  els.btnToggleTrace.addEventListener('click', () => {
    state.traceOpen = !state.traceOpen;
    els.tracePanel.classList.toggle('open', state.traceOpen);
    if (state.traceOpen && state.lastTrace) {
      renderTrace(state.lastTrace);
    }
  });
  els.btnCloseTrace.addEventListener('click', () => {
    state.traceOpen = false;
    els.tracePanel.classList.remove('open');
  });

  // Settings
  els.tempSlider.addEventListener('input', () => {
    els.tempValue.textContent = els.tempSlider.value;
  });

  els.btnSaveOpenrouter.addEventListener('click', () => {
    showToast('OpenRouter API key saved (demo mode — configure in .env)', 'info');
  });
  els.btnSaveTavily.addEventListener('click', () => {
    showToast('Tavily API key saved (demo mode — configure in .env)', 'info');
  });

  els.btnClearHistory.addEventListener('click', async () => {
    if (!confirm('Clear all conversations? This cannot be undone.')) return;
    const ids = state.conversations.map(c => c.id);
    for (const id of ids) {
      await apiFetch(`/api/conversations/${id}`, { method: 'DELETE' }).catch(() => {});
    }
    startNewChat();
    await loadConversations();
    showToast('All conversations cleared');
  });

  // Close mobile sidebar on outside click
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 &&
        els.sidebar.classList.contains('mobile-open') &&
        !els.sidebar.contains(e.target) &&
        !els.mobileMenu.contains(e.target)) {
      els.sidebar.classList.remove('mobile-open');
    }
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  console.log('[App] DOM Content Loaded. Initializing steps...');
  
  try {
    initLogin();
  } catch (e) {
    console.error('[App] Failed to init login:', e);
  }

  try {
    initModelSelect();
  } catch (e) {
    console.error('[App] Failed to init model select:', e);
  }

  // If already logged in this session, skip login
  const sessionUser  = sessionStorage.getItem('gov_ai_user');
  const sessionModel = sessionStorage.getItem('gov_ai_model');

  if (sessionUser && sessionModel) {
    console.log('[App] Session found. Jumping to chat.');
    document.getElementById('loginScreen').classList.add('hidden');
    document.getElementById('modelScreen').classList.add('hidden');
    initApp();
  } else if (sessionUser) {
    console.log('[App] User logged in. Jumping to model select.');
    document.getElementById('loginScreen').classList.add('hidden');
    renderModelGrid();
    showScreen('modelScreen');
  } else {
    console.log('[App] No session. Showing login screen.');
  }
});
