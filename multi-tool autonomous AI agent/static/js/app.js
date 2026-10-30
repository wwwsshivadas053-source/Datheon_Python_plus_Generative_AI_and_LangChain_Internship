/* ═══════════════════════════════════════════════════
   Multi-Tool AI Agent — Frontend Logic
   Reliable async fetch with live reasoning timer
   ═══════════════════════════════════════════════════ */

let isLoading = false;

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  const input = document.getElementById('queryInput');
  if (input) input.focus();
});

// ── Status check ─────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const dot   = document.getElementById('statusDot');
    const lbl   = dot.nextElementSibling.querySelector('.status-label');
    const model = dot.nextElementSibling.querySelector('.status-model');
    const top   = document.getElementById('topProvider');

    if (data.status === 'ok') {
      dot.className = 'status-dot ok';
      lbl.textContent = 'Agent Ready';
    } else {
      dot.className = 'status-dot err';
      lbl.textContent = 'API Key Missing';
    }
    model.textContent = `${data.provider} / ${data.model}`;
    top.textContent = data.provider;
  } catch {
    const dot = document.getElementById('statusDot');
    if (dot) dot.className = 'status-dot err';
  }
}

// ── Sidebar toggle ────────────────────────────────────────────────────────────
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  sb.classList.toggle('collapsed');
  sb.classList.toggle('open');
}

// ── Input helpers ────────────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

function handleKey(e) {
  const input = document.getElementById('queryInput');
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const val = input.value.trim();
    if (val === '/clear') { input.value = ''; clearChat(); return; }
    sendQuery();
  }
}

function useExample(btn) {
  const text = btn.textContent.replace(/^[🔍🧮📄🔀]\s*/, '').trim();
  const input = document.getElementById('queryInput');
  input.value = text;
  autoResize(input);
  input.focus();
}

// ── Clear ────────────────────────────────────────────────────────────────────
async function clearChat() {
  try {
    await fetch('/api/clear', { method: 'POST' });
  } catch {}
  document.getElementById('messages').innerHTML = '';
  document.getElementById('welcome').style.display = '';
}

// ── setLoading ───────────────────────────────────────────────────────────────
function setLoading(on) {
  isLoading = on;
  const btn      = document.getElementById('sendBtn');
  const sendIcon = document.getElementById('sendIcon');
  const loadIcon = document.getElementById('loadIcon');
  const input    = document.getElementById('queryInput');

  if (btn) btn.disabled = on;
  if (input) input.disabled = on;
  if (sendIcon) sendIcon.classList.toggle('hidden', on);
  if (loadIcon) loadIcon.classList.toggle('hidden', !on);
  if (!on && input) input.focus();
}

// ── Scroll to bottom ─────────────────────────────────────────────────────────
function scrollBottom() {
  const w = document.getElementById('chatWindow');
  if (w) w.scrollTop = w.scrollHeight;
}

// ── Tool badge html ──────────────────────────────────────────────────────────
function toolBadge(tool) {
  const map = {
    web_search:          ['🔍', 'badge-search', 'Web Search'],
    python_calculator:   ['🧮', 'badge-calc',   'Calculator'],
    file_reader:         ['📄', 'badge-file',   'File Reader'],
  };
  const [icon, cls, label] = map[tool] || ['⚙️', 'badge-other', tool];
  return `<span class="tool-used-badge ${cls}">${icon} ${label}</span>`;
}

// ── Escape HTML ───────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Append user bubble ───────────────────────────────────────────────────────
function appendUser(text) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.textContent = text;
  msgs.appendChild(div);
  scrollBottom();
}

// ── Append thinking bubble ───────────────────────────────────────────────────
function appendThinking() {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg-agent';
  div.id = 'thinkingBubble';
  div.innerHTML = `
    <div class="thinking-bubble">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
      <span>Agent is reasoning (0s)...</span>
    </div>`;
  msgs.appendChild(div);
  scrollBottom();
  return div;
}

// ── Append agent response ────────────────────────────────────────────────────
function appendAgentResponse(steps, answer, timestamp) {
  const tb = document.getElementById('thinkingBubble');
  if (tb) tb.remove();

  const msgs = document.getElementById('messages');
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-agent';

  // Build reasoning steps HTML
  let stepsHtml = '';
  if (steps && steps.length > 0) {
    const toolsUsed = [...new Set(steps.map(s => s.tool))];
    const badgesHtml = toolsUsed.map(toolBadge).join('');

    let stepsInner = '';
    steps.forEach((step, i) => {
      const obsShort = step.observation && step.observation.length > 400
        ? step.observation.slice(0, 400) + '…'
        : (step.observation || '');

      stepsInner += `
        <div class="step">
          <div class="step-num">STEP ${i + 1} ${toolBadge(step.tool)}</div>
          ${step.thought ? `
          <div class="step-row">
            <span class="step-label lbl-thought">Thought</span>
            <span class="step-text">${esc(step.thought)}</span>
          </div>` : ''}
          <div class="step-row">
            <span class="step-label lbl-action">Action</span>
            <span class="step-text">${esc(step.tool)}</span>
          </div>
          <div class="step-row">
            <span class="step-label lbl-input">Input</span>
            <span class="step-text mono">${esc(step.input || '')}</span>
          </div>
          <div class="step-row">
            <span class="step-label lbl-observe">Observe</span>
            <span class="step-text">${esc(obsShort)}</span>
          </div>
        </div>`;
    });

    stepsHtml = `
      <div class="reasoning-chain">
        <div class="reasoning-header" onclick="toggleChain(this)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          <span class="reasoning-header-text">Reasoning Chain · ${steps.length} step${steps.length > 1 ? 's' : ''} · ${badgesHtml}</span>
          <svg class="chain-toggle open" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2">
            <polyline points="18 15 12 9 6 15"/>
          </svg>
        </div>
        <div class="reasoning-steps">${stepsInner}</div>
      </div>`;
  }

  // Format timestamp
  let ts = '';
  if (timestamp) {
    const d = new Date(timestamp);
    ts = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Final answer
  const answerHtml = `
    <div class="answer-card">
      <div class="answer-header">
        <span class="answer-header-icon">✅</span>
        <span class="answer-header-text">Final Answer</span>
        ${ts ? `<span class="answer-timestamp">${ts}</span>` : ''}
      </div>
      <div class="answer-body">${esc(answer)}</div>
    </div>`;

  wrapper.innerHTML = stepsHtml + answerHtml;
  msgs.appendChild(wrapper);
  scrollBottom();
}

// ── Toggle reasoning chain ────────────────────────────────────────────────────
function toggleChain(header) {
  const steps = header.nextElementSibling;
  const arrow = header.querySelector('.chain-toggle');
  const hidden = steps.style.display === 'none';
  steps.style.display = hidden ? '' : 'none';
  arrow.classList.toggle('open', hidden);
}

// ── Append error ─────────────────────────────────────────────────────────────
function appendError(msg) {
  const tb = document.getElementById('thinkingBubble');
  if (tb) tb.remove();

  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg-agent';
  div.innerHTML = `<div class="error-card"><strong>⚠️ Error</strong>${esc(msg)}</div>`;
  msgs.appendChild(div);
  scrollBottom();
}

// ── Main send function ───────────────────────────────────────────────────────
async function sendQuery() {
  if (isLoading) return;

  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  // Hide welcome banner
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  // Reset input field
  input.value = '';
  input.style.height = 'auto';

  // Display user message
  appendUser(query);
  setLoading(true);

  // Animated reasoning timer
  const startTime = Date.now();
  const tb = appendThinking();
  const timer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const span = tb.querySelector('span:last-child');
    if (span) {
      if (elapsed < 3) {
        span.textContent = `Agent is reasoning (${elapsed}s)...`;
      } else if (elapsed < 8) {
        span.textContent = `Invoking tools & gathering data (${elapsed}s)...`;
      } else {
        span.textContent = `Synthesizing final answer (${elapsed}s)...`;
      }
    }
  }, 1000);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    clearInterval(timer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendError(err.error || `Server returned error (${res.status})`);
      return;
    }

    const data = await res.json();
    let text = data.answer;
    if (Array.isArray(text)) {
      text = text.map(t => typeof t === 'object' ? (t.text || JSON.stringify(t)) : String(t)).join('\n');
    } else if (typeof text === 'object' && text !== null) {
      text = text.text || JSON.stringify(text);
    }

    appendAgentResponse(data.steps, text || 'No answer returned.', data.timestamp);
  } catch (err) {
    clearInterval(timer);
    appendError('Connection error: ' + err.message);
  } finally {
    clearInterval(timer);
    setLoading(false);
  }
}
