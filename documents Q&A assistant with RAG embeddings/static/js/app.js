/**
 * DocuRAG Studio - Frontend Application Controller
 * Handles document ingestion, vector store toggling, chat interactions,
 * markdown parsing, and source page citation drawers.
 */

document.addEventListener("DOMContentLoaded", () => {
    initApp();
    setupDropZone();
});

// App State
const state = {
    activeStore: "FAISS",
    docsCount: 0,
    chunksCount: 0,
    isQuerying: false
};

// ==========================================
// 1. Initialization and Status Fetching
// ==========================================
async function initApp() {
    try {
        const res = await fetch("/api/status");
        const json = await res.json();
        if (json.success) {
            updateDashboardUI(json.data);
        }
    } catch (err) {
        console.error("Failed to load system status:", err);
    }
}

function updateDashboardUI(data) {
    state.activeStore = (data.active_store || "FAISS").toUpperCase();
    state.docsCount = data.total_documents || 0;
    state.chunksCount = data.total_chunks || 0;

    // Update Metrics
    document.getElementById("metric-docs-count").textContent = state.docsCount;
    document.getElementById("metric-chunks-count").textContent = state.chunksCount;
    document.getElementById("metric-chunk-size").textContent = data.chunk_size || 800;
    document.getElementById("metric-chunk-overlap").textContent = data.chunk_overlap || 150;

    // Update Store Buttons
    const btnFaiss = document.getElementById("btn-store-faiss");
    const btnChroma = document.getElementById("btn-store-chroma");
    const badgeStore = document.getElementById("badge-active-store");

    if (state.activeStore === "FAISS") {
        btnFaiss.classList.add("active");
        btnChroma.classList.remove("active");
        badgeStore.textContent = "FAISS INDEX";
        badgeStore.className = "badge badge-cyan";
    } else {
        btnChroma.classList.add("active");
        btnFaiss.classList.remove("active");
        badgeStore.textContent = "CHROMADB";
        badgeStore.className = "badge badge-accent";
    }

    // Model Provider Badge
    const modelBadge = document.getElementById("badge-model-provider");
    let providerName = (data.llm_provider || "gemini").toUpperCase();
    if (providerName === "GEMINI" && !data.has_google_key) {
        providerName += " (Offline Fallback)";
    } else if (providerName === "OPENAI" && !data.has_openai_key) {
        providerName += " (Offline Fallback)";
    }
    modelBadge.innerHTML = `<i class="fa-solid fa-microchip"></i> ${providerName}`;

    // Update Files List
    renderFilesList(data.documents || []);

    // Prefill settings form
    if (document.getElementById("setting-provider")) {
        document.getElementById("setting-provider").value = data.llm_provider || "gemini";
        document.getElementById("setting-top-k").value = data.top_k || 4;
        document.getElementById("setting-chunk-size").value = data.chunk_size || 800;
        document.getElementById("setting-chunk-overlap").value = data.chunk_overlap || 150;
    }
}

function renderFilesList(files) {
    const listEl = document.getElementById("indexed-files-list");
    const countBadge = document.getElementById("files-count-badge");
    countBadge.textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;

    if (!files || files.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-folder-open"></i>
                <p>No documents indexed yet.</p>
                <span>Upload a PDF or load the sample paper above.</span>
            </div>
        `;
        return;
    }

    listEl.innerHTML = files.map(file => `
        <div class="file-item">
            <div class="file-item-left">
                <i class="fa-solid fa-file-pdf"></i>
                <div class="file-meta-text">
                    <span class="file-meta-name" title="${escapeHtml(file.file_name)}">${escapeHtml(file.file_name)}</span>
                    <span class="file-meta-sub">${file.pages_indexed} page(s) &bull; ${file.chunk_count} chunks</span>
                </div>
            </div>
            <span class="badge-pill">${file.chunk_count}c</span>
        </div>
    `).join("");
}

// ==========================================
// 2. Drag and Drop & Upload Management
// ==========================================
function setupDropZone() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-active");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-active");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-active");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files);
        }
    });
}

async function handleFileUpload(fileList) {
    const progressContainer = document.getElementById("upload-progress-container");
    const progressBar = document.getElementById("upload-progress-bar");
    const statusText = document.getElementById("upload-status-text");
    const statusPercent = document.getElementById("upload-status-percent");

    progressContainer.classList.remove("hidden");
    progressBar.style.width = "25%";
    statusPercent.textContent = "25%";
    statusText.textContent = `Uploading ${fileList.length} file(s)...`;

    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("files", fileList[i]);
    }

    try {
        progressBar.style.width = "65%";
        statusPercent.textContent = "65%";
        statusText.textContent = "Chunking text & computing vector embeddings...";

        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const json = await res.json();
        progressBar.style.width = "100%";
        statusPercent.textContent = "100%";

        if (json.success) {
            showToast(json.message, "success");
            initApp();
            appendSystemNotice(`Successfully indexed <strong>${json.documents.length}</strong> document(s) into the <strong>${state.activeStore}</strong> vector store.`);
        } else {
            showToast(json.error || "Upload failed.", "error");
        }
    } catch (err) {
        showToast("Error uploading file: " + err.message, "error");
    } finally {
        setTimeout(() => {
            progressContainer.classList.add("hidden");
            progressBar.style.width = "0%";
        }, 1500);
        document.getElementById("file-input").value = "";
    }
}

async function loadSampleDocument() {
    const btn = document.getElementById("btn-load-sample");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Loading...`;
    btn.disabled = true;

    try {
        const res = await fetch("/api/load-sample", { method: "POST" });
        const json = await res.json();
        if (json.success) {
            showToast("Sample research PDF loaded & indexed!", "success");
            initApp();
            appendSystemNotice(`Preloaded <strong>${json.document.file_name}</strong> (${json.document.pages} pages, ${json.document.chunks} chunks) into <strong>${state.activeStore}</strong>.`);
        } else {
            showToast(json.error || "Failed to load sample document.", "error");
        }
    } catch (err) {
        showToast("Error loading sample: " + err.message, "error");
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// ==========================================
// 3. Vector Store Switching
// ==========================================
async function switchVectorStore(storeType) {
    if (storeType.toUpperCase() === state.activeStore) return;

    try {
        const res = await fetch("/api/switch-store", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ store_type: storeType })
        });
        const json = await res.json();
        if (json.success) {
            showToast(json.message, "success");
            initApp();
            appendSystemNotice(`Vector database engine dynamically switched to <strong>${storeType.toUpperCase()}</strong>.`);
        } else {
            showToast(json.error || "Failed to switch vector store.", "error");
        }
    } catch (err) {
        showToast("Error switching store: " + err.message, "error");
    }
}

// ==========================================
// 4. Chat & Query Processing
// ==========================================
function handleTextareaKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        document.getElementById("chat-form").requestSubmit();
    }
}

function fillPrompt(text) {
    const input = document.getElementById("query-input");
    input.value = text;
    input.focus();
}

async function handleChatSubmit(e) {
    e.preventDefault();
    if (state.isQuerying) return;

    const input = document.getElementById("query-input");
    const query = input.value.trim();
    if (!query) return;

    // Append User Message
    appendMessage("user", query);
    input.value = "";
    input.style.height = "auto";

    // Show Typing Indicator
    const typingId = showTypingIndicator();
    state.isQuerying = true;
    document.getElementById("btn-send-query").disabled = true;

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: query })
        });
        const json = await res.json();
        removeTypingIndicator(typingId);

        if (json.success) {
            appendMessage("assistant", json.data.answer, json.data.citations, json.data.vector_store);
        } else {
            appendMessage("assistant", `⚠️ **Error:** ${json.error || 'Failed to process query.'}`);
        }
    } catch (err) {
        removeTypingIndicator(typingId);
        appendMessage("assistant", `⚠️ **Network Error:** ${err.message}`);
    } finally {
        state.isQuerying = false;
        document.getElementById("btn-send-query").disabled = false;
        input.focus();
    }
}

function appendMessage(role, content, citations = [], storeName = "") {
    const container = document.getElementById("chat-messages");
    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${role}`;

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let citationsHtml = "";
    if (citations && citations.length > 0) {
        const uniqueId = "cit_" + Math.random().toString(36).substring(2, 9);
        citationsHtml = `
            <div class="citations-box">
                <div class="citations-header" onclick="toggleCitations('${uniqueId}')">
                    <div class="citation-title-group">
                        <i class="fa-solid fa-quote-left"></i>
                        <span>Source Citations (${citations.length} Passages &bull; ${storeName || state.activeStore})</span>
                    </div>
                    <i id="icon_${uniqueId}" class="fa-solid fa-chevron-down"></i>
                </div>
                <div id="${uniqueId}" class="citations-body">
                    ${citations.map((c, i) => `
                        <div class="citation-item">
                            <div class="citation-item-header">
                                <div>
                                    <span class="citation-source-pill">${escapeHtml(c.file_name)}</span>
                                    <span class="citation-page-badge">Page ${c.page}</span>
                                </div>
                                <span class="citation-score-badge">Relevance: ${(c.relevance_score * 100).toFixed(1)}%</span>
                            </div>
                            <p class="citation-snippet">"${escapeHtml(c.snippet)}"</p>
                        </div>
                    `).join("")}
                </div>
            </div>
        `;
    }

    wrapper.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid ${role === 'user' ? 'fa-user' : 'fa-sparkles'}"></i>
        </div>
        <div class="message-bubble">
            <div class="message-header-meta">
                <strong>${role === 'user' ? 'You' : 'DocuRAG Assistant'}</strong>
                <span class="timestamp">${now}</span>
            </div>
            <div class="message-content">
                ${formatMarkdown(content)}
            </div>
            ${citationsHtml}
        </div>
    `;

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

function toggleCitations(id) {
    const el = document.getElementById(id);
    const icon = document.getElementById("icon_" + id);
    if (!el) return;
    if (el.classList.contains("hidden")) {
        el.classList.remove("hidden");
        icon.className = "fa-solid fa-chevron-down";
    } else {
        el.classList.add("hidden");
        icon.className = "fa-solid fa-chevron-right";
    }
}

function appendSystemNotice(htmlNotice) {
    const container = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = "message-content";
    div.style.textAlign = "center";
    div.style.fontSize = "0.8rem";
    div.style.color = "var(--text-muted)";
    div.style.margin = "0.5rem 0";
    div.innerHTML = `<i class="fa-solid fa-circle-info text-cyan"></i> ${htmlNotice}`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTypingIndicator() {
    const container = document.getElementById("chat-messages");
    const id = "typing_" + Date.now();
    const wrapper = document.createElement("div");
    wrapper.id = id;
    wrapper.className = "message-wrapper assistant";
    wrapper.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-sparkles"></i>
        </div>
        <div class="message-bubble">
            <div class="typing-dots">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function clearChatMessages() {
    const container = document.getElementById("chat-messages");
    container.innerHTML = `
        <div class="message-wrapper assistant">
            <div class="message-avatar">
                <i class="fa-solid fa-sparkles"></i>
            </div>
            <div class="message-bubble">
                <div class="message-content">
                    <p>Chat window cleared. Ask a new question based on your indexed documents!</p>
                </div>
            </div>
        </div>
    `;
}

// ==========================================
// 5. Settings Modal Management
// ==========================================
function openSettingsModal() {
    document.getElementById("settings-modal").classList.remove("hidden");
}

function closeSettingsModal() {
    document.getElementById("settings-modal").classList.add("hidden");
}

function closeSettingsModalOnOutsideClick(e) {
    if (e.target.id === "settings-modal") {
        closeSettingsModal();
    }
}

function toggleApiKeyFields() {
    const provider = document.getElementById("setting-provider").value;
    const geminiGroup = document.getElementById("group-gemini-key");
    const openaiGroup = document.getElementById("group-openai-key");

    if (provider === "gemini") {
        geminiGroup.classList.remove("hidden");
        openaiGroup.classList.add("hidden");
    } else if (provider === "openai") {
        geminiGroup.classList.add("hidden");
        openaiGroup.classList.remove("hidden");
    } else {
        geminiGroup.classList.add("hidden");
        openaiGroup.classList.add("hidden");
    }
}

function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    input.type = input.type === "password" ? "text" : "password";
}

async function handleSettingsSave(e) {
    e.preventDefault();
    const provider = document.getElementById("setting-provider").value;
    const googleKey = document.getElementById("setting-google-key").value.trim();
    const openaiKey = document.getElementById("setting-openai-key").value.trim();
    const topK = document.getElementById("setting-top-k").value;
    const temperature = document.getElementById("setting-temperature").value;
    const chunkSize = document.getElementById("setting-chunk-size").value;
    const chunkOverlap = document.getElementById("setting-chunk-overlap").value;

    const payload = {
        llm_provider: provider,
        top_k: topK,
        temperature: temperature,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap
    };
    if (googleKey) payload.google_api_key = googleKey;
    if (openaiKey) payload.openai_api_key = openaiKey;

    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const json = await res.json();
        if (json.success) {
            showToast("Settings updated successfully!", "success");
            closeSettingsModal();
            initApp();
        } else {
            showToast(json.error || "Failed to update settings.", "error");
        }
    } catch (err) {
        showToast("Error updating settings: " + err.message, "error");
    }
}

// ==========================================
// 6. Reset All Data
// ==========================================
async function clearAllData() {
    if (!confirm("Are you sure you want to clear all indexed documents, vector stores, and chat memory?")) {
        return;
    }

    try {
        const res = await fetch("/api/clear", { method: "POST" });
        const json = await res.json();
        if (json.success) {
            showToast("All data cleared successfully.", "success");
            clearChatMessages();
            initApp();
        } else {
            showToast(json.error || "Failed to clear data.", "error");
        }
    } catch (err) {
        showToast("Error clearing data: " + err.message, "error");
    }
}

// ==========================================
// 7. Utility Helpers
// ==========================================
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type === 'error' ? 'toast-error' : type === 'success' ? 'toast-success' : ''}`;
    
    let icon = "fa-circle-info text-cyan";
    if (type === "success") icon = "fa-circle-check text-emerald";
    if (type === "error") icon = "fa-circle-exclamation text-rose";

    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // Bold **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italics *text*
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Inline code `code`
    html = html.replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 2px 5px; border-radius: 4px; font-family: var(--font-mono);">$1</code>');

    // Bullet points
    html = html.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Blockquotes
    html = html.replace(/^&gt;\s+(.*)$/gm, '<blockquote>$1</blockquote>');

    // Line breaks to paragraphs
    const paragraphs = html.split(/\n\s*\n/);
    return paragraphs.map(p => {
        p = p.trim();
        if (p.startsWith('<ul>') || p.startsWith('<blockquote>')) return p;
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join("");
}
