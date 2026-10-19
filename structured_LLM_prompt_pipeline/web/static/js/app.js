/**
 * Structured LLM Prompt Pipeline & JSON Extractor - Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    samples: { invoice: [], resume: [] },
    currentDomain: "invoice",
    lastResult: null,
    showingRaw: false,
  };

  // DOM Elements
  const domainSelect = document.getElementById("domainSelect");
  const providerSelect = document.getElementById("providerSelect");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const temperatureInput = document.getElementById("temperatureInput");
  const tempDisplay = document.getElementById("tempDisplay");
  const fewShotToggle = document.getElementById("fewShotToggle");
  const maxRetriesInput = document.getElementById("maxRetriesInput");
  const retriesDisplay = document.getElementById("retriesDisplay");
  const simulateFailToggle = document.getElementById("simulateFailToggle");
  const customInstructions = document.getElementById("customInstructions");

  const sampleSelect = document.getElementById("sampleSelect");
  const dropzone = document.getElementById("dropzone");
  const fileUploadInput = document.getElementById("fileUploadInput");
  const documentEditor = document.getElementById("documentEditor");
  const charCount = document.getElementById("charCount");
  const wordCount = document.getElementById("wordCount");

  const extractBtn = document.getElementById("extractBtn");
  const extractBtnText = document.getElementById("extractBtnText");
  const extractSpinner = document.getElementById("extractSpinner");

  const copyJsonBtn = document.getElementById("copyJsonBtn");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");
  const toggleRawBtn = document.getElementById("toggleRawBtn");

  const auditCard = document.getElementById("auditCard");
  const statusBadge = document.getElementById("statusBadge");
  const metricLatency = document.getElementById("metricLatency");
  const metricAttempts = document.getElementById("metricAttempts");
  const metricEngine = document.getElementById("metricEngine");
  const auditLogText = document.getElementById("auditLogText");

  const entityCardsGrid = document.getElementById("entityCardsGrid");
  const emptyPlaceholder = document.getElementById("emptyPlaceholder");
  const jsonViewerContent = document.getElementById("jsonViewerContent");
  const rawViewerContent = document.getElementById("rawViewerContent");

  const promptPreviewContent = document.getElementById("promptPreviewContent");
  const schemaRefContent = document.getElementById("schemaRefContent");

  // Tab switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const targetTabId = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");
      const activeContent = document.getElementById(targetTabId);
      if (activeContent) {
        activeContent.style.display = "flex";
      }

      if (targetTabId === "promptTab") {
        updatePromptPreview();
      } else if (targetTabId === "schemaTab") {
        updateSchemaReference();
      }
    });
  });

  // Range inputs
  temperatureInput.addEventListener("input", (e) => {
    tempDisplay.textContent = e.target.value;
  });

  maxRetriesInput.addEventListener("input", (e) => {
    retriesDisplay.textContent = e.target.value;
  });

  // Editor counts
  documentEditor.addEventListener("input", updateCounts);

  function updateCounts() {
    const text = documentEditor.value;
    charCount.textContent = `${text.length} characters`;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    wordCount.textContent = `${words} words`;
  }

  // Toast Notification
  function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => {
      toast.className = "toast";
    }, 3200);
  }

  // Fetch Samples from Backend
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      state.samples = data;
      populateSampleSelect();
    } catch (err) {
      console.error("Failed to fetch samples:", err);
    }
  }

  function populateSampleSelect() {
    sampleSelect.innerHTML = '<option value="" disabled selected>-- Load Pre-built Messy Document --</option>';
    const list = state.samples[state.currentDomain] || [];
    list.forEach(sample => {
      const opt = document.createElement("option");
      opt.value = sample.id;
      opt.textContent = `📄 ${sample.title}`;
      sampleSelect.appendChild(opt);
    });
  }

  // Domain change handler
  domainSelect.addEventListener("change", (e) => {
    state.currentDomain = e.target.value;
    populateSampleSelect();
    updateSchemaReference();
    updatePromptPreview();

    // Auto load first sample
    const list = state.samples[state.currentDomain] || [];
    if (list.length > 0) {
      sampleSelect.value = list[0].id;
      documentEditor.value = list[0].content;
      updateCounts();
    }
  });

  // Sample select change
  sampleSelect.addEventListener("change", (e) => {
    const sampleId = e.target.value;
    const list = state.samples[state.currentDomain] || [];
    const found = list.find(s => s.id === sampleId);
    if (found) {
      documentEditor.value = found.content;
      updateCounts();
      showToast(`Loaded "${found.title}"`);
    }
  });

  // File Upload Handlers (Drag & Drop + Browse)
  dropzone.addEventListener("click", () => fileUploadInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent-primary)";
    dropzone.style.background = "rgba(99, 102, 241, 0.1)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "";
    dropzone.style.background = "";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";
    dropzone.style.background = "";
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileUploadInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    showToast(`Parsing ${file.name}...`, "info");
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        documentEditor.value = data.text;
        updateCounts();
        showToast(`Loaded ${file.name} (${data.characters} chars)`);
      } else {
        showToast(data.error || "File parsing failed", "error");
      }
    } catch (err) {
      showToast("Network error uploading file", "error");
    }
  }

  // Schema Reference Tab
  async function updateSchemaReference() {
    try {
      const res = await fetch(`/api/schema/${state.currentDomain}`);
      const data = await res.json();
      if (res.ok) {
        schemaRefContent.innerHTML = syntaxHighlight(JSON.stringify(data.schema, null, 2));
      }
    } catch (err) {
      schemaRefContent.textContent = "Failed to load schema.";
    }
  }

  // Live Prompt Preview Tab
  async function updatePromptPreview() {
    try {
      const res = await fetch("/api/prompt_preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          domain: state.currentDomain,
          text: documentEditor.value,
          enable_few_shot: fewShotToggle.checked,
          custom_instructions: customInstructions.value,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        const fullPrompt = `=== [SYSTEM PROMPT] ===\n${data.system_prompt}\n\n=== [USER EXTRACTION PROMPT] ===\n${data.user_prompt}`;
        promptPreviewContent.textContent = fullPrompt;
      }
    } catch (err) {
      promptPreviewContent.textContent = "Failed to build prompt preview.";
    }
  }

  // Syntax Highlighting for JSON
  function syntaxHighlight(json) {
    if (!json) return "";
    json = json.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return json.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (match) => {
        let cls = "json-number";
        if (/^"/.test(match)) {
          if (/:$/.test(match)) {
            cls = "json-key";
          } else {
            cls = "json-string";
          }
        } else if (/true|false/.test(match)) {
          cls = "json-boolean";
        } else if (/null/.test(match)) {
          cls = "json-null";
        }
        return `<span class="${cls}">${match}</span>`;
      }
    );
  }

  // Render Entity Highlights Cards
  function renderEntityHighlights(domain, data) {
    entityCardsGrid.innerHTML = "";
    if (!data) return;

    if (domain === "invoice") {
      const vendorName = data.vendor?.name || "Unknown";
      const grandTotal = data.financials?.grand_total != null ? `${data.financials.grand_total.toFixed(2)} ${data.financials.currency || 'USD'}` : "N/A";
      const invoiceNo = data.metadata?.invoice_number || "N/A";

      entityCardsGrid.innerHTML = `
        <div class="entity-stat-card">
          <span class="stat-label">Vendor Issuer</span>
          <span class="stat-value">${escapeHtml(vendorName)}</span>
        </div>
        <div class="entity-stat-card">
          <span class="stat-label">Invoice Number</span>
          <span class="stat-value">${escapeHtml(invoiceNo)}</span>
        </div>
        <div class="entity-stat-card">
          <span class="stat-label">Total Payable</span>
          <span class="stat-value highlight">${escapeHtml(grandTotal)}</span>
        </div>
      `;
    } else {
      const candidateName = data.personal_info?.full_name || "Unknown";
      const seniority = data.seniority_level || "N/A";
      const expYears = data.total_years_experience != null ? `${data.total_years_experience} Years` : "N/A";

      entityCardsGrid.innerHTML = `
        <div class="entity-stat-card">
          <span class="stat-label">Candidate Name</span>
          <span class="stat-value">${escapeHtml(candidateName)}</span>
        </div>
        <div class="entity-stat-card">
          <span class="stat-label">Seniority Grade</span>
          <span class="stat-value highlight">${escapeHtml(seniority)}</span>
        </div>
        <div class="entity-stat-card">
          <span class="stat-label">Industry Exp</span>
          <span class="stat-value">${escapeHtml(expYears)}</span>
        </div>
      `;
    }
    entityCardsGrid.style.display = "grid";
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/[&<>"']/g, m => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[m]);
  }

  // Execute Extraction Handler
  extractBtn.addEventListener("click", async () => {
    const text = documentEditor.value.trim();
    if (!text) {
      showToast("Please enter document text or select a sample.", "error");
      return;
    }

    // UI Loading State
    extractBtn.disabled = true;
    extractSpinner.style.display = "inline-block";
    extractBtnText.textContent = "Extracting & Validating Schema...";

    try {
      const payload = {
        domain: state.currentDomain,
        text: text,
        provider: providerSelect.value,
        api_key: apiKeyInput.value.trim() || undefined,
        temperature: parseFloat(temperatureInput.value),
        enable_few_shot: fewShotToggle.checked,
        max_retries: parseInt(maxRetriesInput.value),
        custom_instructions: customInstructions.value.trim() || undefined,
        simulate_fail_once: simulateFailToggle.checked,
      };

      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      state.lastResult = result;

      if (result.success) {
        showToast("✓ Schema validation succeeded!");
        renderResults(result);
      } else {
        showToast("Extraction failed: " + (result.error_message || "Unknown error"), "error");
        renderFailure(result);
      }
    } catch (err) {
      showToast("Network error executing extraction pipeline.", "error");
    } finally {
      extractBtn.disabled = false;
      extractSpinner.style.display = "none";
      extractBtnText.textContent = "⚡ Extract & Validate JSON Schema";
    }
  });

  function renderResults(result) {
    emptyPlaceholder.style.display = "none";
    jsonViewerContent.style.display = "block";
    rawViewerContent.style.display = "none";
    state.showingRaw = false;
    toggleRawBtn.textContent = "👁️ Raw View";

    // Format JSON
    const jsonFormatted = JSON.stringify(result.validated_data || result.raw_json_dict, null, 2);
    jsonViewerContent.innerHTML = syntaxHighlight(jsonFormatted);
    rawViewerContent.textContent = result.raw_llm_output || "";

    // Audit Card
    auditCard.style.display = "flex";
    if (result.attempts > 1) {
      statusBadge.className = "status-badge warning";
      statusBadge.textContent = `⚡ SELF-HEALED (Attempt ${result.attempts})`;
    } else {
      statusBadge.className = "status-badge success";
      statusBadge.textContent = "✓ PYDANTIC VALIDATED";
    }

    metricLatency.textContent = `${result.latency_seconds.toFixed(2)}s`;
    metricAttempts.textContent = `${result.attempts}`;
    metricEngine.textContent = `${result.provider_used || "Provider"}`;

    const trailSummaries = (result.audit_trail || []).map(a => 
      `Attempt ${a.attempt}: [${a.status}] ${a.note || a.error || ''} (${a.latency}s)`
    ).join(" → ");
    auditLogText.textContent = trailSummaries;

    // Entity Highlights
    renderEntityHighlights(result.domain, result.validated_data);
  }

  function renderFailure(result) {
    emptyPlaceholder.style.display = "none";
    jsonViewerContent.style.display = "block";
    rawViewerContent.style.display = "none";

    auditCard.style.display = "flex";
    statusBadge.className = "status-badge error";
    statusBadge.textContent = "✗ VALIDATION FAILED";
    metricLatency.textContent = `${result.latency_seconds?.toFixed(2) || '0.00'}s`;
    metricAttempts.textContent = `${result.attempts || 1}`;
    metricEngine.textContent = `${result.provider_used || 'Provider'}`;

    auditLogText.textContent = result.error_message || "Schema constraints violated.";
    jsonViewerContent.textContent = result.error_message || "Error occurred during pipeline processing.";
  }

  // Copy JSON Handler
  copyJsonBtn.addEventListener("click", () => {
    if (!state.lastResult) {
      showToast("No extracted JSON to copy.", "error");
      return;
    }
    const jsonStr = JSON.stringify(state.lastResult.validated_data || state.lastResult.raw_json_dict, null, 2);
    navigator.clipboard.writeText(jsonStr).then(() => {
      showToast("JSON copied to clipboard!");
    });
  });

  // Download JSON Handler
  downloadJsonBtn.addEventListener("click", () => {
    if (!state.lastResult) {
      showToast("No extracted JSON to download.", "error");
      return;
    }
    const jsonStr = JSON.stringify(state.lastResult.validated_data || state.lastResult.raw_json_dict, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.currentDomain}_extracted_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded JSON file!");
  });

  // Toggle Raw LLM Output Handler
  toggleRawBtn.addEventListener("click", () => {
    if (!state.lastResult) return;
    state.showingRaw = !state.showingRaw;
    if (state.showingRaw) {
      jsonViewerContent.style.display = "none";
      rawViewerContent.style.display = "block";
      toggleRawBtn.textContent = "✨ JSON View";
    } else {
      jsonViewerContent.style.display = "block";
      rawViewerContent.style.display = "none";
      toggleRawBtn.textContent = "👁️ Raw View";
    }
  });

  // Initialize
  loadSamples().then(() => {
    // Initial sample prefill
    const list = state.samples[state.currentDomain] || [];
    if (list.length > 0) {
      sampleSelect.value = list[0].id;
      documentEditor.value = list[0].content;
      updateCounts();
    }
    updateSchemaReference();
  });
});
