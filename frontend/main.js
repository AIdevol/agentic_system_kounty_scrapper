// Set this to your current ngrok URL when exposing the API publicly.
const NGROK_API_BASE = "https://crinkliest-mirna-loftier.ngrok-free.dev";

// Use ngrok override first; otherwise same-origin/public and localhost fallbacks.
function getApiBase() {
  if (NGROK_API_BASE) return NGROK_API_BASE.replace(/\/+$/, "");

  try {
    const o = window.location;
    if (o.origin && !o.origin.startsWith("file:")) {
      if (o.hostname === "localhost" || o.hostname === "127.0.0.1") {
        if (o.port === "8000") return "";
        return "http://127.0.0.1:8000";
      }
      if (o.port && o.port !== "8000") return `${o.protocol}//${o.hostname}:8000`;
      return "";
    }
  } catch (e) {}
  return "http://127.0.0.1:8000";
}
const API_BASE = getApiBase();
const CONVERSATIONS_KEY = "chat_conversations";
const LAST_CONVERSATION_ID_KEY = "chat_last_conversation_id";
const SAVE_HISTORY_CHECKED_KEY = "save_history_checked";
const USER_NAME_KEY = "chat_user_name";

function $(id) {
  return document.getElementById(id);
}

let chatHistory = [];
let conversations = [];
let currentConversationId = null;

function isSaveHistoryChecked() {
  const el = $("save-history-check");
  return el ? el.checked : false;
}

function persistSaveHistoryPreference(checked) {
  try {
    localStorage.setItem(SAVE_HISTORY_CHECKED_KEY, JSON.stringify(!!checked));
  } catch (e) {}
}

function setChatStatus(state, text) {
  const dot = $("status-dot");
  const label = $("chat-status-text");
  if (dot) {
    dot.className = "status-dot " + (state === "error" ? "error" : "idle");
  }
  if (label) label.textContent = text;
}

function toggleEmptyState() {
  const empty = $("chat-empty");
  if (empty) empty.style.display = chatHistory.length === 0 ? "flex" : "none";
}

function generateId() {
  return "id_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9);
}

function getConversationTitle(messages) {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser || typeof firstUser.content !== "string") return "New chat";
  const t = firstUser.content.replace(/\s+/g, " ").trim().slice(0, 36);
  return (t.length < firstUser.content.trim().length ? t + "…" : t) || "New chat";
}

function loadConversationsFromStorage() {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    conversations = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(conversations)) conversations = [];
  } catch (e) {
    conversations = [];
  }
}

function saveConversationsToStorage() {
  if (!isSaveHistoryChecked()) return;
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  } catch (e) {}
}

function renderSidebarHistory() {
  const list = $("sidebar-history-list");
  if (!list) return;
  list.innerHTML = "";
  for (const c of conversations) {
    if (!c || !c.id || !c.title) continue;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "history-item" + (c.id === currentConversationId ? " active" : "");
    item.textContent = c.title;
    item.dataset.conversationId = c.id;
    item.addEventListener("click", () => selectConversation(c.id));
    list.appendChild(item);
  }
}

function selectConversation(id) {
  const convo = conversations.find((c) => c.id === id);
  if (!convo || !Array.isArray(convo.messages)) return;
  currentConversationId = id;
  chatHistory = convo.messages.map((m) => ({
    role: m.role,
    content: m.content,
    attachedFileName: m.attachedFileName,
  }));

  const list = $("chat-messages");
  if (list) list.innerHTML = "";
  for (const m of chatHistory) {
    if (m.role === "user" || m.role === "assistant")
      renderMessageOnly(m.role, m.content, m.attachedFileName);
  }
  const scrollEl = $("chat-view");
  if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  toggleEmptyState();
  renderSidebarHistory();
  try {
    localStorage.setItem(LAST_CONVERSATION_ID_KEY, id);
  } catch (e) {}
}

function renderMessageOnly(role, content, attachedFileName) {
  const list = $("chat-messages");
  const empty = $("chat-empty");
  if (empty) empty.style.display = "none";

  const block = document.createElement("div");
  block.className = "message-block " + role;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "You" : "AI";
  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  // File is shown once in the "Selected:" bar above the input; do not repeat it under every message.
  block.appendChild(avatar);
  block.appendChild(body);
  list.appendChild(block);
}

function renderMessage(role, content, attachedFileName) {
  renderMessageOnly(role, content, attachedFileName);
  const scrollEl = $("chat-view");
  if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  persistCurrentConversation();
  renderSidebarHistory();
}

function persistCurrentConversation() {
  if (!chatHistory.length) return;
  const title = getConversationTitle(chatHistory);
  if (currentConversationId) {
    const convo = conversations.find((c) => c.id === currentConversationId);
    if (convo) {
      convo.title = title;
      convo.messages = chatHistory.map((m) => ({
        role: m.role,
        content: m.content,
        attachedFileName: m.attachedFileName,
      }));
    }
  } else {
    currentConversationId = generateId();
    conversations.unshift({
      id: currentConversationId,
      title,
      messages: chatHistory.map((m) => ({
        role: m.role,
        content: m.content,
        attachedFileName: m.attachedFileName,
      })),
    });
  }
  saveConversationsToStorage();
  try {
    localStorage.setItem(LAST_CONVERSATION_ID_KEY, currentConversationId);
  } catch (e) {}
}

function clearChat() {
  if (chatHistory.length && currentConversationId) {
    const convo = conversations.find((c) => c.id === currentConversationId);
    const messages = chatHistory.map((m) => ({
      role: m.role,
      content: m.content,
      attachedFileName: m.attachedFileName,
    }));
    if (convo) {
      convo.messages = messages;
      convo.title = getConversationTitle(chatHistory);
    } else {
      conversations.unshift({
        id: currentConversationId,
        title: getConversationTitle(chatHistory),
        messages,
      });
    }
    saveConversationsToStorage();
  }

  chatHistory = [];
  currentConversationId = null;
  try {
    localStorage.removeItem(LAST_CONVERSATION_ID_KEY);
  } catch (e) {}

  const list = $("chat-messages");
  if (list) list.innerHTML = "";
  toggleEmptyState();
  renderSidebarHistory();
}

function appendMessage(role, content, attachedFileName) {
  chatHistory.push({ role, content, attachedFileName });
  renderMessage(role, content, attachedFileName);
}

/** Show long messages immediately; use typewriter for shorter ones. */
const TYPEWRITER_THRESHOLD = 1200;

/**
 * Appends an assistant message with a typewriter effect (or immediately for long content).
 */
function appendMessageWithTypewriter(content, delayMs = 18, onComplete = () => {}) {
  const list = $("chat-messages");
  const empty = $("chat-empty");
  if (empty) empty.style.display = "none";

  const block = document.createElement("div");
  block.className = "message-block assistant";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "AI";
  const body = document.createElement("div");
  body.className = "message-content";
  block.appendChild(avatar);
  block.appendChild(body);
  list.appendChild(block);
  const scrollEl = $("chat-view");

  const fullText = content || "(no answer returned)";

  function scrollToBottom() {
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  }

  function finish() {
    chatHistory.push({ role: "assistant", content: fullText });
    persistCurrentConversation();
    renderSidebarHistory();
    onComplete();
  }

  if (fullText.length > TYPEWRITER_THRESHOLD) {
    body.textContent = fullText;
    scrollToBottom();
    finish();
    return;
  }

  let index = 0;
  scrollToBottom();

  function tick() {
    if (index < fullText.length) {
      body.textContent = fullText.slice(0, index + 1);
      index += 1;
      scrollToBottom();
      setTimeout(tick, delayMs);
    } else {
      finish();
    }
  }
  tick();
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file, "UTF-8");
  });
}

function renderStructuredTableFromContexts(contexts) {
  const container = $("table-view");
  const table = $("table-view-table");
  if (!container || !table) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  if (!thead || !tbody) return;

  if (!Array.isArray(contexts) || contexts.length === 0) {
    container.style.display = "none";
    thead.innerHTML = "";
    tbody.innerHTML = "";
    return;
  }

  // Collect headers from metadata keys
  const metaRows = contexts
    .map((c) => c && c.metadata)
    .filter((m) => m && typeof m === "object");
  if (!metaRows.length) {
    container.style.display = "none";
    thead.innerHTML = "";
    tbody.innerHTML = "";
    return;
  }

  const headerSet = new Set();
  for (const row of metaRows) {
    for (const k of Object.keys(row)) {
      headerSet.add(String(k));
    }
  }
  const headers = Array.from(headerSet);
  if (!headers.length) {
    container.style.display = "none";
    thead.innerHTML = "";
    tbody.innerHTML = "";
    return;
  }

  // Render header
  thead.innerHTML = "";
  const trHead = document.createElement("tr");
  for (const h of headers) {
    const th = document.createElement("th");
    th.textContent = h;
    th.style.padding = "4px 6px";
    th.style.borderBottom = "1px solid var(--border)";
    th.style.position = "sticky";
    th.style.top = "0";
    th.style.background = "var(--bg-secondary)";
    th.style.textAlign = "left";
    trHead.appendChild(th);
  }
  thead.appendChild(trHead);

  // Render body (limit to 2000 rows but usually much less)
  tbody.innerHTML = "";
  const MAX_ROWS = 2000;
  const rowsToRender = metaRows.slice(0, MAX_ROWS);
  for (const row of rowsToRender) {
    const tr = document.createElement("tr");
    for (const h of headers) {
      const td = document.createElement("td");
      td.textContent = row[h] != null ? String(row[h]) : "";
      td.style.padding = "3px 6px";
      td.style.borderBottom = "1px solid var(--border)";
      td.style.whiteSpace = "nowrap";
      td.style.textOverflow = "ellipsis";
      td.style.overflow = "hidden";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  container.style.display = "block";
}

async function sendChat() {
  const input = $("chat-input");
  const fileInput = $("file-input");
  const selectedFileBar = $("selected-file-bar");
  const text = input.value.trim();
  if (!text) {
    setChatStatus("error", "Please enter a message first.");
    return;
  }

  // Capture file once at send time (so we use the same reference and don't lose it)
  const file = fileInput && fileInput.files && fileInput.files[0];
  const hasFileUi = selectedFileBar && selectedFileBar.classList.contains("visible");
  if (hasFileUi && !file) {
    setChatStatus("error", "File was cleared. Please select your file again.");
    return;
  }

  appendMessage("user", text, file ? file.name : undefined);
  input.value = "";

  // Build conversation history (all messages before this one) so the agent understands question type and prior questions
  const historyForApi = chatHistory
    .slice(0, -1)
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: (m.content || "").trim() }))
    .filter((m) => m.content.length > 0);

  const userName = ($("user-name-input") && $("user-name-input").value || "").trim();

  $("chat-send").disabled = true;
  setChatStatus("idle", "Thinking…");

  try {
    let res;
    if (file) {
      const formData = new FormData();
      formData.append("question", text);
      formData.append("top_k", "5");
      formData.append("file", file);
      if (historyForApi.length > 0) {
        formData.append("conversation_history", JSON.stringify(historyForApi));
      }
      if (userName) formData.append("user_name", userName);
      // Do not set Content-Type or other headers: browser must set multipart/form-data; boundary=...
      res = await fetch(`${API_BASE}/v1/dataset-rag/query`, {
        method: "POST",
        headers: {},
        body: formData,
      });
    } else {
      const body = { question: text, top_k: 5 };
      if (historyForApi.length > 0) body.conversation_history = historyForApi;
      if (userName) body.user_name = userName;
      res = await fetch(`${API_BASE}/v1/dataset-rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }

    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }

    const data = await res.json();
    let answerText = (data.answer || "").trim() || "I couldn't find matching data for that question.";
    if (/data not available in the (dataset|table)|no relevant rows were found/i.test(answerText)) {
      answerText = "I couldn't find matching data for that question.";
    }

    // If the backend returned structured contexts (real rows), render them as a spreadsheet-style table.
    if (Array.isArray(data.contexts) && data.contexts.length > 0) {
      renderStructuredTableFromContexts(data.contexts);
    } else {
      // Hide table when there is no structured data for this answer.
      renderStructuredTableFromContexts([]);
    }
    setChatStatus("idle", "Writing…");
    appendMessageWithTypewriter(answerText, 18, () => {
      setChatStatus("idle", "Ready");
    });
  } catch (err) {
    console.error(err);
    appendMessage("assistant", `Error: ${(err && err.message) || String(err)}`);
    setChatStatus("error", "Error");
  } finally {
    $("chat-send").disabled = false;
  }
}

function main() {
  loadConversationsFromStorage();

  const saveHistoryCheck = $("save-history-check");
  if (saveHistoryCheck) {
    try {
      const saved = localStorage.getItem(SAVE_HISTORY_CHECKED_KEY);
      if (saved !== null) saveHistoryCheck.checked = JSON.parse(saved);
    } catch (e) {}
    saveHistoryCheck.addEventListener("change", () => {
      persistSaveHistoryPreference(saveHistoryCheck.checked);
      if (saveHistoryCheck.checked) saveConversationsToStorage();
    });
  }

  if (isSaveHistoryChecked()) {
    const lastId = localStorage.getItem(LAST_CONVERSATION_ID_KEY);
    const convo = lastId && conversations.find((c) => c.id === lastId);
    if (convo) selectConversation(convo.id);
  }

  const userNameInput = $("user-name-input");
  if (userNameInput) {
    try {
      const saved = localStorage.getItem(USER_NAME_KEY);
      if (saved != null) userNameInput.value = saved;
    } catch (e) {}
    userNameInput.addEventListener("blur", () => {
      try {
        localStorage.setItem(USER_NAME_KEY, (userNameInput.value || "").trim());
      } catch (e) {}
    });
    userNameInput.addEventListener("input", () => {
      try {
        localStorage.setItem(USER_NAME_KEY, (userNameInput.value || "").trim());
      } catch (e) {}
    });
  }

  toggleEmptyState();
  renderSidebarHistory();

  const newChatBtn = $("new-chat-btn");
  if (newChatBtn) newChatBtn.addEventListener("click", clearChat);

  const fileInput = $("file-input");
  const fileTrigger = $("file-trigger");

  const selectedFileBar = $("selected-file-bar");
  const selectedFileNameEl = $("selected-file-name");
  const clearFileBtn = $("clear-file-btn");

  if (fileTrigger && fileInput) {
    fileTrigger.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      fileTrigger.classList.toggle("has-file", !!file);
      if (selectedFileBar) selectedFileBar.classList.toggle("visible", !!file);
      if (selectedFileNameEl) selectedFileNameEl.textContent = file ? file.name : "";
    });
  }

  if (clearFileBtn && fileInput) {
    clearFileBtn.addEventListener("click", () => {
      fileInput.value = "";
      fileTrigger.classList.remove("has-file");
      if (selectedFileBar) selectedFileBar.classList.remove("visible");
      if (selectedFileNameEl) selectedFileNameEl.textContent = "";
    });
  }

  $("chat-send").addEventListener("click", sendChat);
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
}

document.addEventListener("DOMContentLoaded", main);
