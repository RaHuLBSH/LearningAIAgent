(() => {
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendButton");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const stepLabel = document.getElementById("stepLabel");
  const topicsListEl = document.getElementById("topicsList");

  let sessionId = window.localStorage.getItem("agentic_session_id") || null;
  let isSending = false;

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  // Minimal, safe markdown rendering: bold + line breaks + simple bullets.
  function renderMiniMarkdown(md) {
    let html = escapeHtml(md || "");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // bullets: lines starting with "- "
    const lines = html.split("\n");
    let inList = false;
    const out = [];
    for (const line of lines) {
      if (line.trim().startsWith("- ")) {
        if (!inList) {
          inList = true;
          out.push("<ul>");
        }
        out.push("<li>" + line.trim().slice(2) + "</li>");
      } else {
        if (inList) {
          inList = false;
          out.push("</ul>");
        }
        out.push(line);
      }
    }
    if (inList) out.push("</ul>");
    html = out.join("\n").replaceAll("\n", "<br/>");
    return html;
  }

  function parseLearningCard(text) {
    const lines = (text || "").split("\n");
    const getBlock = (label) => {
      const idx = lines.findIndex((l) => {
        const t = l.trim();
        return (
          t.startsWith(label + ":") ||
          t.startsWith("**" + label + "**") ||
          t.startsWith("**" + label + ":**") ||
          t.toUpperCase() === label
        );
      });
      if (idx === -1) return null;
      const start = idx + 1;
      const nextIdx = lines
        .slice(start)
        .findIndex((l) => /^[A-Z_]+:/.test(l.trim()) || /^\*\*[A-Z_]+/.test(l.trim()) || l.trim() === "END");
      const end = nextIdx === -1 ? lines.length : start + nextIdx;
      const rawHeader = lines[idx].trim();
      const headerValue = rawHeader.includes(":") ? rawHeader.split(":").slice(1).join(":").trim() : "";
      const body = lines.slice(start, end).join("\n").trim();
      return { headerValue, body };
    };

    const title = getBlock("TITLE")?.headerValue || "";
    const summary = getBlock("SUMMARY")?.headerValue || getBlock("SUMMARY")?.body || "";
    const keyPoints = getBlock("KEY_POINTS")?.body || "";
    const example = getBlock("EXAMPLE")?.body || "";
    const mistakes = getBlock("COMMON_MISTAKES")?.body || "";

    // If the format doesn't match, return null so we render plain text.
    if (!title && !summary && !keyPoints && !example) return null;

    return { title, summary, keyPoints, example, mistakes };
  }

  function createActionButtons() {
    const row = document.createElement("div");
    row.className = "action-buttons";

    const mk = (label, textToSend) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "action-btn";
      b.textContent = label;
      b.addEventListener("click", () => {
        inputEl.value = textToSend;
        sendMessage();
      });
      return b;
    };

    row.appendChild(mk("Learn more", "Tell me more"));
    row.appendChild(mk("Explain simpler", "Explain simpler"));
    row.appendChild(mk("Ask me", "Ask me"));
    return row;
  }

  function appendMessage(role, content, options = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "user" : "ai"}`;

    const bubble = document.createElement("div");
    const kind = options.kind || "message"; // teaching|quiz|evaluation|system
    bubble.className = `bubble ${role === "user" ? "user" : "ai"} ${kind}`;
    // Render interactive learning card for teaching messages
    if (role !== "user" && kind === "teaching") {
      const card = parseLearningCard(content);
      if (card) {
        bubble.classList.add("learning-card");
        bubble.textContent = "";

        const titleEl = document.createElement("div");
        titleEl.className = "card-title";
        titleEl.textContent = card.title || "Learning Card";
        bubble.appendChild(titleEl);

        const summaryEl = document.createElement("div");
        summaryEl.className = "card-summary";
        summaryEl.innerHTML = renderMiniMarkdown(card.summary);
        bubble.appendChild(summaryEl);

        const mkDetails = (label, body) => {
          if (!body) return null;
          const details = document.createElement("details");
          details.className = "card-details";
          const summary = document.createElement("summary");
          summary.textContent = label;
          const pre = document.createElement("pre");
          pre.className = "card-pre";
          pre.innerHTML = renderMiniMarkdown(body);
          details.appendChild(summary);
          details.appendChild(pre);
          return details;
        };

        const kp = mkDetails("Key points", card.keyPoints);
        const ex = mkDetails("Example", card.example);
        const cm = mkDetails("Common mistakes", card.mistakes);
        if (kp) bubble.appendChild(kp);
        if (ex) bubble.appendChild(ex);
        if (cm) bubble.appendChild(cm);
      } else {
        bubble.innerHTML = renderMiniMarkdown(content);
      }
    } else {
      // Render mini-markdown for all AI messages (quiz/evaluation/system)
      if (role !== "user") {
        bubble.innerHTML = renderMiniMarkdown(content);
      } else {
        bubble.textContent = content;
      }
    }

    wrapper.appendChild(bubble);

    // Meta action buttons under teaching
    if (role !== "user" && kind === "teaching") {
      wrapper.appendChild(createActionButtons());
    }

    // MCQ quick buttons
    if (role !== "user" && options.quizType === "mcq") {
      const btnRow = document.createElement("div");
      btnRow.className = "mcq-buttons";
      ["A", "B", "C", "D"].forEach((letter) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "mcq-btn";
        b.textContent = letter;
        b.addEventListener("click", () => {
          inputEl.value = letter;
          sendMessage();
        });
        btnRow.appendChild(b);
      });
      wrapper.appendChild(btnRow);
      // no meta buttons under quiz (per requirement)
    }

    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setLoading(loading) {
    isSending = loading;
    sendBtn.disabled = loading;
    sendBtn.textContent = loading ? "Thinking..." : "Send";
  }

  async function sendMessage() {
    if (isSending) return;
    const text = (inputEl.value || "").trim();
    if (!text) return;

    appendMessage("user", text);
    inputEl.value = "";
    setLoading(true);

    try {
      const payload = {
        message: text,
        session_id: sessionId,
        user_id: "demo-user", // simple stable id for long-term memory POC
      };

      const resp = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();
      sessionId = data.session_id;
      window.localStorage.setItem("agentic_session_id", sessionId);

      if (Array.isArray(data.events) && data.events.length > 0) {
        data.events.forEach((ev) => {
          const kind =
            ev.type === "teaching"
              ? "teaching"
              : ev.type === "quiz"
              ? "quiz"
              : ev.type === "evaluation"
              ? "evaluation"
              : "system";
          appendMessage("ai", ev.content || "", { kind, quizType: ev.quiz_type });
        });
      } else if (data.reply) {
        appendMessage("ai", data.reply);
      }

      const pct = Math.round(data.progress || 0);
      progressFill.style.width = `${pct}%`;
      progressText.textContent = `${pct}%`;

      stepLabel.textContent = `Step: ${data.current_step || "unknown"}`;

      if (topicsListEl && Array.isArray(data.topic_progress)) {
        topicsListEl.textContent = "";
        data.topic_progress.forEach((t) => {
          const row = document.createElement("div");
          row.className = "topic-row";
          const name = document.createElement("div");
          name.className = "topic-name";
          name.textContent = t.topic || "(unknown)";
          const pct = document.createElement("div");
          pct.className = "topic-pct";
          pct.textContent = `${Math.round(t.progress || 0)}%`;
          row.appendChild(name);
          row.appendChild(pct);
          topicsListEl.appendChild(row);
        });
      }
    } catch (err) {
      console.error(err);
      appendMessage("ai", "Oops, something went wrong talking to the backend.");
    } finally {
      setLoading(false);
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // initial welcome message
  appendMessage(
    "ai",
    'Hi! I am your agentic tutor. Start with something like "Teach me Dynamic Programming".'
  );
})();
