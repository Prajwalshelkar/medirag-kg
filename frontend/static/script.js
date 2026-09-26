const API_URL = "http://localhost:8000";

// Query Tab Elements
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-btn");
const askButtonText = askButton.querySelector(".ask-button__text");
const askButtonSpinner = askButton.querySelector(".ask-button__spinner");

// Upload Tab Elements
const uploadImageInput = document.getElementById("upload-image");
const uploadTextInput = document.getElementById("upload-text");
const uploadButton = document.getElementById("upload-btn");
const uploadButtonText = uploadButton.querySelector(".ask-button__text");
const uploadButtonSpinner = uploadButton.querySelector(".ask-button__spinner");

// Shared Output Elements
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const evidenceSection = document.getElementById("evidence-section");
const evidenceList = document.getElementById("evidence-list");
const errorSection = document.getElementById("error-section");
const errorText = document.getElementById("error-text");
const idlePlaceholder = document.getElementById("idle-placeholder");

function setQueryLoading(isLoading) {
  askButton.disabled = isLoading;
  askButtonSpinner.hidden = !isLoading;
  askButtonText.hidden = isLoading;
}

function setUploadLoading(isLoading) {
  uploadButton.disabled = isLoading;
  uploadButtonSpinner.hidden = !isLoading;
  uploadButtonText.hidden = isLoading;
}

function hideAllOutputs() {
  answerSection.hidden = true;
  evidenceSection.hidden = true;
  errorSection.hidden = true;
}

function formatMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

function renderFindingLine(type, labelText, iconClass, values) {
  const line = document.createElement("div");
  line.className = `finding-line finding-line--${type}`;

  const pills = values
    .map((val) => `<span class="finding-line__values">${val}</span>`)
    .join(", ");

  line.innerHTML = `
    <span class="finding-line__marker">
      <i class="${iconClass}"></i> ${labelText}:
    </span>
    ${pills}
  `;
  return line;
}

function renderEvidence(reports) {
  evidenceList.innerHTML = "";

  if (!reports || reports.length === 0) {
    evidenceSection.hidden = true;
    if (idlePlaceholder) idlePlaceholder.hidden = false;
    return;
  }

  reports.forEach((report) => {
    const item = document.createElement("div");
    item.className = "evidence-item";

    const meta = document.createElement("div");
    meta.className = "evidence-item__meta";
    meta.innerHTML = `
      <i class="fa-solid fa-file-medical"></i>
      <span>report-${report.uid}</span>
      <span style="opacity: 0.4;">|</span>
      <span>dist: ${typeof report.distance === "number" ? report.distance.toFixed(3) : report.distance}</span>
    `;
    item.appendChild(meta);

    const text = document.createElement("p");
    text.className = "evidence-item__text";
    text.textContent = report.text_preview;
    item.appendChild(text);

    if (report.present_findings && report.present_findings.length) {
      item.appendChild(
        renderFindingLine("confirmed", "Confirmed", "fa-solid fa-circle-check", report.present_findings)
      );
    }

    if (report.negated_findings && report.negated_findings.length) {
      item.appendChild(
        renderFindingLine("ruled-out", "Ruled Out", "fa-solid fa-ban", report.negated_findings)
      );
    }

    evidenceList.appendChild(item);
  });

  if (idlePlaceholder) idlePlaceholder.hidden = true;
  evidenceSection.hidden = false;
}

// ---- Query Execution ----
async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  hideAllOutputs();
  setQueryLoading(true);

  try {
    const response = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 3 }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned status ${response.status}`);
    }

    const data = await response.json();

    answerText.innerHTML = formatMarkdown(data.answer);
    answerSection.hidden = false;

    renderEvidence(data.retrieved_reports);
  } catch (err) {
    if (idlePlaceholder) idlePlaceholder.hidden = false;
    errorText.textContent =
      "Could not reach backend API. Ensure Uvicorn is active: " +
      "uvicorn backend.backend_main:app --reload --port 8000";
    errorSection.hidden = false;
    console.error(err);
  } finally {
    setQueryLoading(false);
  }
}

// ---- Multimodal Upload Execution ----
async function uploadReport() {
  const file = uploadImageInput.files[0];
  const reportText = uploadTextInput.value.trim();

  if (!file || !reportText) {
    errorText.textContent = "Please provide both an X-ray image and the report findings text.";
    errorSection.hidden = false;
    return;
  }

  hideAllOutputs();
  setUploadLoading(true);

  const formData = new FormData();
  formData.append("image", file);
  formData.append("report_text", reportText);

  try {
    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Backend returned status ${response.status}`);
    }

    const data = await response.json();

    answerText.innerHTML = formatMarkdown(data.conclusion || "Report successfully analyzed.");
    answerSection.hidden = false;

    renderEvidence([
      {
        uid: data.uid,
        distance: 0,
        text_preview: reportText.slice(0, 300),
        present_findings: data.present_findings || [],
        negated_findings: data.negated_findings || [],
      },
    ]);
  } catch (err) {
    if (idlePlaceholder) idlePlaceholder.hidden = false;
    errorText.textContent =
      "Upload failed. Ensure backend has the /upload route defined: " + err.message;
    errorSection.hidden = false;
    console.error(err);
  } finally {
    setUploadLoading(false);
  }
}

// ---- Recent Reports Loader ----
async function loadRecentReports() {
  const recentList = document.getElementById("recent-list");

  try {
    const response = await fetch(`${API_URL}/recent`);
    const data = await response.json();

    if (!data.recent || data.recent.length === 0) {
      recentList.innerHTML = '<p class="empty-state-text">No reports studied yet this session.</p>';
      return;
    }

    recentList.innerHTML = "";
    data.recent.forEach((entry) => {
      const item = document.createElement("div");
      item.className = "evidence-item";

      const meta = document.createElement("div");
      meta.className = "evidence-item__meta";
      const time = new Date(entry.timestamp).toLocaleTimeString();
      meta.innerHTML = `
        <i class="fa-solid fa-file-medical"></i>
        <span>${entry.uid}</span>
        <span class="recent-time"><i class="fa-regular fa-clock"></i> ${time}</span>
      `;
      item.appendChild(meta);

      if (entry.present_findings && entry.present_findings.length) {
        item.appendChild(renderFindingLine("confirmed", "Confirmed", "fa-solid fa-circle-check", entry.present_findings));
      }
      if (entry.negated_findings && entry.negated_findings.length) {
        item.appendChild(renderFindingLine("ruled-out", "Ruled Out", "fa-solid fa-ban", entry.negated_findings));
      }

      recentList.appendChild(item);
    });
  } catch (err) {
    recentList.innerHTML = '<p class="empty-state-text">Could not load recent reports from backend.</p>';
    console.error(err);
  }
}

// ---- Tabs Routing ----
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = {
  ask: document.getElementById("tab-ask"),
  upload: document.getElementById("tab-upload"),
  recent: document.getElementById("tab-recent"),
};

tabButtons.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabButtons.forEach((t) => t.classList.remove("tab-btn--active"));
    tab.classList.add("tab-btn--active");

    Object.values(tabPanels).forEach((panel) => {
      if (panel) panel.hidden = true;
    });

    const activePanel = tabPanels[tab.dataset.tab];
    if (activePanel) activePanel.hidden = false;

    hideAllOutputs();
    if (idlePlaceholder) idlePlaceholder.hidden = false;

    if (tab.dataset.tab === "recent") {
      loadRecentReports();
    }
  });
});

// Preset helpers
function setQuery(text) {
  questionInput.value = text;
}

function runPreset(text) {
  // Switch to 'ask' tab if currently on upload/recent
  const askTab = document.querySelector('[data-tab="ask"]');
  if (askTab && !askTab.classList.contains("tab-btn--active")) {
    askTab.click();
  }
  questionInput.value = text;
  askQuestion();
}

askButton.addEventListener("click", askQuestion);
uploadButton.addEventListener("click", uploadReport);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    askQuestion();
  }
});

// Theme switch
const themeToggle = document.getElementById("theme-toggle");
const body = document.body;
const themeIcon = themeToggle.querySelector("i");

const savedTheme = localStorage.getItem("medirag-theme");
if (savedTheme === "light") {
  body.classList.add("light-mode");
  themeIcon.className = "fa-solid fa-moon";
}

themeToggle.addEventListener("click", () => {
  body.classList.toggle("light-mode");
  if (body.classList.contains("light-mode")) {
    themeIcon.className = "fa-solid fa-moon";
    localStorage.setItem("medirag-theme", "light");
  } else {
    themeIcon.className = "fa-solid fa-sun";
    localStorage.setItem("medirag-theme", "dark");
  }
});