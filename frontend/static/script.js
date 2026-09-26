const API_URL = "http://localhost:8000";

const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-btn");
const askButtonText = askButton.querySelector(".ask-button__text");
const askButtonSpinner = askButton.querySelector(".ask-button__spinner");

const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const evidenceSection = document.getElementById("evidence-section");
const evidenceList = document.getElementById("evidence-list");
const errorSection = document.getElementById("error-section");
const errorText = document.getElementById("error-text");
const idlePlaceholder = document.getElementById("idle-placeholder");

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  askButtonSpinner.hidden = !isLoading;
  askButtonText.hidden = isLoading;
}

function hideAll() {
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

    // Meta Header with Distance & UID
    const meta = document.createElement("div");
    meta.className = "evidence-item__meta";
    meta.innerHTML = `
      <i class="fa-solid fa-file-medical"></i>
      <span>report-${report.uid}</span>
      <span style="opacity: 0.4;">|</span>
      <span>dist: ${report.distance.toFixed(3)}</span>
    `;
    item.appendChild(meta);

    // Context Raw Text Preview
    const text = document.createElement("p");
    text.className = "evidence-item__text";
    text.textContent = report.text_preview;
    item.appendChild(text);

    // Present / Confirmed Findings
    if (report.present_findings && report.present_findings.length) {
      item.appendChild(
        renderFindingLine(
          "confirmed",
          "Confirmed",
          "fa-solid fa-circle-check",
          report.present_findings
        )
      );
    }

    // Negated / Ruled-Out Findings
    if (report.negated_findings && report.negated_findings.length) {
      item.appendChild(
        renderFindingLine(
          "ruled-out",
          "Ruled Out",
          "fa-solid fa-ban",
          report.negated_findings
        )
      );
    }

    evidenceList.appendChild(item);
  });

  if (idlePlaceholder) idlePlaceholder.hidden = true;
  evidenceSection.hidden = false;
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  hideAll();
  setLoading(true);

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
    setLoading(false);
  }
}

// Preset helpers
function setQuery(text) {
  questionInput.value = text;
}

function runPreset(text) {
  questionInput.value = text;
  askQuestion();
}

askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    askQuestion();
  }
});

// Theme Toggle
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