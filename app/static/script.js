const SAMPLE_MESSAGES = {
  "phishing-bank":
    "URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-verify-login.com",
  "phishing-prize":
    "Congratulations! You won a $1,000 gift card in our lucky draw! Claim immediately before midnight at http://claim-prize-now.biz",
  "legit-meeting":
    "Hi team, please find the updated Q3 project timeline attached. Let me know if you have any feedback before Friday's standup.",
  "legit-otp":
    "Your one-time verification code is 482910. Never share this code with anyone, including bank employees.",
};

const form = document.getElementById("detector-form");
const messageInput = document.getElementById("message-input");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const errorBox = document.getElementById("error-box");
const resultPanel = document.getElementById("result-panel");
const resultTitle = document.getElementById("result-title");
const riskBadge = document.getElementById("risk-badge");
const confidenceValue = document.getElementById("confidence-value");
const probValue = document.getElementById("prob-value");
const modelUsed = document.getElementById("model-used");
const probBar = document.getElementById("prob-bar");
const signalsList = document.getElementById("signals-list");
const cleanedText = document.getElementById("cleaned-text");

document.querySelectorAll(".sample-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.getAttribute("data-sample");
    if (SAMPLE_MESSAGES[key]) {
      messageInput.value = SAMPLE_MESSAGES[key];
      errorBox.classList.add("hidden");
    }
  });
});

clearBtn.addEventListener("click", () => {
  messageInput.value = "";
  errorBox.classList.add("hidden");
  resultPanel.classList.add("hidden");
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultPanel.classList.add("hidden");
}

function renderResult(data) {
  errorBox.classList.add("hidden");
  resultPanel.classList.remove("hidden");

  resultTitle.textContent = data.prediction;
  resultTitle.className = data.is_phishing ? "result-phishing" : "result-legitimate";

  const riskClass = `risk-${data.risk_level.toLowerCase()}`;
  riskBadge.textContent = `Risk: ${data.risk_level}`;
  riskBadge.className = `risk-badge ${riskClass}`;

  confidenceValue.textContent = `${data.confidence}%`;
  probValue.textContent = `${data.phishing_probability}%`;
  modelUsed.textContent = data.model_used || "ML Classifier";

  probBar.style.width = `${Math.min(100, Math.max(2, data.phishing_probability))}%`;
  if (data.phishing_probability >= 75) {
    probBar.style.background = "#ef4444";
  } else if (data.phishing_probability >= 45) {
    probBar.style.background = "#f59e0b";
  } else {
    probBar.style.background = "#10b981";
  }

  signalsList.innerHTML = "";
  if (Array.isArray(data.signals) && data.signals.length > 0) {
    data.signals.forEach((signal) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${signal.title}</strong> <span class="signal-evidence">(${signal.evidence})</span>`;
      signalsList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "No obvious rule-based phishing cues detected.";
    signalsList.appendChild(li);
  }

  cleanedText.textContent = data.cleaned_text || "";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    showError("Please enter a message to analyze.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Failed to classify message.");
      return;
    }
    renderResult(data);
  } catch (err) {
    showError("Unable to reach the prediction server. Please verify the app is running.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Message";
  }
});
