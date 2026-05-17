// ============ Sidebar toggle ============
const app = document.querySelector(".app");
const toggleBtn = document.getElementById("sidebar-toggle");
const openBtn = document.getElementById("sidebar-open");

function toggleSidebar() {
  app.classList.toggle("sidebar-collapsed");
}
toggleBtn.addEventListener("click", toggleSidebar);
openBtn.addEventListener("click", toggleSidebar);

// ============ Metric tabs ============
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const split = tab.dataset.split;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll("[data-split-panel]").forEach((p) => {
      p.classList.toggle("hidden", p.dataset.splitPanel !== split);
    });
  });
});

// ============ Form submission ============
const form = document.getElementById("predict-form");
const resultCard = document.getElementById("result");
const resultLabel = document.getElementById("result-label");
const resultBadge = document.getElementById("result-badge");
const resultProb = document.getElementById("result-prob");
const meterFill = document.getElementById("meter-fill");
const resultConfidence = document.getElementById("result-confidence");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });

  // Cast numerics
  ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"].forEach((k) => {
    if (data[k] !== undefined) data[k] = Number(data[k]);
  });

  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Prediction failed");

    renderResult(body);
  } catch (err) {
    resultCard.classList.remove("hidden");
    resultLabel.textContent = "Error";
    resultBadge.textContent = "Failed";
    resultBadge.className = "result-badge high";
    resultProb.textContent = "—";
    meterFill.style.width = "0%";
    resultConfidence.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = 'Predict Risk <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>';
  }
});

function renderResult(body) {
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const isHigh = body.prediction === 1;
  resultLabel.textContent = body.label;
  resultBadge.textContent = isHigh ? "High Risk" : "Low Risk";
  resultBadge.className = "result-badge " + (isHigh ? "high" : "low");

  if (body.probability_default !== null && body.probability_default !== undefined) {
    const pct = (body.probability_default * 100).toFixed(2);
    resultProb.textContent = pct + "%";
    meterFill.style.width = pct + "%";
  } else {
    resultProb.textContent = "n/a";
    meterFill.style.width = isHigh ? "100%" : "0%";
  }
  resultConfidence.textContent = body.confidence !== null
    ? `Model confidence: ${body.confidence}%`
    : "";
}
