const runBtn = document.getElementById("runBtn");
const refreshLatestBtn = document.getElementById("refreshLatestBtn");
const statusEl = document.getElementById("status");
const processLogEl = document.getElementById("processLog");
const finalReportEl = document.getElementById("finalReport");
const cycleStatusEl = document.getElementById("cycleStatus");
const latestSummaryEl = document.getElementById("latestSummary");
const chartGridEl = document.getElementById("chartGrid");
const chartModalEl = document.getElementById("chartModal");
const chartModalImageEl = document.getElementById("chartModalImage");
const closeChartModalBtn = document.getElementById("closeChartModal");

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderReport(text) {
  if (!text || !text.trim()) {
    finalReportEl.textContent = "No report generated yet.";
    return;
  }
  const blocks = text
    .split(/\n\n+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((chunk) => `<p>${escapeHtml(chunk).replaceAll("\n", "<br>")}</p>`);
  finalReportEl.innerHTML = blocks.join("");
}

function renderCycleStatus(results) {
  if (!results) {
    cycleStatusEl.textContent = "No terminal cycle results file found yet.";
    return;
  }

  const lines = [
    `before_test: ${results.before_test_status || "n/a"}`,
    `train: ${results.train_status || "n/a"}`,
    `after_test: ${results.after_test_status || "n/a"}`,
    `kickoff: ${results.kickoff_status || "n/a"}`,
    `user_feedback: ${results.user_feedback || "n/a"}`,
    `training_file: ${results.training_file_path || "n/a"}`,
    `report_file: ${results.report_path || "n/a"}`,
  ];
  cycleStatusEl.textContent = lines.join("\n");
}

function openChartModal(url, alt) {
  chartModalImageEl.src = url;
  chartModalImageEl.alt = alt || "Expanded chart";
  chartModalEl.classList.remove("hidden");
}

function closeChartModal() {
  chartModalEl.classList.add("hidden");
  chartModalImageEl.src = "";
}

function renderChartSection(title, urls) {
  const section = document.createElement("section");
  section.className = "chart-section";

  const header = document.createElement("h4");
  header.className = "chart-section-title";
  header.textContent = title;
  section.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "chart-subgrid";

  for (const chartUrl of urls) {
    const card = document.createElement("figure");
    card.className = "chart-card";

    const img = document.createElement("img");
    img.src = chartUrl;
    img.alt = chartUrl.split("/").pop() || "chart";
    img.loading = "lazy";
    img.addEventListener("click", () => openChartModal(chartUrl, img.alt));

    const caption = document.createElement("figcaption");
    caption.textContent = `${img.alt} (click to zoom)`;

    card.appendChild(img);
    card.appendChild(caption);
    grid.appendChild(card);
  }

  section.appendChild(grid);
  return section;
}

function renderCharts(chartUrls) {
  chartGridEl.innerHTML = "";
  if (!chartUrls || chartUrls.length === 0) {
    chartGridEl.innerHTML = "<p>No saved charts yet. Run terminal train/test first.</p>";
    return;
  }

  const beforeUrls = chartUrls.filter((u) => u.split("/").pop()?.startsWith("before_"));
  const afterUrls = chartUrls.filter((u) => u.split("/").pop()?.startsWith("after_"));
  const otherUrls = chartUrls.filter((u) => {
    const name = u.split("/").pop() || "";
    return !name.startsWith("before_") && !name.startsWith("after_");
  });

  if (beforeUrls.length > 0) {
    chartGridEl.appendChild(renderChartSection("Before Training", beforeUrls));
  }
  if (afterUrls.length > 0) {
    chartGridEl.appendChild(renderChartSection("After Training", afterUrls));
  }
  if (otherUrls.length > 0) {
    chartGridEl.appendChild(renderChartSection("Charts", otherUrls));
  }
}

async function loadLatestResults() {
  try {
    const response = await fetch("/results/latest");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderCycleStatus(data.results);
    latestSummaryEl.textContent = data.summary_text || "No saved summary yet.";
    renderCharts(data.chart_urls || []);
  } catch (error) {
    cycleStatusEl.textContent = `Failed to load latest results: ${error}`;
    latestSummaryEl.textContent = "Failed to load saved summary.";
    chartGridEl.innerHTML = "<p>Failed to load charts.</p>";
  }
}

async function runCycle() {
  const sampleSize = Number(document.getElementById("sampleSize").value || 5);

  runBtn.disabled = true;
  statusEl.textContent = "Running kickoff...";

  try {
    const response = await fetch("/run/kickoff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_size: sampleSize }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    processLogEl.textContent = data.process_log || "No logs captured.";
    renderReport(data.final_report || "");
    statusEl.textContent = data.kickoff_status || "Kickoff finished.";
    await loadLatestResults();
  } catch (error) {
    statusEl.textContent = `Run failed: ${error}`;
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runCycle);
refreshLatestBtn.addEventListener("click", loadLatestResults);
closeChartModalBtn.addEventListener("click", closeChartModal);
chartModalEl.addEventListener("click", (event) => {
  if (event.target === chartModalEl) {
    closeChartModal();
  }
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeChartModal();
  }
});
loadLatestResults();
