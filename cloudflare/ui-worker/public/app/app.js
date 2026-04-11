function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function importanceList(items) {
  if (!items.length) {
    return "<p>No importance values were returned for this run.</p>";
  }
  return `<ol class="importance-list">${items
    .map(
      (item) =>
        `<li><strong>${escapeHtml(item.feature)}</strong><span>${Number(item.importance).toFixed(4)}</span></li>`,
    )
    .join("")}</ol>`;
}

export function buildQuery(form) {
  const params = new URLSearchParams();
  const FormDataCtor = form.ownerDocument?.defaultView?.FormData ?? FormData;
  const entries = new FormDataCtor(form);
  for (const [key, value] of entries.entries()) {
    params.set(key, String(value));
  }
  return params.toString();
}

export function renderResults(payload) {
  const metrics = payload.metrics;
  const finalMonth = payload.final_month;
  const dataStatus = payload.data_status;

  return `
    <div class="results-grid">
      <article class="metric-card">
        <div class="metric-label">Prediction / Actual Correlation</div>
        <div class="metric-value">${Number(metrics.prediction_actual_correlation).toFixed(3)}</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">Directional Accuracy</div>
        <div class="metric-value">${Number(metrics.directional_accuracy).toFixed(3)}</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">Average Ex-Ante Fit</div>
        <div class="metric-value">${Number(metrics.average_ex_ante_fit).toFixed(3)}</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">Feature Pack</div>
        <div class="metric-value metric-value--small">${escapeHtml(metrics.feature_pack)}</div>
      </article>
    </div>

    <div class="panel-stack">
      <section class="panel">
        <div class="panel-heading">
          <h2>Data Status</h2>
        </div>
        <div class="detail-grid">
          <div><span>Source</span><strong>${escapeHtml(dataStatus.source)}</strong></div>
          <div><span>Purchase Type</span><strong>${escapeHtml(dataStatus.purchase_type)}</strong></div>
          <div><span>Data As Of</span><strong>${escapeHtml(dataStatus.data_as_of)}</strong></div>
          <div><span>Refreshed At</span><strong>${escapeHtml(dataStatus.refreshed_at)}</strong></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <h2>Final Out-of-Sample Month</h2>
        </div>
        <div class="detail-grid">
          <div><span>Predicted Month</span><strong>${escapeHtml(finalMonth.prediction_date)}</strong></div>
          <div><span>Predicted Next-Month Log Return</span><strong>${Number(finalMonth.predicted_next_month_log_return).toFixed(4)}</strong></div>
          <div><span>Actual Next-Month Log Return</span><strong>${Number(finalMonth.actual_next_month_log_return).toFixed(4)}</strong></div>
          <div><span>Starting Month Price Average</span><strong>${Number(finalMonth.starting_month_price_average).toFixed(2)}</strong></div>
          <div><span>Predicted Next Month Price Average</span><strong>${Number(finalMonth.predicted_next_month_price_average).toFixed(2)}</strong></div>
          <div><span>Actual Next Month Price Average</span><strong>${Number(finalMonth.actual_next_month_price_average).toFixed(2)}</strong></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <h2>Average Feature Importance</h2>
        </div>
        ${importanceList(payload.average_feature_importance)}
      </section>

      ${
        payload.average_exogenous_importance.length
          ? `<section class="panel">
               <div class="panel-heading">
                 <h2>Average Exogenous Importance</h2>
               </div>
               ${importanceList(payload.average_exogenous_importance)}
             </section>`
          : ""
      }
    </div>
  `;
}

export function wireInfoButtons(root) {
  for (const button of root.querySelectorAll("[data-info]")) {
    button.addEventListener("click", () => {
      const panel = root.getElementById(button.dataset.info);
      if (!panel) {
        return;
      }
      panel.hidden = !panel.hidden;
    });
  }
}

export async function runBacktest({ form, statusBar, errorBanner, resultsRoot, submitButton, fetchImpl = fetch }) {
  const query = buildQuery(form);
  statusBar.textContent = "Running backtest...";
  errorBanner.hidden = true;
  submitButton.disabled = true;
  try {
    const response = await fetchImpl(`/api/backtest?${query}`, {
      headers: { accept: "application/json" },
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Backtest failed.");
    }
    resultsRoot.innerHTML = renderResults(payload);
    statusBar.textContent = "Backtest complete.";
  } catch (error) {
    errorBanner.hidden = false;
    errorBanner.textContent = error instanceof Error ? error.message : String(error);
    statusBar.textContent = "Backtest failed.";
  } finally {
    submitButton.disabled = false;
  }
}

export function boot(doc = document, fetchImpl = fetch) {
  const form = doc.getElementById("controls-form");
  const statusBar = doc.getElementById("status-bar");
  const errorBanner = doc.getElementById("error-banner");
  const resultsRoot = doc.getElementById("results-root");
  const submitButton = form.querySelector("button[type='submit']");

  wireInfoButtons(doc);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runBacktest({ form, statusBar, errorBanner, resultsRoot, submitButton, fetchImpl });
  });
  return runBacktest({ form, statusBar, errorBanner, resultsRoot, submitButton, fetchImpl });
}

if (typeof window !== "undefined" && window.document) {
  void boot(window.document, window.fetch.bind(window));
}
