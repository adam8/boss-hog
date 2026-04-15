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

function renderInfoPanel(id, summary, more) {
  return `
    <div class="info-panel metric-info-panel" id="${escapeHtml(id)}" hidden>
      <p class="info-summary">${escapeHtml(summary)}</p>
      ${
        more
          ? `<button class="more-button" type="button" data-more-target="${escapeHtml(id)}-more" aria-expanded="false">More</button>
             <div class="info-more" id="${escapeHtml(id)}-more" hidden>
               <p>${escapeHtml(more)}</p>
             </div>`
          : ""
      }
    </div>
  `;
}

function renderMetricCard({ label, value, valueClass = "", infoId, summary, more }) {
  const metricValueClass = valueClass ? `metric-value ${valueClass}` : "metric-value";
  return `
    <article class="metric-card">
      <div class="metric-label">
        ${escapeHtml(label)}
        <button class="info-button" type="button" data-info="${escapeHtml(infoId)}" aria-expanded="false">i</button>
      </div>
      <div class="${metricValueClass}">${value}</div>
      ${renderInfoPanel(infoId, summary, more)}
    </article>
  `;
}

function renderLoadingStatus(message) {
  return `
    <span class="status-content">
      <span class="status-spinner" aria-hidden="true">
        <img class="status-spinner-logo" src="/app/logo.png" alt="">
      </span>
      <span>${escapeHtml(message)}</span>
    </span>
  `;
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
      ${renderMetricCard({
        label: "Prediction / Actual Correlation",
        value: Number(metrics.prediction_actual_correlation).toFixed(3),
        infoId: "corr-info",
        summary: "Correlation between predicted and realized next-month log returns over the out-of-sample window.",
        more: "In simple terms, this asks whether the model tended to move in the same direction as reality over time. A higher positive value means the forecasts tracked the actual pattern better, while a value near zero means they were not lining up much at all.",
      })}
      ${renderMetricCard({
        label: "Directional Accuracy",
        value: Number(metrics.directional_accuracy).toFixed(3),
        infoId: "direction-info",
        summary: "Share of out-of-sample months where the sign of the predicted move matched the sign of the realized move.",
        more: "This ignores the exact size of the forecast and asks a simpler yes-or-no question: did the model get up versus down right? It is useful when direction matters more than perfect magnitude.",
      })}
      ${renderMetricCard({
        label: "Average Ex-Ante Fit",
        value: Number(metrics.average_ex_ante_fit).toFixed(3),
        infoId: "fit-info",
        summary: "RBP's internal reliability score averaged across the rolling predictions. Higher is better, but it is not the same thing as realized forecast accuracy.",
        more: "This is the model's own estimate of how trustworthy each prediction looked before the future happened. It is useful as a confidence signal, but a confident forecast can still turn out wrong.",
      })}
      ${renderMetricCard({
        label: "Feature Pack",
        value: escapeHtml(metrics.feature_pack),
        valueClass: "metric-value--small",
        infoId: "pack-info",
        summary: "The set of inputs included in each monthly RBP feature row. price_only uses only price-derived signals; core_fundamentals also adds same-report hog fundamentals.",
        more: "This setting changes what the model is allowed to consider when it searches for relevant past months. The broader pack adds more economic context from the same USDA report, not a different model.",
      })}
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
      const panel = findPanel(root, button.dataset.info);
      if (!panel) {
        return;
      }
      const isOpen = !panel.hidden;
      panel.hidden = isOpen;
      if (isOpen) {
        for (const moreButton of panel.querySelectorAll("[data-more-target]")) {
          const morePanel = root.getElementById(moreButton.dataset.moreTarget);
          if (morePanel) {
            morePanel.hidden = true;
          }
          moreButton.textContent = "More";
          moreButton.setAttribute("aria-expanded", "false");
        }
      }
    });
  }

  for (const button of root.querySelectorAll("[data-more-target]")) {
    button.addEventListener("click", () => {
      const panel = findPanel(root, button.dataset.moreTarget);
      if (!panel) {
        return;
      }
      const willOpen = panel.hidden;
      panel.hidden = !willOpen;
      button.textContent = willOpen ? "Less" : "More";
      button.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });
  }
}

function findPanel(root, id) {
  if (!id) {
    return null;
  }
  if (typeof root.getElementById === "function") {
    return root.getElementById(id);
  }
  return root.querySelector(`#${id}`);
}

export async function runBacktest({ form, statusBar, errorBanner, resultsRoot, submitButton, fetchImpl = fetch }) {
  const query = buildQuery(form);
  statusBar.innerHTML = renderLoadingStatus("Running backtest...");
  statusBar.classList.add("status-bar--loading");
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
    wireInfoButtons(resultsRoot);
    statusBar.textContent = "Backtest complete.";
    statusBar.classList.remove("status-bar--loading");
  } catch (error) {
    errorBanner.hidden = false;
    errorBanner.textContent = error instanceof Error ? error.message : String(error);
    statusBar.textContent = "Backtest failed.";
    statusBar.classList.remove("status-bar--loading");
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
