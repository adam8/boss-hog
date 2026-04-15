function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const FEATURE_METADATA = {
  ret_1m: {
    label: "1-Month Momentum",
    summary: "The log return from the prior month to the current month.",
    more: "In simple terms, this is the freshest one-month momentum signal. It tells the model whether price was just rising or falling right before the forecast month.",
  },
  ret_3m: {
    label: "3-Month Momentum",
    summary: "The total log return over the last 3 months.",
    more: "This is a short-term momentum view that smooths out one noisy month. It helps the model see whether the recent quarter has been generally strong or weak.",
  },
  ret_6m: {
    label: "6-Month Momentum",
    summary: "The total log return over the last 6 months.",
    more: "This is a medium-term momentum signal. It gives the model a broader price trend than the shorter 1-month and 3-month views.",
  },
  ret_12m: {
    label: "12-Month Momentum",
    summary: "The total log return over the last 12 months.",
    more: "This is the one-year momentum view. It helps the model compare the current month with longer seasonal and trend patterns in hog prices.",
  },
  ma_gap_3m: {
    label: "Price vs 3-Month Average",
    summary: "How far the current price sits above or below the 3-month average price.",
    more: "In simple terms, this asks whether the market is stretched relative to its recent level. A positive gap means price is above its short moving average; a negative gap means it is below.",
  },
  ma_gap_12m: {
    label: "Price vs 12-Month Average",
    summary: "How far the current price sits above or below the 12-month average price.",
    more: "This is the same idea as the 3-month gap, but against the last year instead of the last quarter. It gives a broader view of whether price is rich or weak versus its longer history.",
  },
  vol_3m: {
    label: "3-Month Volatility",
    summary: "The sample volatility of monthly returns over the last 3 months.",
    more: "This measures how choppy the market has been very recently. Higher values mean price has been moving around more from month to month.",
  },
  vol_12m: {
    label: "12-Month Volatility",
    summary: "The sample volatility of monthly returns over the last 12 months.",
    more: "This is the longer-run version of recent volatility. It helps the model tell calm market periods apart from more unstable ones.",
  },
  month_sin: {
    label: "Seasonality (Sine)",
    summary: "A sine-based encoding of the calendar month, used to represent seasonality on a smooth yearly cycle.",
    more: "Months repeat in a circle, not on a straight line. `month_sin` and `month_cos` work together so December and January are treated as close neighbors instead of opposite ends of a numeric scale.",
  },
  month_cos: {
    label: "Seasonality (Cosine)",
    summary: "A cosine-based encoding of the calendar month, paired with `month_sin` to represent yearly seasonality.",
    more: "On its own this is not very intuitive, but together with `month_sin` it gives the model a clean circular map of the year. That lets it learn seasonal behavior without treating month numbers as a straight line.",
  },
  head_count_avg: {
    label: "Average Head Count",
    summary: "The average daily head count in the month.",
    more: "In simple terms, this is a rough quantity signal for how many hogs were moving through the report. It gives the model some supply context beyond price alone.",
  },
  live_weight_avg: {
    label: "Average Live Weight",
    summary: "The average live weight reported during the month.",
    more: "This tells the model how heavy the hogs were before processing. It can help distinguish different market conditions that may not show up in price alone.",
  },
  carcass_weight_avg: {
    label: "Average Carcass Weight",
    summary: "The average carcass weight reported during the month.",
    more: "This is a processed-weight measure after slaughter. It gives another physical-market context signal that can differ from live weight.",
  },
  sort_loss_avg: {
    label: "Average Sort Loss",
    summary: "The average sort loss reported during the month.",
    more: "Sort loss reflects deductions or adjustments tied to how hogs fit preferred specifications. It can capture quality or composition effects in the reported market.",
  },
  backfat_avg: {
    label: "Average Backfat",
    summary: "The average backfat measure reported during the month.",
    more: "This is one of the carcass composition signals in the USDA report. It helps describe what type of hogs were in the market, not just what price they fetched.",
  },
  loin_depth_avg: {
    label: "Average Loin Depth",
    summary: "The average loin depth reported during the month.",
    more: "Loin depth is another carcass composition measure. In simple terms, it helps describe hog quality and physical characteristics that may matter for how relevant past months really were.",
  },
  lean_percent_avg: {
    label: "Average Lean Percent",
    summary: "The average lean percent reported during the month.",
    more: "This is the reported share of lean meat. It gives the model a simple quality mix signal from the same USDA report used for price.",
  },
};

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "-")
    .replaceAll(/^-+|-+$/g, "");
}

function getFeatureMetadata(featureName) {
  const metadata = FEATURE_METADATA[featureName];
  if (metadata) {
    return metadata;
  }
  return {
    label: featureName,
    summary: `Model feature \`${featureName}\`.`,
    more: "This feature does not yet have a custom plain-English description in the UI.",
  };
}

function importanceList(items, listIdPrefix) {
  if (!items.length) {
    return "<p>No importance values were returned for this run.</p>";
  }
  return `<ol class="importance-list">${items
    .map(
      (item, index) => {
        const featureName = String(item.feature);
        const metadata = getFeatureMetadata(featureName);
        const infoId = `${listIdPrefix}-${slugify(featureName)}-${index}`;
        return `
          <li>
            <div class="importance-item-top">
              <div class="importance-item-label">
                <div class="importance-item-copy">
                  <strong>${escapeHtml(metadata.label)}</strong>
                  <div class="importance-code"><code>${escapeHtml(featureName)}</code></div>
                </div>
                <button class="info-button" type="button" data-info="${escapeHtml(infoId)}" aria-expanded="false">i</button>
              </div>
              <span>${Number(item.importance).toFixed(4)}</span>
            </div>
            ${renderInfoPanel(infoId, metadata.summary, metadata.more)}
          </li>
        `;
      },
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

function formatMonthBucket(bucketDate) {
  const [year, month] = String(bucketDate).split("-");
  const monthIndex = Number(month) - 1;
  const monthNames = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  if (!year || !monthNames[monthIndex]) {
    return {
      label: escapeHtml(bucketDate),
      code: escapeHtml(bucketDate),
    };
  }
  return {
    label: `${monthNames[monthIndex]} ${year}`,
    code: String(bucketDate),
  };
}

function previousMonthBucket(bucketDate) {
  const [yearRaw, monthRaw] = String(bucketDate).split("-");
  const year = Number(yearRaw);
  const month = Number(monthRaw);
  if (!Number.isInteger(year) || !Number.isInteger(month) || month < 1 || month > 12) {
    return String(bucketDate);
  }
  if (month === 1) {
    return `${year - 1}-12-01`;
  }
  return `${year}-${String(month - 1).padStart(2, "0")}-01`;
}

function renderDetailCard({ label, value, code = "", infoId, summary, more }) {
  return `
    <div class="detail-card">
      <div class="detail-label-row">
        <span>${escapeHtml(label)}</span>
        <button class="info-button" type="button" data-info="${escapeHtml(infoId)}" aria-expanded="false">i</button>
      </div>
      <strong>${value}</strong>
      ${code ? `<div class="detail-code"><code>${escapeHtml(code)}</code></div>` : ""}
      ${renderInfoPanel(infoId, summary, more)}
    </div>
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
  const targetMonthBucket = finalMonth.target_month_bucket || finalMonth.prediction_date;
  const startingMonthBucket = finalMonth.starting_month_bucket || previousMonthBucket(targetMonthBucket);
  const targetMonth = formatMonthBucket(targetMonthBucket);
  const startingMonth = formatMonthBucket(startingMonthBucket);

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
          <h2>Final Completed Target Month</h2>
          <button class="info-button" type="button" data-info="final-target-info" aria-expanded="false">i</button>
        </div>
        ${renderInfoPanel(
          "final-target-info",
          "This section shows the latest fully completed target month included in the rolling backtest.",
          "The model uses one month to predict the next month's average. If the current calendar month is still in progress, that unfinished month is excluded so the final example stays comparable and complete.",
        )}
        <div class="detail-grid">
          ${renderDetailCard({
            label: "Target Month (monthly average bucket)",
            value: escapeHtml(targetMonth.label),
            code: targetMonth.code,
            infoId: "target-month-info",
            summary: "The month being forecast, labeled with the first day of that month.",
            more: "For example, 2026-04-01 means the April 2026 monthly bucket. It is not a point forecast made on April 1, and it is not a forecast for May.",
          })}
          ${renderDetailCard({
            label: "Starting Month Used for Forecast",
            value: escapeHtml(startingMonth.label),
            code: startingMonth.code,
            infoId: "starting-month-info",
            summary: "The completed month whose information was used to forecast the target month.",
            more: "In simple terms, the model looks at what the market looked like in this starting month, then predicts the average price move into the next month.",
          })}
          ${renderDetailCard({
            label: "Predicted Target-Month Log Return",
            value: Number(finalMonth.predicted_next_month_log_return).toFixed(4),
            infoId: "predicted-return-info",
            summary: "The model's forecasted log return from the starting month average into the target month average.",
            more: "This is not a plain percentage, but for small moves it is close. A positive value means the model expected the target month average price to be higher than the starting month average price.",
          })}
          ${renderDetailCard({
            label: "Actual Target-Month Log Return",
            value: Number(finalMonth.actual_next_month_log_return).toFixed(4),
            infoId: "actual-return-info",
            summary: "The realized log return from the starting month average into the target month average.",
            more: "This is what actually happened after the target month finished. Comparing this with the predicted log return shows whether the model got direction and size roughly right.",
          })}
          ${renderDetailCard({
            label: "Starting Month Average Price",
            value: Number(finalMonth.starting_month_price_average).toFixed(2),
            infoId: "starting-price-info",
            summary: "The monthly average price in the starting month used as the forecast base.",
            more: "Think of this as the base price level the forecast starts from before projecting into the target month.",
          })}
          ${renderDetailCard({
            label: "Predicted Target-Month Average Price",
            value: Number(finalMonth.predicted_next_month_price_average).toFixed(2),
            infoId: "predicted-price-info",
            summary: "The target month average price implied by the model's predicted log return.",
            more: "This translates the return forecast back into a dollar price so it is easier to compare with the realized target-month average.",
          })}
          ${renderDetailCard({
            label: "Actual Target-Month Average Price",
            value: Number(finalMonth.actual_next_month_price_average).toFixed(2),
            infoId: "actual-price-info",
            summary: "The realized average price for the completed target month.",
            more: "This is the actual monthly average once the target month finished. It is the clean comparison point for the model's predicted target-month average price.",
          })}
        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <h2>Average Feature Importance</h2>
        </div>
        ${importanceList(payload.average_feature_importance, "feature")}
      </section>

      ${
        payload.average_exogenous_importance.length
          ? `<section class="panel">
               <div class="panel-heading">
                 <h2>Average Exogenous Importance</h2>
               </div>
               ${importanceList(payload.average_exogenous_importance, "exogenous")}
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
