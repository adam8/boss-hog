import { JSDOM } from "jsdom";
import { describe, expect, it, vi } from "vitest";

import {
  BOSS_HOGG_LOADING_QUOTES,
  boot,
  buildQuery,
  pickLoadingQuote,
  renderResults,
} from "../public/app/app.js";

function makeDom() {
  return new JSDOM(
    `<!DOCTYPE html>
      <html>
        <body>
          <form id="controls-form">
            <button class="info-button" type="button" data-info="panel-a">i</button>
            <div id="panel-a" hidden>
              <button class="more-button" type="button" data-more-target="panel-a-more" aria-expanded="false">More</button>
              <div id="panel-a-more" hidden>Deeper panel text</div>
            </div>
            <select name="feature_pack"><option value="price_only" selected>price_only</option></select>
            <input name="max_observations" value="240">
            <input name="initial_window" value="120">
            <input name="random_cells" value="20">
            <input name="seed" value="11">
            <button type="submit">Run</button>
          </form>
          <section id="status-bar"></section>
          <section id="error-banner" hidden></section>
          <section id="results-root"></section>
        </body>
      </html>`,
    { url: "https://example.com/" },
  );
}

const payload = {
  data_status: {
    source: "USDA AMS direct hog avg_net_price",
    purchase_type: "Prod. Sold (All Purchase Types)",
    data_as_of: "2026-03-31",
    refreshed_at: "2026-04-11T16:00:00+00:00",
  },
  metrics: {
    prediction_actual_correlation: 0.42,
    directional_accuracy: 0.74,
    average_ex_ante_fit: 0.06,
    feature_pack: "core_fundamentals",
  },
  final_month: {
    target_month_bucket: "2026-03-01",
    starting_month_bucket: "2026-02-01",
    prediction_date: "2026-03-01",
    predicted_next_month_log_return: 0.0123,
    actual_next_month_log_return: 0.0119,
    starting_month_price_average: 87.33,
    predicted_next_month_price_average: 88.41,
    actual_next_month_price_average: 88.37,
  },
  current_forecast: {
    starting_month_bucket: "2026-03-01",
    target_month_bucket: "2026-04-01",
    starting_month_price_average: 88.37,
    predicted_target_month_log_return: 0.0172,
    predicted_target_month_price_average: 89.91,
    ex_ante_fit: 0.081,
    top_feature_importance: [{ feature: "month_sin", importance: 0.22 }],
  },
  provisional_next_next_forecast: {
    starting_month_bucket: "2026-04-01",
    target_month_bucket: "2026-05-01",
    starting_month_price_average_so_far: 89.12,
    predicted_target_month_log_return: 0.0138,
    predicted_target_month_price_average: 90.36,
    ex_ante_fit: 0.071,
    data_through: "2026-04-14",
    top_feature_importance: [{ feature: "month_cos", importance: 0.19 }],
  },
  average_feature_importance: [{ feature: "loin_depth_avg", importance: 0.21 }],
  average_exogenous_importance: [{ feature: "loin_depth_avg", importance: 0.21 }],
};

function deferred() {
  let resolve;
  const promise = new Promise((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

describe("app ui", () => {
  it("picks a loading quote from the fixed Boss Hogg-inspired list", () => {
    expect(pickLoadingQuote(() => 0)).toBe(BOSS_HOGG_LOADING_QUOTES[0]);
    expect(pickLoadingQuote(() => 0.9999)).toBe(BOSS_HOGG_LOADING_QUOTES[BOSS_HOGG_LOADING_QUOTES.length - 1]);
  });

  it("builds the backtest query string", () => {
    const dom = makeDom();
    const form = dom.window.document.getElementById("controls-form");
    expect(buildQuery(form)).toContain("feature_pack=price_only");
    expect(buildQuery(form)).toContain("max_observations=240");
  });

  it("renders the summary panels", () => {
    const html = renderResults(payload);
    expect(html).toContain("Current Forecast");
    expect(html).toContain("Forecast Target Month");
    expect(html).toContain("April 2026");
    expect(html).toContain("<code>2026-04-01</code>");
    expect(html).toContain("Provisional Following-Month Forecast");
    expect(html).toContain("Provisional Target Month");
    expect(html).toContain("May 2026");
    expect(html).toContain("<code>2026-05-01</code>");
    expect(html).toContain("Final Completed Target Month");
    expect(html).toContain("Average Exogenous Importance");
    expect(html).toContain("Average Loin Depth");
    expect(html).toContain("<code>loin_depth_avg</code>");
    expect(html).toContain('data-info="corr-info"');
    expect(html).toContain("Correlation between predicted and realized next-month log returns");
    expect(html).toContain('data-info="feature-loin-depth-avg-0"');
    expect(html).toContain("another carcass composition measure");
    expect(html).toContain("Target Month (monthly average bucket)");
    expect(html).toContain("March 2026");
    expect(html).toContain("<code>2026-03-01</code>");
  });

  it("toggles the deeper info text with the more button", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));
    await boot(dom.window.document, fetchMock);

    const infoButton = dom.window.document.querySelector("[data-info='panel-a']");
    const panel = dom.window.document.getElementById("panel-a");
    const moreButton = dom.window.document.querySelector("[data-more-target='panel-a-more']");
    const morePanel = dom.window.document.getElementById("panel-a-more");

    infoButton.click();
    expect(panel.hidden).toBe(false);

    moreButton.click();
    expect(morePanel.hidden).toBe(false);
    expect(moreButton.textContent).toBe("Less");

    infoButton.click();
    expect(panel.hidden).toBe(true);
    expect(morePanel.hidden).toBe(true);
    expect(moreButton.textContent).toBe("More");
  });

  it("boots and renders data from the API", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));
    await boot(dom.window.document, fetchMock);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(dom.window.document.getElementById("results-root").innerHTML).toContain("Directional Accuracy");
    expect(dom.window.document.getElementById("status-bar").textContent).toContain("complete");
  });

  it("shows the hog spinner while the backtest is running", async () => {
    const dom = makeDom();
    const pending = deferred();
    const fetchMock = vi.fn(() => pending.promise);

    const bootPromise = boot(dom.window.document, fetchMock);
    await Promise.resolve();

    const statusBar = dom.window.document.getElementById("status-bar");
    expect(statusBar.innerHTML).toContain("status-spinner-logo");
    expect(statusBar.textContent).toContain("Running backtest");
    expect(statusBar.className).toContain("status-bar--loading");
    expect(BOSS_HOGG_LOADING_QUOTES.some((quote) => statusBar.textContent.includes(quote))).toBe(true);

    pending.resolve({
      ok: true,
      json: async () => payload,
    });
    await bootPromise;

    expect(statusBar.textContent).toContain("Backtest complete");
    expect(statusBar.className).not.toContain("status-bar--loading");
  });

  it("binds info buttons inside dynamically rendered metric cards", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    await boot(dom.window.document, fetchMock);

    const resultsRoot = dom.window.document.getElementById("results-root");
    const metricInfoButton = resultsRoot.querySelector("[data-info='corr-info']");
    const metricInfoPanel = resultsRoot.querySelector("#corr-info");

    expect(metricInfoPanel.hidden).toBe(true);

    metricInfoButton.click();
    expect(metricInfoPanel.hidden).toBe(false);
  });

  it("binds info buttons inside the current forecast cards", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    await boot(dom.window.document, fetchMock);

    const resultsRoot = dom.window.document.getElementById("results-root");
    const currentForecastButton = resultsRoot.querySelector("[data-info='current-forecast-info']");
    const currentForecastPanel = resultsRoot.querySelector("#current-forecast-info");

    expect(currentForecastPanel.hidden).toBe(true);

    currentForecastButton.click();
    expect(currentForecastPanel.hidden).toBe(false);
    expect(currentForecastPanel.textContent).toContain("does not forecast May until April itself is complete");
  });

  it("binds info buttons inside the final target month cards", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    await boot(dom.window.document, fetchMock);

    const resultsRoot = dom.window.document.getElementById("results-root");
    const targetMonthButton = resultsRoot.querySelector("[data-info='target-month-info']");
    const targetMonthPanel = resultsRoot.querySelector("#target-month-info");

    expect(targetMonthPanel.hidden).toBe(true);

    targetMonthButton.click();
    expect(targetMonthPanel.hidden).toBe(false);
    expect(targetMonthPanel.textContent).toContain("2026-04-01 means the April 2026 monthly bucket");
  });

  it("binds info buttons inside the provisional forecast cards", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    await boot(dom.window.document, fetchMock);

    const resultsRoot = dom.window.document.getElementById("results-root");
    const provisionalButton = resultsRoot.querySelector("[data-info='provisional-forecast-info']");
    const provisionalPanel = resultsRoot.querySelector("#provisional-forecast-info");

    expect(provisionalPanel.hidden).toBe(true);

    provisionalButton.click();
    expect(provisionalPanel.hidden).toBe(false);
    expect(provisionalPanel.textContent).toContain("uses April-so-far as the starting month and projects May");
  });

  it("binds info buttons for feature rows inside importance lists", async () => {
    const dom = makeDom();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    await boot(dom.window.document, fetchMock);

    const resultsRoot = dom.window.document.getElementById("results-root");
    const featureInfoButton = resultsRoot.querySelector("[data-info='feature-loin-depth-avg-0']");
    const featureInfoPanel = resultsRoot.querySelector("#feature-loin-depth-avg-0");

    expect(featureInfoPanel.hidden).toBe(true);

    featureInfoButton.click();
    expect(featureInfoPanel.hidden).toBe(false);
    expect(featureInfoPanel.textContent).toContain("carcass composition");
  });
});
