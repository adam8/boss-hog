from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from hog_backtest_service import BacktestRequest, run_request_against_monthly_series
from hog_price_baseline import (
    CORE_FUNDAMENTALS_FEATURE_PACK,
    DEFAULT_CACHE_PATH,
    DEFAULT_PURCHASE_TYPE,
    PRICE_ONLY_FEATURE_PACK,
    BacktestSummary,
    aggregate_monthly_average,
    download_direct_hog_history,
    load_cached_daily_series,
)


@dataclass(frozen=True)
class UIState:
    feature_pack: str = PRICE_ONLY_FEATURE_PACK
    max_observations: int = 240
    initial_window: int = 120
    random_cells: int = 20
    seed: int = 11
    force_download: bool = False

    @classmethod
    def from_query(cls, query: str) -> "UIState":
        params = parse_qs(query)
        return cls(
            feature_pack=_read_choice(params, "feature_pack", {PRICE_ONLY_FEATURE_PACK, CORE_FUNDAMENTALS_FEATURE_PACK}, PRICE_ONLY_FEATURE_PACK),
            max_observations=_read_int(params, "max_observations", 240, minimum=14),
            initial_window=_read_int(params, "initial_window", 120, minimum=2),
            random_cells=_read_int(params, "random_cells", 20, minimum=1),
            seed=_read_int(params, "seed", 11, minimum=0),
            force_download="force_download" in params,
        )


def run_ui_backtest(state: UIState) -> BacktestSummary:
    csv_path = download_direct_hog_history(DEFAULT_CACHE_PATH, purchase_type=DEFAULT_PURCHASE_TYPE, force=state.force_download)
    daily_series = load_cached_daily_series(csv_path)
    monthly_series = aggregate_monthly_average(daily_series)
    return run_request_against_monthly_series(
        monthly_series,
        BacktestRequest(
            feature_pack=state.feature_pack,
            max_observations=state.max_observations,
            initial_window=state.initial_window,
            random_cells=state.random_cells,
            seed=state.seed,
        ),
        series_name=DEFAULT_PURCHASE_TYPE,
        source_path=str(csv_path),
    )


def render_page(state: UIState, *, summary: BacktestSummary | None, error: str | None = None) -> str:
    metrics_html = ""
    importances_html = ""
    if summary is not None:
        last_index = len(summary.predictions) - 1
        metrics_html = f"""
        <section class="results-grid">
          <article class="metric-card">
            <div class="metric-label">Prediction / Actual Correlation {_info_button("corr-info")}</div>
            <div class="metric-value">{summary.correlation:.3f}</div>
            {_info_panel("corr-info", "Correlation between predicted and realized next-month log returns over the out-of-sample window.")}
          </article>
          <article class="metric-card">
            <div class="metric-label">Directional Accuracy {_info_button("direction-info")}</div>
            <div class="metric-value">{summary.directional_accuracy:.3f}</div>
            {_info_panel("direction-info", "Share of out-of-sample months where the sign of the predicted move matched the sign of the realized move.")}
          </article>
          <article class="metric-card">
            <div class="metric-label">Average Ex-Ante Fit {_info_button("fit-info")}</div>
            <div class="metric-value">{summary.average_fit:.3f}</div>
            {_info_panel("fit-info", "RBP's internal reliability score averaged across the rolling predictions. Higher is better, but it is not the same thing as realized forecast accuracy.")}
          </article>
          <article class="metric-card">
            <div class="metric-label">Feature Pack {_info_button("pack-info")}</div>
            <div class="metric-value metric-value--small">{escape(summary.feature_pack)}</div>
            {_info_panel("pack-info", "The set of inputs included in each monthly RBP feature row. `price_only` uses only price-derived signals; `core_fundamentals` also adds same-report hog fundamentals.")}
          </article>
        </section>

        <section class="panel">
          <div class="panel-heading">
            <h2>Final Out-of-Sample Month</h2>
            {_info_button("final-info")}
          </div>
          {_info_panel("final-info", "The most recent backtest month in the selected window. The prediction was produced using only prior months in the rolling training sample. Returns are log returns, and prices are monthly averages.")}
          <div class="detail-grid">
            <div><span>Predicted Month</span><strong>{escape(summary.prediction_dates[last_index])}</strong></div>
            <div><span>Predicted Next-Month Log Return</span><strong>{summary.predictions[last_index]:.4f}</strong></div>
            <div><span>Actual Next-Month Log Return</span><strong>{summary.actuals[last_index]:.4f}</strong></div>
            <div><span>Starting Month Price Average</span><strong>{summary.current_prices[last_index]:.2f}</strong></div>
            <div><span>Predicted Next Month Price Average</span><strong>{summary.implied_next_prices[last_index]:.2f}</strong></div>
            <div><span>Actual Next Month Price Average</span><strong>{summary.next_prices[last_index]:.2f}</strong></div>
          </div>
        </section>
        """
        importances_html = f"""
        <section class="panel">
          <div class="panel-heading">
            <h2>Average Feature Importance</h2>
            {_info_button("importance-info")}
          </div>
          {_info_panel("importance-info", "Average variable importance across the rolling predictions. Higher values mean the feature tended to improve adjusted fit more often across the sampled RBP cells.")}
          {_importance_list(summary.average_variable_importance)}
        </section>
        """
        if summary.average_exogenous_variable_importance:
            importances_html += f"""
            <section class="panel">
              <div class="panel-heading">
                <h2>Average Exogenous Importance</h2>
                {_info_button("exo-info")}
              </div>
              {_info_panel("exo-info", "The same averaged importance view, filtered to the extra same-report hog fundamentals only.")}
              {_importance_list(summary.average_exogenous_variable_importance)}
            </section>
            """

    error_html = (
        f'<section class="error-banner"><strong>Run failed.</strong> {escape(error)}</section>'
        if error
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Boss Hog RBP UI</title>
    <style>
      :root {{
        --bg: #f5f1e8;
        --panel: #fffdf8;
        --ink: #2b241d;
        --muted: #6d6357;
        --line: #d9cbb8;
        --accent: #994c35;
        --accent-strong: #6f2b17;
        --accent-soft: #efe0d8;
        --success: #1f5a3c;
        --shadow: 0 16px 40px rgba(54, 37, 23, 0.08);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(153, 76, 53, 0.12), transparent 26rem),
          linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
      }}

      .page {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 32px 20px 56px;
      }}

      .hero {{
        display: grid;
        gap: 12px;
        margin-bottom: 28px;
      }}

      .eyebrow {{
        color: var(--accent);
        font-size: 2rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 700;
      }}

      h1, h2 {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        font-weight: 700;
      }}

      h1 {{
        font-size: clamp(2rem, 5vw, 3.6rem);
        line-height: 1.02;
        max-width: 10ch;
      }}

      .hero p {{
        margin: 0;
        max-width: 72ch;
        color: var(--muted);
        line-height: 1.6;
      }}

      .layout {{
        display: grid;
        grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
        gap: 20px;
      }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 20px;
      }}

      .panel + .panel {{
        margin-top: 18px;
      }}

      .panel-heading {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
      }}

      .controls {{
        position: sticky;
        top: 18px;
        align-self: start;
      }}

      form {{
        display: grid;
        gap: 14px;
      }}

      label {{
        display: grid;
        gap: 8px;
        font-size: 0.94rem;
        font-weight: 700;
      }}

      .label-row {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }}

      input,
      select {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        background: #fff;
        color: var(--ink);
        font: inherit;
      }}

      input:focus,
      select:focus,
      button:focus {{
        outline: 2px solid rgba(153, 76, 53, 0.28);
        outline-offset: 2px;
      }}

      .checkbox-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding-top: 4px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      .checkbox-row input {{
        width: auto;
      }}

      .cta {{
        border: 0;
        border-radius: 999px;
        padding: 13px 18px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        color: #fff;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
      }}

      .info-button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.3rem;
        height: 1.3rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-size: 0.78rem;
        font-weight: 700;
        cursor: pointer;
      }}

      .info-panel {{
        display: none;
        margin: 8px 0 0;
        padding: 12px 14px;
        border-radius: 12px;
        background: #f7efe9;
        color: var(--muted);
        line-height: 1.55;
      }}

      .info-panel.open {{
        display: block;
      }}

      .results-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 18px;
      }}

      .metric-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 18px;
      }}

      .metric-label {{
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      .metric-value {{
        margin-top: 10px;
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        font-weight: 800;
      }}

      .metric-value--small {{
        font-size: 1.2rem;
        line-height: 1.3;
      }}

      .detail-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
      }}

      .detail-grid div {{
        border-top: 1px solid var(--line);
        padding-top: 12px;
      }}

      .detail-grid span {{
        display: block;
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 6px;
      }}

      .detail-grid strong {{
        font-size: 1.04rem;
      }}

      .importance-list {{
        display: grid;
        gap: 10px;
      }}

      .importance-row {{
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        align-items: center;
        padding-bottom: 10px;
        border-bottom: 1px solid #efe7da;
      }}

      .importance-name {{
        font-weight: 700;
      }}

      .importance-value {{
        color: var(--success);
        font-variant-numeric: tabular-nums;
      }}

      .error-banner {{
        margin-bottom: 18px;
        border: 1px solid #d59f8d;
        background: #fff1eb;
        color: #7d2d14;
        border-radius: 16px;
        padding: 14px 16px;
      }}

      .footnote {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
      }}

      @media (max-width: 900px) {{
        .layout {{
          grid-template-columns: 1fr;
        }}

        .controls {{
          position: static;
        }}

        .results-grid,
        .detail-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div class="eyebrow">Boss Hog</div>
        <h1>Monthly RBP Hog Explorer</h1>
      </section>

      {error_html}

      <section class="layout">
        <aside class="controls panel">
          <div class="panel-heading">
            <h2>Controls</h2>
            {_info_button("controls-info")}
          </div>
          {_info_panel("controls-info", "These settings drive the monthly backtest shown on the right. The UI uses the same AMS data and RBP code as the command-line script.")}
          <form method="get">
            <label>
              <span class="label-row">Feature Pack {_info_button("feature-pack-info")}</span>
              <select name="feature_pack">
                {_option(PRICE_ONLY_FEATURE_PACK, state.feature_pack)}
                {_option(CORE_FUNDAMENTALS_FEATURE_PACK, state.feature_pack)}
              </select>
              {_info_panel("feature-pack-info", "`price_only` uses only price-derived features. `core_fundamentals` adds the same-report hog fundamentals such as head count, live weight, carcass weight, and lean percent.")}
            </label>

            <label>
              <span class="label-row">Monthly Observations {_info_button("observations-info")}</span>
              <input type="number" min="14" name="max_observations" value="{state.max_observations}">
              {_info_panel("observations-info", "How many of the most recent monthly observations to keep before building the supervised dataset. Lower values run faster; higher values use more history.")}
            </label>

            <label>
              <span class="label-row">Initial Window {_info_button("window-info")}</span>
              <input type="number" min="2" name="initial_window" value="{state.initial_window}">
              {_info_panel("window-info", "How many supervised rows RBP sees before making the first out-of-sample prediction. This controls how much history is in each rolling fit.")}
            </label>

            <label>
              <span class="label-row">Random Cells {_info_button("cells-info")}</span>
              <input type="number" min="1" name="random_cells" value="{state.random_cells}">
              {_info_panel("cells-info", "How many extra sparse-grid cells RBP samples on top of its base cells. More cells can capture more structure, but they increase run time.")}
            </label>

            <label>
              <span class="label-row">Seed {_info_button("seed-info")}</span>
              <input type="number" min="0" name="seed" value="{state.seed}">
              {_info_panel("seed-info", "Controls the random sparse-grid sampling so runs are repeatable.")}
            </label>

            <label class="checkbox-row">
              <input type="checkbox" name="force_download"{" checked" if state.force_download else ""}>
              Refresh AMS cache before running
            </label>

            <button class="cta" type="submit">Run Backtest</button>
          </form>

          <p class="footnote">
            Source: USDA AMS direct hog report for <strong>{escape(DEFAULT_PURCHASE_TYPE)}</strong>.
            The UI reuses the same rolling RBP implementation as the CLI script.
          </p>
        </aside>

        <section>
          {metrics_html}
          {importances_html}
        </section>
      </section>
    </main>

    <script>
      document.querySelectorAll(".info-button").forEach((button) => {{
        button.addEventListener("click", () => {{
          const targetId = button.getAttribute("data-target");
          const panel = document.getElementById(targetId);
          if (!panel) return;
          const isOpen = panel.classList.toggle("open");
          button.setAttribute("aria-expanded", isOpen ? "true" : "false");
        }});
      }});
    </script>
  </body>
</html>
"""


class HogUIHandler(BaseHTTPRequestHandler):
    server_version = "BossHogUI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_request(include_body=False)

    def _handle_request(self, *, include_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(404, "Not found")
            return

        state = UIState.from_query(parsed.query)
        summary: BacktestSummary | None = None
        error: str | None = None
        try:
            summary = run_ui_backtest(state)
        except Exception as exc:  # pragma: no cover - exercised manually
            error = str(exc)

        body = render_page(state, summary=summary, error=error).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers may cancel or replace in-flight requests while the backtest is running.
            # Treat client disconnects as a normal no-op instead of emitting noisy tracebacks.
            return

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local hog RBP UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), HogUIHandler)
    print(f"Hog UI available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _option(value: str, selected: str) -> str:
    return f'<option value="{escape(value)}"{" selected" if value == selected else ""}>{escape(value)}</option>'


def _info_button(target_id: str) -> str:
    return (
        f'<button class="info-button" type="button" data-target="{escape(target_id)}" '
        f'aria-expanded="false" aria-controls="{escape(target_id)}">i</button>'
    )


def _info_panel(panel_id: str, body: str) -> str:
    return f'<div class="info-panel" id="{escape(panel_id)}">{escape(body)}</div>'


def _importance_list(importances: dict[str, float], *, count: int = 7) -> str:
    rows = []
    for feature_name, importance in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:count]:
        rows.append(
            "<div class=\"importance-row\">"
            f"<div class=\"importance-name\">{escape(feature_name)}</div>"
            f"<div class=\"importance-value\">{importance:.4f}</div>"
            "</div>"
        )
    return f'<div class="importance-list">{"".join(rows)}</div>'


def _read_choice(params: dict[str, list[str]], key: str, allowed: set[str], default: str) -> str:
    raw = params.get(key, [default])[0]
    return raw if raw in allowed else default


def _read_int(params: dict[str, list[str]], key: str, default: int, *, minimum: int) -> int:
    raw = params.get(key, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


if __name__ == "__main__":
    main()
