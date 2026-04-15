from __future__ import annotations

import unittest

from hog_price_baseline import BacktestSummary
from hog_ui import CORE_FUNDAMENTALS_FEATURE_PACK, PRICE_ONLY_FEATURE_PACK, UIState, render_page


class HogUITests(unittest.TestCase):
    def test_ui_state_parses_defaults_and_checkbox(self) -> None:
        state = UIState.from_query("feature_pack=core_fundamentals&max_observations=300&force_download=on")

        self.assertEqual(state.feature_pack, CORE_FUNDAMENTALS_FEATURE_PACK)
        self.assertEqual(state.max_observations, 300)
        self.assertTrue(state.force_download)
        self.assertEqual(state.initial_window, 120)

    def test_render_page_includes_info_buttons_and_summary(self) -> None:
        summary = BacktestSummary(
            series_name="Prod. Sold (All Purchase Types)",
            feature_pack=PRICE_ONLY_FEATURE_PACK,
            feature_names=["ret_1m", "month_sin"],
            source_path=None,  # type: ignore[arg-type]
            observation_count=240,
            train_window=120,
            prediction_dates=["2026-03-01"],
            predictions=[0.05],
            actuals=[0.04],
            implied_next_prices=[92.02],
            current_prices=[87.33],
            next_prices=[90.89],
            correlation=0.404,
            directional_accuracy=0.664,
            average_fit=0.056,
            average_variable_importance={"month_sin": 0.18, "ret_1m": 0.12},
            average_exogenous_variable_importance={},
        )

        html = render_page(UIState(), summary=summary, error=None)

        self.assertIn("Monthly RBP Hog Explorer", html)
        self.assertIn("info-button", html)
        self.assertIn("more-button", html)
        self.assertIn("Feature Pack", html)
        self.assertIn("price_only", html)
        self.assertIn("Average Feature Importance", html)
        self.assertIn("month_sin", html)


if __name__ == "__main__":
    unittest.main()
