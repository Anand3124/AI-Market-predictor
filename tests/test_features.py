import unittest

import numpy as np
import pandas as pd

from ml.features import TARGET, build_features, split_supervised


def sample_prices(rows=100):
    close = pd.Series(np.linspace(100.0, 150.0, rows))
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows),
        "Open": close - 0.25,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.linspace(1_000, 2_000, rows),
    })


class FeatureTests(unittest.TestCase):
    def test_target_is_exactly_next_bar_return(self):
        data = sample_prices()
        frame, _ = build_features(data)
        expected = data.loc[1, "Close"] / data.loc[0, "Close"] - 1.0
        self.assertAlmostEqual(frame.loc[0, TARGET], expected)
        self.assertTrue(np.isnan(frame.iloc[-1][TARGET]))

    def test_features_do_not_change_when_only_future_prices_change(self):
        original = sample_prices()
        changed = original.copy()
        changed.loc[81:, "Close"] *= 3
        changed.loc[81:, ["Open", "High", "Low"]] *= 3
        before, names = build_features(original)
        after, _ = build_features(changed)
        pd.testing.assert_series_equal(before.loc[80, names], after.loc[80, names])

    def test_live_row_is_excluded_from_supervised_data(self):
        frame, names = build_features(sample_prices())
        supervised, live = split_supervised(frame, names)
        self.assertLess(supervised["Date"].max(), live.iloc[0]["Date"])
        self.assertTrue(live[TARGET].isna().all())


if __name__ == "__main__":
    unittest.main()
