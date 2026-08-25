import unittest

import pandas as pd

from ml.trading import BUY, HOLD, SELL, build_plan, classify


class TradingTests(unittest.TestCase):
    def test_signal_thresholds(self):
        self.assertEqual(classify(0.02, 0.01), BUY)
        self.assertEqual(classify(-0.02, 0.01), SELL)
        self.assertEqual(classify(0.005, 0.01), HOLD)
        self.assertEqual(classify(0.01, 0.01), HOLD)

    def test_long_levels_and_risk_sizing(self):
        plan = build_plan(
            signal=BUY, price=100.0, pred_return=0.02,
            live=pd.Series({"atr_pct": 0.02}), budget=10_000,
            risk_pct=1.0, atr_mult=1.0, rr=2.0,
        )
        self.assertAlmostEqual(plan["position_value"], 5_000)
        self.assertAlmostEqual(plan["max_loss"], 100)
        self.assertAlmostEqual(plan["stop_loss"], 98)
        self.assertAlmostEqual(plan["take_profit"], 104)

    def test_short_levels_and_disabled_short(self):
        plan = build_plan(
            signal=SELL, price=100.0, pred_return=-0.02,
            live=pd.Series({"atr_pct": 0.02}), budget=10_000,
            risk_pct=1.0, atr_mult=1.0, rr=2.0, allow_short=False,
        )
        self.assertFalse(plan["tradeable"])
        self.assertEqual(plan["position_value"], 0)
        self.assertAlmostEqual(plan["stop_loss"], 102)
        self.assertAlmostEqual(plan["take_profit"], 96)

    def test_position_is_capped_at_budget(self):
        plan = build_plan(
            signal=BUY, price=100.0, pred_return=0.02,
            live=pd.Series({"atr_pct": 0.005}), budget=10_000,
            risk_pct=10.0, atr_mult=1.0, rr=2.0,
        )
        self.assertTrue(plan["capped_by_budget"])
        self.assertEqual(plan["position_value"], 10_000)
        self.assertAlmostEqual(plan["max_loss"], 50)


if __name__ == "__main__":
    unittest.main()
