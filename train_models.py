#!/usr/bin/env python3
"""Train (or retrain) every market model and write the artifacts to models/.

    python train_models.py            # train only what is missing or stale
    python train_models.py --force    # rebuild everything from scratch
    python train_models.py --status   # report without training
"""
from __future__ import annotations

import argparse
import sys
import time

from ml.config import MARKETS
from ml.model import artifact_status, get_model


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the AI Market Predictor models")
    ap.add_argument("--force", action="store_true", help="retrain even if current")
    ap.add_argument("--status", action="store_true", help="only report artifact state")
    ap.add_argument("--market", help="limit to a single market key")
    args = ap.parse_args()

    keys = [args.market] if args.market else list(MARKETS)
    bad = [k for k in keys if k not in MARKETS]
    if bad:
        print(f"unknown market(s): {bad}. valid: {list(MARKETS)}", file=sys.stderr)
        return 2

    if args.status:
        for k in keys:
            st = artifact_status(k)
            print(f"{k:<10} {st['state']:<14} trained_at={st.get('trained_at', '-')}")
        return 0

    failures = 0
    for k in keys:
        label = MARKETS[k]["name"]
        t0 = time.perf_counter()
        try:
            bundle, source = get_model(k, force=args.force)
        except Exception as exc:
            failures += 1
            print(f"FAIL  {k:<10} {label}: {exc}", file=sys.stderr)
            continue
        m = bundle["metrics"]
        dt = time.perf_counter() - t0
        print(
            f"{source:<8} {k:<10} {label:<20} "
            f"train={m['n_train']:>5} test={m['n_test']:>4}  "
            f"R2={m['test_r2']:+.4f}  dir={m['directional_accuracy']*100:5.1f}%  "
            f"grade={m['grade']:<12} ({dt:.2f}s)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
