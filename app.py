"""AI Market Predictor - Streamlit front end.

Run with:  streamlit run app.py

Models are trained once per market and cached on disk (models/*.joblib). Opening
or refreshing this page loads the saved model; it retrains only when the CSV
changes or when you press Retrain Models.
"""
from __future__ import annotations

import os
import sys
import time

# Make the project importable no matter how the app is launched (streamlit run,
# python -m streamlit, an IDE run button, or the test harness).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from ml.config import MARKETS, MODEL_DIR, market_path
from ml.data import file_fingerprint
from ml.model import artifact_status, get_model
from ml.predict import predict
from ui import charts
from ui.theme import CSS, GRADE_STYLE, SIGNAL_STYLE, TEXT_MUTED

st.set_page_config(
    page_title="AI Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# Hover stays on; the modebar toolbar is hidden to keep the dashboard clean.
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------
# model access - the caching layer that keeps refreshes instant
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle(key: str, fingerprint: str):
    """Cached per (market, data fingerprint).

    `fingerprint` is part of the cache key on purpose: edit the CSV and this
    entry is invalidated automatically, which is the only condition (besides
    the Retrain button) under which training runs again.
    """
    bundle, source = get_model(key)
    return bundle, source


def current_fingerprint(key: str) -> str:
    return file_fingerprint(market_path(key))


def money(x: float) -> str:
    return f"${x:,.2f}"


def price_fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def card(k: str, v: str, s: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    sub = f'<div class="s">{s}</div>' if s else ""
    return f'<div class="card"><div class="k">{k}</div><div class="v"{style}>{v}</div>{sub}</div>'


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.05rem;font-weight:680;letter-spacing:-0.01em;">'
        '📈 AI Market Predictor</div>'
        '<div style="color:#8b8a80;font-size:0.76rem;margin:2px 0 16px;">'
        'Linear Regression · next-day forecast</div>',
        unsafe_allow_html=True,
    )

    keys = list(MARKETS)
    market_key = st.selectbox(
        "Market",
        keys,
        format_func=lambda k: f"{MARKETS[k]['icon']}  {MARKETS[k]['name']}",
    )
    st.caption(MARKETS[market_key]["blurb"])

    st.markdown("---")
    # A form submits every planning input as one consistent snapshot. Without
    # it, editing a slider after Predict could mix an old plan with new labels.
    with st.form("prediction_inputs"):
        budget = st.number_input(
            "Budget (USD)", min_value=100.0, max_value=100_000_000.0,
            value=10_000.0, step=500.0, format="%.2f",
            help="Total capital available for this position.",
        )
        risk_pct = st.slider(
            "Risk per trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            help="How much of the budget you accept losing if the stop loss is hit.",
        )
        with st.expander("Advanced settings"):
            atr_mult = st.slider(
                "Stop distance (x volatility)", 0.5, 4.0, 1.5, 0.1,
                help="Stop = this multiple of recent ATR / volatility.",
            )
            rr = st.slider(
                "Reward : risk ratio", 1.0, 5.0, 2.0, 0.5,
                help="Take profit sits this many stop-distances away from entry.",
            )
            allow_short = st.checkbox(
                "Allow short positions on SELL", value=True,
                help="Uncheck if SELL should mean exit / stay out.",
            )
        predict_clicked = st.form_submit_button(
            "⚡  Predict", type="primary", width="stretch"
        )
        st.caption("Educational forecast only — not financial advice.")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;'
        'color:#8b8a80;font-weight:650;margin-bottom:8px;">Saved models</div>',
        unsafe_allow_html=True,
    )
    STATE_TEXT = {
        "current": ("✅", "up to date"),
        "stale-data": ("⚠️", "dataset changed"),
        "stale-schema": ("⚠️", "code changed"),
        "missing": ("⭕", "not trained"),
        "unreadable": ("❌", "unreadable"),
    }
    stale_any = False
    for k in keys:
        st_info = artifact_status(k)
        icon, text = STATE_TEXT.get(st_info["state"], ("❔", st_info["state"]))
        if st_info["state"] != "current":
            stale_any = True
        when = (st_info.get("trained_at") or "")[:10]
        st.markdown(
            f'<div style="font-size:0.76rem;color:#c3c2b7;padding:2px 0;">'
            f'{icon} {MARKETS[k]["name"]} '
            f'<span style="color:#8b8a80;">— {text}{" · " + when if when else ""}</span></div>',
            unsafe_allow_html=True,
        )

    if stale_any:
        st.caption("Stale models retrain automatically the next time you open them.")

    model_dir_writable = os.path.isdir(MODEL_DIR) and os.access(MODEL_DIR, os.W_OK)
    retrain_clicked = st.button(
        "🔄  Retrain Models", width="stretch", disabled=not model_dir_writable,
        help=(None if model_dir_writable else
              "Model files are read-only on this deployment. Use train_models.py locally."),
    )
    st.caption("Rebuilds all 5 models from the CSVs and overwrites the saved files.")


# --------------------------------------------------------------------------
# retrain handler
# --------------------------------------------------------------------------
if retrain_clicked:
    with st.status("Retraining all models…", expanded=True) as status:
        rows = []
        failed = 0
        for k in MARKETS:
            st.write(f"Training **{MARKETS[k]['name']}**…")
            t0 = time.perf_counter()
            try:
                bundle, _ = get_model(k, force=True)
            except Exception as exc:
                failed += 1
                st.write(f"❌ {MARKETS[k]['name']}: {exc}")
                continue
            m = bundle["metrics"]
            rows.append(
                {
                    "Market": MARKETS[k]["name"],
                    "Train rows": m["n_train"],
                    "Test rows": m["n_test"],
                    "Test R²": round(m["test_r2"], 4),
                    "Direction acc.": f"{m['directional_accuracy']*100:.1f}%",
                    "Verdict": m["grade"],
                    "Seconds": round(time.perf_counter() - t0, 2),
                }
            )
        load_bundle.clear()
        st.session_state.pop("result", None)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        status.update(
            label=f"Retrained {len(rows)} / {len(MARKETS)} models"
            + (f" · {failed} failed" if failed else ""),
            state="error" if failed else "complete",
            expanded=bool(failed),
        )
    if failed:
        st.toast(f"Retraining finished with {failed} failure(s).", icon="⚠️")
    else:
        st.toast("All models retrained and saved to models/", icon="✅")


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
meta = MARKETS[market_key]
st.markdown(
    f'<div class="hero"><h1>{meta["icon"]}  {meta["name"]} '
    f'<span style="color:#8b8a80;font-weight:400;font-size:1rem;">'
    f'{meta["symbol"]}</span></h1>'
    f'<p>Next-session forecast from a Linear Regression model trained on daily '
    f'candles — momentum, moving averages, RSI, volatility and volume.</p></div>',
    unsafe_allow_html=True,
)

# Load (or build) the model for the selected market.
try:
    t0 = time.perf_counter()
    bundle, source = load_bundle(market_key, current_fingerprint(market_key))
    load_secs = time.perf_counter() - t0
except FileNotFoundError as exc:
    st.error(f"Dataset not found: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not prepare a model for {meta['name']}: {exc}")
    st.stop()

metrics = bundle["metrics"]
report = bundle["data_report"]

if predict_clicked:
    st.session_state["result"] = predict(
        bundle, budget=budget, risk_pct=risk_pct, atr_mult=atr_mult, rr=rr,
        allow_short=allow_short,
    )
    st.session_state["result_key"] = market_key

result = st.session_state.get("result")
if result is not None and st.session_state.get("result_key") != market_key:
    result = None  # market switched - the old prediction no longer applies


# --------------------------------------------------------------------------
# idle state
# --------------------------------------------------------------------------
if result is None:
    c1, c2, c3, c4 = st.columns(4)
    last_close = float(bundle["live_row"]["Close"].iloc[0])
    last_date = pd.Timestamp(bundle["live_row"]["Date"].iloc[0])
    c1.markdown(card("Latest close", price_fmt(last_close),
                     last_date.strftime("%d %b %Y")), unsafe_allow_html=True)
    c2.markdown(card("History", f"{report['rows_clean']:,} days",
                     f"{report['start']} → {report['end']}"), unsafe_allow_html=True)
    c3.markdown(card("Features", str(len(bundle["feature_names"])),
                     "technical indicators"), unsafe_allow_html=True)
    c4.markdown(card("Model", metrics["grade"],
                     f"{'loaded from disk' if source == 'cache' else 'trained just now'}"
                     f" · {load_secs*1000:.0f} ms",
                     GRADE_STYLE.get(metrics["grade"])), unsafe_allow_html=True)

    st.markdown('<div class="sec">Price history</div>', unsafe_allow_html=True)
    st.plotly_chart(charts.price_history(bundle["history"], meta["unit"]),
                    config=PLOTLY_CONFIG, key="idle_price")
    st.info("Set your budget and risk in the sidebar, then press **⚡ Predict**.")
    st.stop()


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
sig = result["signal"]
style = SIGNAL_STYLE[sig]
plan = result["plan"]
grade = metrics["grade"]
grade_color = GRADE_STYLE.get(grade, TEXT_MUTED)

left, right = st.columns([2.05, 1])
with left:
    action = {
        "BUY": "The model expects the next session to close higher.",
        "SELL": "The model expects the next session to close lower.",
        "HOLD": "The expected move is too small to justify a position.",
    }[sig]
    st.markdown(
        f'<div class="signal-card">'
        f'<div class="bar" style="background:{style["color"]}"></div>'
        f'<div class="signal-word" style="color:{style["color"]}">'
        f'{style["icon"]} {style["word"]}</div>'
        f'<div class="signal-sub">{action}</div>'
        f'<div style="display:flex;gap:34px;margin-top:20px;flex-wrap:wrap;">'
        f'<div><div class="k">Predicted return</div>'
        f'<div style="font-size:1.5rem;font-weight:650;color:{style["color"]};'
        f'font-variant-numeric:tabular-nums;">'
        f'{result["predicted_return"]*100:+.2f}%</div></div>'
        f'<div><div class="k">Next price</div>'
        f'<div style="font-size:1.5rem;font-weight:650;'
        f'font-variant-numeric:tabular-nums;">'
        f'{price_fmt(result["predicted_price"])}</div></div>'
        f'<div><div class="k">From</div>'
        f'<div style="font-size:1.5rem;font-weight:650;color:#c3c2b7;'
        f'font-variant-numeric:tabular-nums;">'
        f'{price_fmt(result["price"])}</div></div>'
        f'</div>'
        f'<div style="margin-top:18px;color:#8b8a80;font-size:0.82rem;">'
        f'Forecast for the session after '
        f'<b style="color:#c3c2b7">{result["as_of"].strftime("%d %b %Y")}</b> · '
        f'signal strength <b style="color:#c3c2b7">{result["strength_label"]}</b> · '
        f'model verdict <b style="color:{grade_color}">{grade}</b></div></div>',
        unsafe_allow_html=True,
    )
with right:
    with st.container(border=True):
        st.markdown('<div class="k">Signal strength</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(charts.gauge(result["strength"], style["color"]),
                        config=PLOTLY_CONFIG, key="gauge")
        st.markdown(
            '<div style="font-size:0.72rem;color:#8b8a80;margin:-14px 0 2px;">'
            "How far the forecast clears the model's own noise floor, capped by "
            'its measured accuracy. <b>Not a probability of being right.</b></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="sec">Forecast</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(card(
    "Realistic range",
    f"{price_fmt(result['price_low'])} – {price_fmt(result['price_high'])}",
    f"±{result['typical_error_pct']:.2f}% typical error on unseen data "
    f"({meta['unit']})",
), unsafe_allow_html=True)
c2.markdown(card(
    "Decision threshold", f"±{result['threshold']*100:.2f}%",
    f"a forecast inside this band is called HOLD",
), unsafe_allow_html=True)
c3.markdown(card("Model verdict", grade, metrics["grade_note"],
                 grade_color), unsafe_allow_html=True)

st.markdown(
    f'<div class="note">The point forecast '
    f'(<b>{result["predicted_return"]*100:+.2f}%</b>) is much smaller than the '
    f'model\'s typical error (<b>±{result["typical_error_pct"]:.2f}%</b>). That '
    f'is normal for daily price prediction and it is the honest picture: the '
    f'signal is a small tilt in the odds, not a forecast you can rely on.</div>',
    unsafe_allow_html=True,
)

# ---------------------------- position plan -------------------------------
st.markdown('<div class="sec">Position plan</div>', unsafe_allow_html=True)

if not plan["tradeable"]:
    reason = ("The signal is HOLD, so no position is recommended."
              if sig == "HOLD" else
              "Short selling is switched off, so this SELL means exit or stay out.")
    st.markdown(
        f'<div class="note" style="border-left-color:{style["color"]}">'
        f'<b>No position recommended.</b> {reason} The levels below show what the '
        f'plan would be if you chose to take the trade anyway.</div>',
        unsafe_allow_html=True,
    )

p1, p2, p3, p4 = st.columns(4)
p1.markdown(card(
    "Position size", money(plan["position_value_if_taken"]),
    f"{plan['units_if_taken']:,.4f} units · {plan['capital_used_pct']:.0f}% of budget",
), unsafe_allow_html=True)
p2.markdown(card(
    "Amount at risk", money(plan["max_loss"]),
    f"{plan['max_loss']/budget*100:.2f}% of budget"
    + (" · capped by budget" if plan["capped_by_budget"] else ""),
), unsafe_allow_html=True)
p3.markdown(card(
    "Stop loss", price_fmt(plan["stop_loss"]),
    f"{plan['stop_pct']*100:.2f}% away · from {plan['stop_basis']}",
), unsafe_allow_html=True)
p4.markdown(card(
    "Take profit", price_fmt(plan["take_profit"]),
    f"{plan['stop_pct']*rr*100:.2f}% away · {rr:.1f}:1 reward-to-risk",
), unsafe_allow_html=True)

lp, rp = st.columns([1, 1.4])
with lp:
    st.plotly_chart(
        charts.risk_ladder(plan["entry"], plan["stop_loss"], plan["take_profit"],
                           plan["direction"], price_fmt),
        config=PLOTLY_CONFIG, key="ladder",
    )
with rp:
    st.markdown(
        f'<div class="card">'
        f'<div class="k">How this size was chosen</div>'
        f'<div style="color:#c3c2b7;font-size:0.86rem;line-height:1.65;">'
        f'You risk <b>{risk_pct:.1f}%</b> of {money(budget)} = '
        f'<b>{money(plan["risk_amount_requested"])}</b>. '
        f'{meta["name"]}\'s recent volatility puts a sensible stop '
        f'<b>{plan["stop_pct"]*100:.2f}%</b> from entry, so the largest position '
        f'that loses only that amount is '
        f'<b>{money(plan["position_value_if_taken"])}</b> '
        f'({plan["units_if_taken"]:,.4f} units at {price_fmt(plan["entry"])}).<br><br>'
        + (
            "That exceeded your budget, so the position is capped at 100% of "
            f"capital and the real risk falls to <b>{money(plan['max_loss'])}</b>.<br><br>"
            if plan["capped_by_budget"] else ""
        )
        + f'Direction: <b>{plan["direction"]}</b> · '
          f'max loss <b>{money(plan["max_loss"])}</b> · '
          f'max gain at target <b>{money(plan["max_gain"])}</b>.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# Keep the default presentation focused on the decision and risk plan. The
# explanation, validation charts and data audit remain available on demand.
show_technical = st.toggle(
    "Show technical analysis and model validation", value=False,
    help="Opens feature contributions, held-out metrics, backtests and data details.",
)
if not show_technical:
    st.markdown('<div class="sec">Recent price</div>', unsafe_allow_html=True)
    st.plotly_chart(charts.price_history(bundle["history"], meta["unit"]),
                    config=PLOTLY_CONFIG, key="summary_price")
    st.caption(
        "Educational tool, not financial advice. The forecast uses historical "
        "prices only and does not account for news or market costs."
    )
    st.stop()

# ---------------------------- explanation ---------------------------------
st.markdown('<div class="sec">Why the AI made this call</div>',
            unsafe_allow_html=True)
story = result["explanation"]
st.markdown(f'<div class="note">{story["summary"]}</div>', unsafe_allow_html=True)

e1, e2 = st.columns([1.25, 1])
with e1:
    st.plotly_chart(charts.drivers(story["bullets"]), config=PLOTLY_CONFIG,
                    key="drivers")
    st.caption(
        "A linear model is fully transparent: the forecast is the sum of every "
        "indicator's effect. Bars show the five indicators moving it most today."
    )
with e2:
    def dot(b):
        c = "#3987e5" if b["direction"] == "up" else "#e66767"
        return (f'<span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;background:{c};margin-right:9px;'
                f'vertical-align:middle;"></span>')

    rows = "".join(
        f'<div class="driver"><span class="name">{dot(b)}{b["label"]}<br>'
        f'<span style="color:#8b8a80;font-size:0.78rem;margin-left:17px;">'
        f'{b["display"]} · {b["note"]}</span></span>'
        f'<span class="val" style="color:'
        f'{"#3987e5" if b["direction"] == "up" else "#e66767"};">'
        f'{b["contribution_pct"]:+.3f}%</span></div>'
        for b in story["bullets"]
    )
    st.markdown(
        f'<div class="card"><div class="k">Today\'s readings &amp; their effect</div>{rows}</div>',
        unsafe_allow_html=True,
    )


ctx = "".join(
    f'<div style="flex:1 1 260px;color:#c3c2b7;font-size:0.85rem;'
    f'line-height:1.55;border-left:2px solid #2e2e2a;padding-left:12px;">{c}</div>'
    for c in story["context"]
)
st.markdown(
    f'<div class="card" style="margin-top:14px;"><div class="k">Market context</div>'
    f'<div style="display:flex;gap:22px;flex-wrap:wrap;">{ctx}</div></div>',
    unsafe_allow_html=True,
)

# ---------------------------- model performance ---------------------------
st.markdown('<div class="sec">Model performance on unseen data</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="note" style="border-left-color:{grade_color}">'
    f'<b style="color:{grade_color}">{grade}.</b> {metrics["grade_note"]} '
    f'Everything below is measured on <b>{metrics["n_test"]} days '
    f'({metrics["test_start"]} → {metrics["test_end"]})</b> that came after all '
    f'{metrics["n_train"]} training days and were never used to fit the model.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
dir_acc = metrics["directional_accuracy"] * 100
base_dir = metrics["baseline_directional_accuracy"] * 100
m1.markdown(card(
    "Direction accuracy", f"{dir_acc:.1f}%",
    f"training-chosen always-{metrics.get('baseline_direction', 'common')} baseline: {base_dir:.1f}%",
), unsafe_allow_html=True)
m2.markdown(card(
    "Test R²", f"{metrics['test_r2']:+.4f}",
    f"in-sample R² {metrics['train_r2']:+.4f} · 0 means no better than the mean",
), unsafe_allow_html=True)
m3.markdown(card(
    "Mean abs. error", f"{metrics['test_mae']*100:.3f}%",
    f"naive constant guess: {metrics['baseline_mae']*100:.3f}%",
), unsafe_allow_html=True)
m4.markdown(card(
    "Walk-forward accuracy",
    f"{metrics['cv_directional_accuracy_mean']*100:.1f}%",
    f"± {metrics['cv_directional_accuracy_std']*100:.1f}% over 5 expanding folds",
), unsafe_allow_html=True)

bt = metrics["backtest"]
tab1, tab2, tab3 = st.tabs(
    ["Simulated equity", "Predicted vs actual", "Recent price"]
)
with tab1:
    st.plotly_chart(charts.equity_curve(bundle["test_frame"], bt),
                    config=PLOTLY_CONFIG, key="equity")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Signal strategy", f"{bt['strategy_return']*100:+.1f}%")
    b2.metric("Buy & hold", f"{bt['buy_hold_return']*100:+.1f}%")
    b3.metric("Sharpe", f"{bt['sharpe']:.2f}")
    b4.metric("Max drawdown", f"{bt['max_drawdown']*100:.1f}%")
    st.caption(
        f"Long / flat / short on the model's own signal, one day held, "
        f"{bt['n_trades']} position changes, win rate "
        f"{bt['win_rate']*100:.1f}%. **Excludes commission, spread, slippage and "
        f"financing** — a real account would return less, and the "
        f"{bt['n_trades']} switches make those costs material."
    )
with tab2:
    st.plotly_chart(charts.pred_vs_actual(bundle["test_frame"]),
                    config=PLOTLY_CONFIG, key="scatter")
    st.caption(
        f"Each dot is one unseen trading day. Points in the top-right and "
        f"bottom-left quadrants are days the model got the direction right "
        f"({dir_acc:.1f}% of them). The near-flat spread of predictions against "
        f"a wide spread of outcomes is what an information coefficient of "
        f"{metrics['information_coefficient']:.3f} looks like."
    )
with tab3:
    st.plotly_chart(charts.price_history(bundle["history"], meta["unit"]),
                    config=PLOTLY_CONFIG, key="price")

# ---------------------------- details -------------------------------------
with st.expander("Model & data details"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Dataset**")
        st.dataframe(
            pd.DataFrame(
                [
                    ("File", report["file"]),
                    ("Rows in file", f"{report['rows_raw']:,}"),
                    ("Rows after cleaning", f"{report['rows_clean']:,}"),
                    ("Date range", f"{report['start']} → {report['end']}"),
                    ("Dropped: missing close", report["dropped_missing_close"]),
                    ("Dropped: non-positive close", report["dropped_nonpositive_close"]),
                    ("Duplicate dates removed", report["duplicate_dates"]),
                    ("Has OHLC", "yes" if report["has_ohlc"] else "no"),
                    ("Has volume", "yes" if report["has_volume"] else "no"),
                    ("Fingerprint", report["fingerprint"][:16] + "…"),
                ],
                columns=["Field", "Value"],
            ).astype(str),
            hide_index=True, width="stretch",
        )
    with d2:
        st.markdown("**Model artifact**")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Algorithm", "StandardScaler → LinearRegression"),
                    ("Features", str(len(bundle["feature_names"]))),
                    ("Target", "next session's close-to-close return"),
                    ("Train window", f"{metrics['train_start']} → {metrics['train_end']}"),
                    ("Test window", f"{metrics['test_start']} → {metrics['test_end']}"),
                    ("Trained at", bundle["trained_at"].replace("T", " ")),
                    ("Loaded from", "saved file" if source == "cache" else "fresh training"),
                    ("Saved to", f"models/{market_key}_linreg.joblib"),
                    ("Information coefficient", f"{metrics['information_coefficient']:.4f}"),
                    ("Signal threshold", f"±{bundle['threshold']*100:.3f}%"),
                ],
                columns=["Field", "Value"],
            ).astype(str),
            hide_index=True, width="stretch",
        )

    st.markdown("**Every feature's effect on today's forecast**")
    tbl = story["table"][["label", "display", "contribution"]].copy()
    tbl.columns = ["Indicator", "Today's reading", "Effect on forecast"]
    tbl["Effect on forecast"] = (tbl["Effect on forecast"] * 100).map("{:+.4f}%".format)
    st.dataframe(tbl, hide_index=True, width="stretch", height=380)

    st.markdown(
        "**How data leakage is prevented** — every indicator at day *t* uses "
        "only bars up to and including *t*; the target is day *t+1*'s return "
        "(one forward shift, applied once); the train/test split is "
        "chronological, so the test set is strictly newer than everything the "
        "model saw; and the feature scaler is fitted inside the pipeline on the "
        "training slice alone, so test statistics never reach it. The reported "
        "metrics come from that held-out slice. The model that produces the "
        "live forecast above is the same pipeline refitted on all history so "
        "tomorrow's call can use the most recent bars — it is never scored on "
        "data it was fitted to."
    )

st.markdown(
    '<div style="margin-top:34px;padding-top:16px;border-top:1px solid #2e2e2a;'
    'color:#8b8a80;font-size:0.76rem;line-height:1.6;">'
    '<b>Educational tool, not financial advice.</b> These are statistical '
    'forecasts from historical prices with no view on news, earnings or macro '
    'events. Daily direction prediction is close to a coin flip and the '
    'performance panel above reports that honestly rather than hiding it. '
    'Never risk money you cannot afford to lose.</div>',
    unsafe_allow_html=True,
)
