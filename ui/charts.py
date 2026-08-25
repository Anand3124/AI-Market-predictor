"""Plotly chart builders.

Rules applied throughout: at most three categorical series per chart (the
validated all-pairs limit for this palette), one y-axis per chart, 2px lines,
>=8px markers, a legend whenever there are two or more series, direct labels
used selectively, and a hover layer on every plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import (
    BORDER, CRITICAL, GOOD, GRID, MID, NEG, POS, S1, S2, S3, TEXT_MUTED,
    TEXT_SECONDARY, style_fig,
)


def price_history(history: pd.DataFrame, unit: str, bars: int = 260) -> go.Figure:
    d = history.tail(bars).copy()
    d["SMA 20"] = history["Close"].rolling(20).mean().tail(bars)
    d["SMA 50"] = history["Close"].rolling(50).mean().tail(bars)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["Close"], name="Close", mode="lines",
        line=dict(color=S1, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Close <b>%{y:,.2f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["SMA 20"], name="20-day average", mode="lines",
        line=dict(color=S2, width=2, dash="dot"),
        hovertemplate="20-day avg %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["SMA 50"], name="50-day average", mode="lines",
        line=dict(color=S3, width=2, dash="dot"),
        hovertemplate="50-day avg %{y:,.2f}<extra></extra>",
    ))
    # selective direct label: the latest close only
    last = d.iloc[-1]
    fig.add_annotation(
        x=pd.Timestamp(last["Date"]).to_pydatetime(), y=last["Close"],
        text=f"<b>{last['Close']:,.2f}</b>",
        showarrow=False, xanchor="left", xshift=8, font=dict(color=S1, size=12),
    )
    fig.update_layout(hovermode="x unified")
    return style_fig(fig, height=330, ytitle=unit)


def drivers(bullets: list[dict]) -> go.Figure:
    """Diverging bar: how much each indicator moved today's forecast."""
    b = list(reversed(bullets))
    vals = [x["contribution_pct"] for x in b]
    labels = [x["label"] for x in b]
    colors = [POS if v > 0 else NEG for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0),
                    cornerradius=4),
        width=0.42,
        text=[f"{v:+.3f}%" for v in vals],
        textposition="outside",
        textfont=dict(color=TEXT_SECONDARY, size=11),
        hovertemplate="%{y}<br>moves the forecast <b>%{x:+.3f}%</b><extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MID, width=1))
    span = max(abs(min(vals)), abs(max(vals)), 0.01) * 1.45
    fig.update_xaxes(range=[-span, span], showgrid=True, gridcolor=GRID,
                     ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXT_SECONDARY, size=12))
    return style_fig(fig, height=64 + 62 * len(b), legend=False,
                     xtitle="effect on the predicted return")


def equity_curve(test: pd.DataFrame, backtest: dict) -> go.Figure:
    dates = test["Date"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=(backtest["equity"] - 1) * 100, name="Model signal",
        mode="lines", line=dict(color=S1, width=2),
        hovertemplate="Model <b>%{y:+.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=(backtest["buy_hold_equity"] - 1) * 100, name="Buy & hold",
        mode="lines", line=dict(color=S2, width=2),
        hovertemplate="Buy & hold <b>%{y:+.1f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MID, width=1))
    fig.update_layout(hovermode="x unified")
    return style_fig(fig, height=320, ytitle="cumulative return (%)")


def pred_vs_actual(test: pd.DataFrame) -> go.Figure:
    """Each unseen day plotted as forecast vs outcome, split by whether the
    model called the direction right. The two zero lines cut the plot into
    quadrants: top-right and bottom-left are the correct calls."""
    x = test["predicted"] * 100
    y = test["actual"] * 100
    right = np.sign(test["predicted"]) == np.sign(test["actual"])
    labels = test["Date"].dt.strftime("%d %b %Y")

    fig = go.Figure()
    for mask, name, color in (
        (right, "Direction correct", POS),
        (~right, "Direction wrong", NEG),
    ):
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers", name=name,
            marker=dict(size=8, color=color, opacity=0.7,
                        line=dict(color="#1a1a19", width=2)),
            customdata=labels[mask],
            hovertemplate="%{customdata}<br>predicted %{x:+.2f}%<br>"
                          "actual <b>%{y:+.2f}%</b><extra></extra>",
        ))

    # least-squares fit: the visible slope IS the model's edge, and it is faint
    if len(x) > 2 and float(np.std(x)) > 0:
        m, c = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + c, mode="lines", name="fitted trend",
            line=dict(color=TEXT_MUTED, width=2, dash="dash"),
            hoverinfo="skip",
        ))

    fig.add_vline(x=0, line=dict(color=MID, width=1))
    fig.add_hline(y=0, line=dict(color=MID, width=1))
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(ticksuffix="%")
    return style_fig(fig, height=360,
                     xtitle="predicted next-day return (note the narrow range)",
                     ytitle="what actually happened")


def gauge(strength: float, color: str) -> go.Figure:
    """Single-number meter for signal strength."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=strength,
        number=dict(valueformat=".0f", suffix=" / 100",
                    font=dict(size=30, color="#ffffff")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=BORDER,
                      tickfont=dict(color=TEXT_MUTED, size=10)),
            bar=dict(color=color, thickness=0.7),
            bgcolor="#232320", borderwidth=0,
            steps=[dict(range=[0, 100], color="#232320")],
        ),
    ))
    fig.update_layout(
        height=195, margin=dict(l=34, r=34, t=20, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    return fig


def risk_ladder(entry: float, stop: float, target: float, direction: str,
                fmt) -> go.Figure:
    """Entry, stop and target on one price axis, with the loss zone and the
    profit zone shaded so the reward-to-risk shape is visible at a glance."""
    lo, hi = min(stop, target, entry), max(stop, target, entry)
    pad = (hi - lo) * 0.16 or 1.0
    risk_pct = abs(stop - entry) / entry * 100.0
    reward_pct = abs(target - entry) / entry * 100.0

    fig = go.Figure()
    # zones (status colours, each carrying its own text label)
    fig.add_hrect(y0=min(entry, stop), y1=max(entry, stop),
                  fillcolor=CRITICAL, opacity=0.13, line_width=0, layer="below")
    fig.add_hrect(y0=min(entry, target), y1=max(entry, target),
                  fillcolor=GOOD, opacity=0.13, line_width=0, layer="below")

    for level, color, name, extra in (
        (target, GOOD, "Take profit", f"+{reward_pct:.2f}%"),
        (entry, S1, "Entry", direction.lower()),
        (stop, CRITICAL, "Stop loss", f"-{risk_pct:.2f}%"),
    ):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[level, level], mode="lines",
            line=dict(color=color, width=2,
                      dash="solid" if name == "Entry" else "dash"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=0.985, y=level,
            text=f"{name} <b>{fmt(level)}</b>"
                 f"<span style='color:{TEXT_MUTED}'>  {extra}</span>",
            showarrow=False, xanchor="right", yshift=12,
            font=dict(color=color, size=12),
        )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(range=[lo - pad, hi + pad], showgrid=False,
                     tickfont=dict(color=TEXT_MUTED, size=11))
    fig.update_layout(
        height=230, margin=dict(l=8, r=8, t=18, b=8), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    return fig
