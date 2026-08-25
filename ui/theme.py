"""Design tokens + Plotly styling.

Colours come from the validated reference palette, dark column, on the
documented dark chart surface (#1a1a19). Categorical slots 1-3 clear the
all-pairs colour-vision gates on that surface, so charts never use more than
three categorical series; polarity uses the blue/red diverging pair, and
signal state uses the reserved status palette (always with an icon + label,
never colour alone).
"""

SURFACE = "#1a1a19"
SURFACE_PAGE = "#121211"
BORDER = "#2e2e2a"
GRID = "#262623"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#8b8a80"

# categorical slots (dark)
S1 = "#3987e5"   # blue
S2 = "#d95926"   # orange
S3 = "#199e70"   # aqua

# diverging pair + neutral midpoint
POS = "#3987e5"
NEG = "#e66767"
MID = "#383835"

# reserved status palette
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

SIGNAL_STYLE = {
    "BUY": {"color": GOOD, "icon": "▲", "word": "BUY"},
    "HOLD": {"color": WARNING, "icon": "■", "word": "HOLD"},
    "SELL": {"color": CRITICAL, "icon": "▼", "word": "SELL"},
}

GRADE_STYLE = {
    "Promising": GOOD,
    "Slight edge": WARNING,
    "Coin-flip": SERIOUS,
    "No edge": CRITICAL,
}


def style_fig(fig, *, height=340, legend=True, ytitle=None, xtitle=None):
    """Apply the shared chart chrome: recessive axes, transparent surface."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, Segoe UI, sans-serif",
            size=12,
            color=TEXT_SECONDARY,
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY, size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=BORDER,
        tickcolor=BORDER, tickfont=dict(color=TEXT_MUTED, size=11),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
        linecolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT_MUTED, size=11),
    )
    # Only set a title when one is asked for -- otherwise a title set by the
    # caller before styling would be silently wiped out.
    if xtitle:
        fig.update_xaxes(title=dict(text=xtitle, font=dict(color=TEXT_MUTED, size=11)))
    if ytitle:
        fig.update_yaxes(title=dict(text=ytitle, font=dict(color=TEXT_MUTED, size=11)))
    return fig


CSS = f"""
<style>
:root {{
  --surface: {SURFACE};
  --page: {SURFACE_PAGE};
  --border: {BORDER};
  --ink: {TEXT_PRIMARY};
  --ink-2: {TEXT_SECONDARY};
  --ink-3: {TEXT_MUTED};
}}
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}}

.hero {{
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 26px;
  background: linear-gradient(135deg, #1c1c1a 0%, #141413 100%);
  margin-bottom: 18px;
}}
.hero h1 {{
  font-size: 1.55rem; font-weight: 650; margin: 0; color: var(--ink);
  letter-spacing: -0.02em;
}}
.hero p {{margin: 6px 0 0; color: var(--ink-3); font-size: 0.88rem;}}

.card {{
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  background: var(--surface);
  height: 100%;
}}
.card .k, .k {{
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--ink-3); font-weight: 600; margin-bottom: 7px;
}}
/* st.container(border=True) styled to match the hand-rolled .card blocks */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > .stElementContainer) {{
  border-radius: 14px;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: var(--border) !important;
  border-radius: 14px !important;
  background: var(--surface);
}}
.card .v {{font-size: 1.45rem; font-weight: 640; color: var(--ink); line-height: 1.15;
  letter-spacing: -0.01em; font-variant-numeric: tabular-nums;}}
.card .s {{font-size: 0.76rem; color: var(--ink-3); margin-top: 5px;}}

.signal-card {{
  border-radius: 16px; padding: 26px 28px; border: 1px solid var(--border);
  background: var(--surface); position: relative; overflow: hidden;
}}
.signal-card .bar {{
  position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
}}
.signal-word {{
  font-size: 3.1rem; font-weight: 720; letter-spacing: -0.035em; line-height: 1;
}}
.signal-sub {{color: var(--ink-2); font-size: 0.92rem; margin-top: 10px;}}

.chip {{
  display: inline-block; padding: 3px 11px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid var(--border); color: var(--ink-2); background: #201f1d;
}}
.sec {{
  font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--ink-3); font-weight: 650; margin: 26px 0 12px;
  border-bottom: 1px solid var(--border); padding-bottom: 8px;
}}
.note {{
  border-left: 3px solid var(--border); padding: 10px 14px; margin: 10px 0;
  color: var(--ink-2); font-size: 0.86rem; background: #171716; border-radius: 0 8px 8px 0;
}}
.driver {{
  display: flex; justify-content: space-between; gap: 12px; padding: 9px 0;
  border-bottom: 1px solid #232320; font-size: 0.87rem;
}}
.driver:last-child {{border-bottom: none;}}
.driver .name {{color: var(--ink);}}
.driver .val {{color: var(--ink-3); font-variant-numeric: tabular-nums; white-space: nowrap;}}
.stButton>button {{border-radius: 10px; font-weight: 600;}}
div[data-testid="stMetricValue"] {{font-size: 1.2rem;}}
</style>
"""
