"""Build the window/chunk ablation radar plots.

Writes assets/ablation_fig.js with window.ABLATION_FIGS: one radar figure
per training regime (ft / scratch), rendered side by side on desktop and
stacked on mobile. The legend is plain HTML in index.html.

Usage: python3 scripts/build_ablation_plot.py [path-to-csv]
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/experiments_clean.csv").expanduser()
OUT = Path(__file__).resolve().parent.parent / "assets" / "ablation_fig.js"

FONT = "Charter, Georgia, serif"
INK = "#2f2f2f"
AXIS = "#9c9c9c"

TASKS = ["platerack", "cupstack", "jenga", "mugtree", "bottles"]
TASK_LABELS = {
    "platerack": "Plates",
    "cupstack": "Cups",
    "jenga": "Jenga",
    "mugtree": "Mugs",
    "bottles": "Bottles",
}

# Seaborn "deep" palette; ours keeps the site blue.
VARIANTS = [
    (1, 8, "#C44E52"),
    (1, 32, "#DD8452"),
    (32, 32, "#55A868"),
    (32, 8, "#4c81b6"),
]

TITLES = {"ft": "SPD, pre-trained", "scratch": "BC, from-scratch"}

df = pd.read_csv(CSV).ffill()
df["progress"] = 100 * df["score"] / df["max_score"]

figs = {}
for weights in ["ft", "scratch"]:
    sub = df[df["weights"] == weights]
    fig = go.Figure()
    for win, chunk, color in VARIANTS:
        r = []
        for task in TASKS:
            rows = sub[(sub["win"] == win) & (sub["chunk"] == chunk) & (sub["task"] == task)]["progress"]
            r.append(round(rows.mean(), 1) if len(rows) else 0.0)
        theta = [TASK_LABELS[t] for t in TASKS]
        r = r + r[:1]
        theta = theta + theta[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                showlegend=False,
                hovertemplate="%{r:.0f}%<extra></extra>",
            )
        )
    fig.update_layout(
        height=360,
        autosize=True,
        dragmode=False,
        paper_bgcolor="white",
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=45, r=45, t=60, b=15),
        polar=dict(
            bgcolor="white",
            domain=dict(x=[0, 1], y=[0, 1]),
            radialaxis=dict(
                range=[-5, 100],
                tickvals=[0, 100],
                angle=54,
                showticklabels=False,
                showline=False,
                gridcolor="#e8e8e8",
            ),
            angularaxis=dict(
                gridcolor="#e8e8e8",
                linecolor=AXIS,
                tickfont=dict(size=13),
                rotation=90,
                direction="clockwise",
            ),
        ),
        annotations=[
            dict(x=0.5, y=1.13, xref="paper", yref="paper", xanchor="center",
                 text=TITLES[weights], showarrow=False, font=dict(size=15, color=INK)),
        ],
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    figs[weights] = json.loads(fig.to_json())

OUT.write_text("window.ABLATION_FIGS = " + json.dumps(figs) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
