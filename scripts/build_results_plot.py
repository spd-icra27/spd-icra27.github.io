"""Build the interactive results figures for the website.

Rebuilds two figures from the paper's notebook data
(notebooks/results.ipynb in the paper repository) in Plotly:
  - window.RESULTS_FIG: average task progress bar chart
  - window.LOSS_FIGS:  per-task training-loss curves for the carousel

Writes assets/results_fig.js, loaded by index.html.

Usage: python3 scripts/build_results_plot.py [path-to-notebooks-dir]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NOTEBOOKS = Path(
    sys.argv[1] if len(sys.argv) > 1 else "~/notebooks"
).expanduser()
OUT = Path(__file__).resolve().parent.parent / "assets" / "results_fig.js"

COLORS = {"ft": "#4c81b6", "scratch": "#d5e4ef"}
SERIES_LABELS = {"ft": "SPD, pre-trained", "scratch": "BC, from-scratch"}
TASKS = ["platerack", "cupstack", "jenga", "mugtree", "bottles"]
BAR_LABELS = {
    "bottles": "Bottles",
    "platerack": "Plates",
    "cupstack": "Cups",
    "jenga": "Jenga",
    "mugtree": "Mugs",
}

LOSS_RUNS = {
    "bottles": {
        "ft": "bc-real-bottles-op-2607-dino-frozen-xblock-chunk8-win32-from-ep0207",
        "scratch": "bc-real-bottles-op-2607-dino-scratch-xblock-chunk8-win32",
    },
    "platerack": {
        "ft": "demo_dino_ft_20260626_203251_off1_xblock",
        "scratch": "demo_dino_scratch_20260708_141150_off1_xblock",
    },
    "cupstack": {
        "ft": "cupstackv2_dino_ft_20260709_054622_off1_xblock",
        "scratch": "bc-real-cupstackv2-dino-scratch-xblock-chunk8-win32",
    },
    "jenga": {
        "ft": "bc-real-jengapullv2-193traj-dino-frozen-xblock-chunk8-win32-from-ep0207",
        "scratch": "bc-real-jengapullv2-193traj-dino-scratch-xblock-chunk8-win32",
    },
    "mugtree": {
        "ft": "bc-real-mugtreev3-op-260727-dino-frozen-xblock-chunk8-win32-from-ep0207",
        "scratch": "bc-real-mugtreev3-op-260727-dino-scratch-xblock-chunk8-win32",
    },
}

FONT = "Charter, Georgia, serif"  # matches the site's body text
INK = "#2f2f2f"
AXIS = "#9c9c9c"


def axis_style(**kw):
    base = dict(
        fixedrange=True,  # hover-only: no zoom/pan on any axis
        showline=True,
        linecolor=AXIS,
        linewidth=1,
        ticks="outside",
        tickcolor=AXIS,
        ticklen=4,
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=13),
        title_font=dict(size=14),
    )
    base.update(kw)
    return base


def base_layout(**kw):
    layout = dict(
        autosize=True,
        dragmode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, color=INK, size=13),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.04,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13),
            itemclick=False,
            itemdoubleclick=False,
        ),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    layout.update(kw)
    return layout


def load_experiments():
    df = pd.read_csv(NOTEBOOKS / "experiments_clean.csv").ffill()
    df["progress"] = 100 * df["score"] / df["max_score"]
    return df


def load_loss(task):
    renames = {"train/iter": "iter"}
    renames.update(
        {f"{run} - train/iter_flow_action_mse": w for w, run in LOSS_RUNS[task].items()}
    )
    df = pd.read_csv(NOTEBOOKS / f"{task}_train_loss.csv").rename(columns=renames)
    df = df[["iter", "scratch", "ft"]].query("(iter >= 200) & (iter < 10000)")
    # The two runs log on different iteration grids, so each series has NaN
    # rows where only the other run logged; drop and downsample per series.
    series = {}
    for w in ["ft", "scratch"]:
        s = df[["iter", w]].dropna()
        series[w] = s.iloc[:: max(1, len(s) // 400)]  # ~400 points per curve
    return series


# Average task progress with standard-error bars
experiments = load_experiments()
bar_fig = go.Figure()
for w in ["ft", "scratch"]:
    g = experiments.query("weights == @w").groupby("task")["progress"]
    mean = g.mean().reindex(TASKS)
    se = (g.std() / np.sqrt(g.count())).reindex(TASKS)
    bar_fig.add_trace(
        go.Bar(
            x=[BAR_LABELS[t] for t in TASKS],
            y=mean,
            error_y=dict(type="data", array=se, color=INK, thickness=1, width=3),
            marker_color=COLORS[w],
            name=SERIES_LABELS[w],
            hovertemplate="%{y:.0f}%<extra></extra>",
        )
    )
bar_fig.update_layout(
    base_layout(
        height=420,
        barmode="group",
        bargap=0.3,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=axis_style(),
        yaxis=axis_style(range=[0, 100], tickvals=[0, 50, 100], title_text="Task progress (%)"),
    )
)

# Per-task training-loss curves
loss_figs = {}
for task in TASKS:
    loss = load_loss(task)
    fig = go.Figure()
    for w in ["ft", "scratch"]:
        fig.add_trace(
            go.Scatter(
                x=loss[w]["iter"],
                y=loss[w][w],
                mode="lines",
                line=dict(color=COLORS[w], width=2.2),
                name=SERIES_LABELS[w],
                hoverinfo="skip",
            )
        )
    ymax = np.ceil(max(loss[w][w].max() for w in ["ft", "scratch"]) * 25) / 25
    fig.update_layout(
        base_layout(
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=axis_style(
                range=[0, 10000], tickvals=[0, 5000, 10000], ticktext=["0", "5k", "10k"], title_text="Iteration"
            ),
            yaxis=axis_style(
                range=[0, ymax],
                tickvals=list(np.linspace(0, ymax, 3)),
                tickformat=".2f",
                title_text="Loss",
            ),
        )
    )
    loss_figs[task] = json.loads(fig.to_json())

out = (
    "window.RESULTS_FIG = "
    + bar_fig.to_json()
    + ";\nwindow.LOSS_FIGS = "
    + json.dumps(loss_figs)
    + ";\n"
)
OUT.write_text(out)
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
