"""Reusable Plotly figures (shared by notebook and app) with enhanced modern aesthetics."""
from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

LAB = {
    "Total_Population": "Total population (persons)",
    "Child_Population": "Child population, age 0-6 (persons)",
    "Child_Population_Share": "Child population share, age 0-6 (% of total)",
    "Literacy_Rate": "Literacy rate (%)",
    "Literacy_Gap": "Literacy gap (percentage points)",
    "Sex_Ratio": "Sex ratio (females per 1000 males)",
    "Workforce_Participation_Rate": "Workforce participation (% of total population)",
    "SC_Population_Share": "SC share (% of total)",
    "ST_Population_Share": "ST share (% of total)",
    "UIPS": "UIPS (0-100)",
}
lab = lambda c: LAB.get(c, c)

# Modern clean styling presets
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
COLOR_PRIMARY = "#4338CA"  # Deep Indigo
COLOR_PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#EC4899", "#14B8A6"]


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12, color="#1E293B"),
        title=dict(font=dict(family=FONT_FAMILY, size=15, color="#0F172A", weight=600)),
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#0F172A",
            bordercolor="#CBD5E1",
        ),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", gridwidth=0.8, zerolinecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", gridwidth=0.8, zerolinecolor="#CBD5E1")
    return fig


def bar_h(df, col, label_col="District_Label", title=None, color=None):
    d = df.sort_values(col)
    color_scale = px.colors.sequential.Tealgrn if not color else None
    f = px.bar(
        d,
        x=col,
        y=label_col,
        orientation="h",
        color=color or col,
        color_continuous_scale=color_scale,
        labels={col: lab(col), label_col: ""},
        title=title or lab(col),
    )
    f.update_layout(
        height=max(360, 32 * len(d) + 120),
        yaxis=dict(automargin=True),
        coloraxis_showscale=False if not color else True,
    )
    return _apply_theme(f)


def histogram(df, col, nbins=30, title=None):
    f = px.histogram(
        df,
        x=col,
        nbins=nbins,
        labels={col: lab(col)},
        title=title or f"Distribution of {lab(col)}",
        color_discrete_sequence=[COLOR_PRIMARY],
        marginal="box",
    )
    f.update_layout(yaxis_title="Number of districts")
    return _apply_theme(f)


def box(df, col, title=None):
    f = px.box(
        df,
        y=col,
        points="outliers",
        hover_name="District_Label",
        labels={col: lab(col)},
        title=title or f"Box plot: {lab(col)}",
        color_discrete_sequence=["#0284C7"],
    )
    return _apply_theme(f)


def scatter(df, x, y, color=None, size=None, title=None, trend=False):
    f = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name="District_Label",
        labels={x: lab(x), y: lab(y)},
        title=title or f"{lab(y)} vs {lab(x)}",
        trendline="ols" if trend else None,
        trendline_color_override="#DC2626" if trend else None,
        opacity=0.8,
        color_discrete_sequence=COLOR_PALETTE,
    )
    return _apply_theme(f)


def state_bar(df, x, y="state_name", title=None, xlabel=None):
    d = df.sort_values(x)
    f = px.bar(
        d,
        x=x,
        y=y,
        orientation="h",
        labels={x: xlabel or lab(x), y: ""},
        title=title or lab(x),
        color=x,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    f.update_layout(
        height=max(480, 24 * len(d) + 120),
        yaxis=dict(automargin=True),
        coloraxis_showscale=False,
    )
    return _apply_theme(f)


def category_bar(df, col="Priority_Category", order=("Low", "Medium", "High"), title="Districts per priority category"):
    c = df[col].value_counts().reindex(list(order)).fillna(0).reset_index()
    c.columns = [col, "Districts"]
    colors = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
    c["Color"] = [colors.get(cat, "#6366F1") for cat in c[col]]
    f = px.bar(
        c,
        x=col,
        y="Districts",
        text="Districts",
        title=title,
        color=col,
        color_discrete_map=colors,
        labels={col: "Category", "Districts": "Number of districts"},
    )
    f.update_traces(textposition="outside", textfont=dict(size=13, weight="bold"))
    f.update_layout(showlegend=False)
    return _apply_theme(f)


def corr_heatmap(corr, title="Correlation matrix"):
    f = px.imshow(
        corr.round(2),
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title=title,
        aspect="auto",
    )
    return _apply_theme(f)


def confusion_heatmap(cm, title="Confusion matrix (test set)"):
    f = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
        labels=dict(x="Predicted Class", y="Actual Class", color="Districts"),
    )
    return _apply_theme(f)


def importance_bar(imp, col="Impurity_Importance", title="Random Forest feature importance"):
    d = imp.sort_values(col)
    f = px.bar(
        d,
        x=col,
        y="Feature",
        orientation="h",
        title=title,
        color=col,
        color_continuous_scale=px.colors.sequential.Purp,
        labels={col: col.replace("_", " "), "Feature": ""},
    )
    f.update_layout(coloraxis_showscale=False)
    return _apply_theme(f)


def gauge_chart(score: float, title: str = "UIPS Priority Index", subtitle: str = "") -> go.Figure:
    """Semi-circular gauge chart for priority indexing."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{subtitle}</span>", "font": {"family": FONT_FAMILY, "size": 16}},
            number={"font": {"family": FONT_FAMILY, "size": 36, "color": "#0F172A"}, "suffix": ""},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": "#1E293B", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 33.33], "color": "rgba(16, 185, 129, 0.35)"},
                    {"range": [33.33, 66.67], "color": "rgba(245, 158, 11, 0.35)"},
                    {"range": [66.67, 100], "color": "rgba(239, 68, 68, 0.40)"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY),
    )
    return fig


def radar_profile(district_name: str, values: dict[str, float]) -> go.Figure:
    """Radar/Spider chart showing normalized demographic attributes (0-100%)."""
    categories = list(values.keys())
    vals = list(values.values())
    categories.append(categories[0])
    vals.append(vals[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=vals,
            theta=categories,
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.25)",
            line=dict(color="#4F46E5", width=2.5),
            name=district_name,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#64748B")),
            angularaxis=dict(tickfont=dict(family=FONT_FAMILY, size=11, color="#1E293B", weight="bold")),
        ),
        title=dict(text=f"Demographic Profile: {district_name}", font=dict(family=FONT_FAMILY, size=14)),
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def probability_bar(classes: list[str], probs: list[float], title: str = "Class Probabilities") -> go.Figure:
    """Horizontal bar for predicted class probabilities."""
    palette = {"Low": "#EF4444", "Medium": "#F59E0B", "High": "#10B981"}
    colors = [palette.get(c, "#6366F1") for c in classes]
    df = pd.DataFrame({"Class": classes, "Probability": [p * 100 for p in probs], "Color": colors})
    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        text=df["Probability"].apply(lambda p: f"{p:.1f}%"),
        color="Class",
        color_discrete_map=palette,
        title=title,
    )
    fig.update_traces(textposition="outside", textfont=dict(size=12, weight="bold"))
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="Probability (%)"),
        yaxis=dict(title=""),
        height=220,
        showlegend=False,
        margin=dict(l=20, r=40, t=40, b=20),
    )
    return _apply_theme(fig)
