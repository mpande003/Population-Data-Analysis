"""Module D - Urban Infrastructure Planning Priority Score (UIPS).
A district-level DEMOGRAPHIC planning priority score: a screening tool, not a measure of infrastructure deficits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# key: (column in cleaned data, short symbol, higher_is_priority, default weight)
COMPONENTS = {
    "Total_Population": ("Total_Population", "P", True, 0.35),
    "Child_Population_Share": ("Child_Population_Share", "C", True, 0.25),
    "Literacy_Gap": ("Literacy_Gap", "L", True, 0.25),
    "Workforce_Participation_Rate": ("Workforce_Participation_Rate", "W", False, 0.15),
}
WORKFORCE_RATIONALE = ("Assumption: lower workforce participation (workers / total population) is treated as higher priority because it "
                       "may indicate a weaker local employment base and higher dependency burden. This is a weak, assumption-laden "
                       "direction (it overlaps with child share, since children are non-workers); it can be switched or omitted.")
CATEGORY_EDGES = (33.33, 66.67)
DEFAULT_WEIGHTS = {k: v[3] for k, v in COMPONENTS.items()}
SCENARIOS = {
    "Baseline (illustrative)": DEFAULT_WEIGHTS,
    "Equal weights": {k: 0.25 for k in COMPONENTS},
    "Population-heavy": {"Total_Population": .50, "Child_Population_Share": .15, "Literacy_Gap": .20, "Workforce_Participation_Rate": .15},
    "Social-development-heavy": {"Total_Population": .20, "Child_Population_Share": .30, "Literacy_Gap": .35, "Workforce_Participation_Rate": .15},
    "Without workforce (renormalised)": {"Total_Population": .35, "Child_Population_Share": .25, "Literacy_Gap": .25, "Workforce_Participation_Rate": 0.0},
}


def normalize_weights(weights: dict) -> dict:
    """Drop zero/None weights, validate non-negativity, rescale to sum to 1."""
    w = {k: float(v) for k, v in weights.items() if v not in (None,) and float(v) != 0.0}
    if any(v < 0 for v in w.values()):
        raise ValueError("Weights must be non-negative.")
    s = sum(w.values())
    if s <= 0:
        raise ValueError("At least one component needs a positive weight.")
    return {k: v / s for k, v in w.items()}


def minmax(s: pd.Series, higher_is_priority: bool = True, winsor: tuple[float, float] | None = None):
    """Min-max to 0-1. Returns (normalized, warning or None). Constant columns -> 0.0 with a warning."""
    x = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if winsor:
        lo, hi = x.quantile(winsor[0] / 100), x.quantile(winsor[1] / 100)
        x = x.clip(lo, hi)
    mn, mx = x.min(), x.max()
    if not np.isfinite(mn) or mx == mn:
        return pd.Series(np.where(x.notna(), 0.0, np.nan), index=s.index), f"{s.name}: constant/empty; normalised score set to 0"
    n = (x - mn) / (mx - mn)
    return (n if higher_is_priority else 1 - n), None


def categorize(u: pd.Series) -> pd.Series:
    lo, hi = CATEGORY_EDGES
    out = pd.Series(np.select([u <= lo, u <= hi, u > hi], ["Low", "Medium", "High"], default=""), index=u.index)
    return out.where(u.notna(), np.nan)


def compute_uips(df: pd.DataFrame, weights: dict | None = None, directions: dict | None = None,
                 winsor: tuple[float, float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (district table, info). Rows with any missing selected component get no UIPS and are listed in info."""
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    unknown = set(w) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components: {unknown}")
    missing_cols = [k for k in w if k not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing for UIPS: {missing_cols}")
    directions = directions or {}
    d = df[["District_Label", "state_name"] + list(w)].copy()
    warns, norm_cols = [], []
    for k in w:
        hip = directions.get(k, COMPONENTS[k][2])
        n, warn = minmax(d[k], hip, winsor)
        d[f"{COMPONENTS[k][1]}_norm"] = n
        norm_cols.append(f"{COMPONENTS[k][1]}_norm")
        if warn:
            warns.append(warn)
    valid = d[norm_cols].notna().all(axis=1)
    contrib = sum(d[f"{COMPONENTS[k][1]}_norm"] * w[k] for k in w)
    d["UIPS"] = (100 * contrib).where(valid)
    assert d["UIPS"].dropna().between(-1e-9, 100 + 1e-9).all() and np.isfinite(d["UIPS"].dropna()).all()
    for k in w:  # percentage-point contribution of each component
        d[f"{COMPONENTS[k][1]}_contribution"] = (100 * d[f"{COMPONENTS[k][1]}_norm"] * w[k]).where(valid)
    d["Priority_Category"] = categorize(d["UIPS"])
    d["Rank"] = d["UIPS"].rank(ascending=False, method="min")
    d = d.sort_values("Rank").reset_index(drop=True)
    info = {"weights": w, "weight_sum": sum(w.values()), "excluded_missing": int((~valid).sum()), "warnings": warns, "winsor": winsor,
            "directions": {k: directions.get(k, COMPONENTS[k][2]) for k in w}}
    return d, info


def component_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between components: checks for double-counting."""
    return df[[c for c in COMPONENTS if c in df]].corr(method="spearman")


def sensitivity_analysis(df: pd.DataFrame, scenarios: dict | None = None, winsor=None):
    """Recompute UIPS under several weight sets. Returns (scores, ranks, categories, summary, rank_corr, category_changes)."""
    scenarios = scenarios or SCENARIOS
    base = next(iter(scenarios))
    scores, ranks, cats = {}, {}, {}
    for name, w in scenarios.items():
        t, _ = compute_uips(df, w, winsor=winsor)
        t = t.set_index("District_Label")
        scores[name], ranks[name], cats[name] = t["UIPS"], t["Rank"], t["Priority_Category"]
    S, R, C = pd.DataFrame(scores), pd.DataFrame(ranks), pd.DataFrame(cats)
    rows = []
    for n in scenarios:
        ch = (R[n] - R[base]).abs()
        top10 = set(R[n].nsmallest(10).index) & set(R[base].nsmallest(10).index)
        rows.append({"Scenario": n, "Weights": {k: round(v, 4) for k, v in normalize_weights(scenarios[n]).items()},
                     "Mean_abs_rank_change_vs_base": ch.mean(), "Max_abs_rank_change_vs_base": ch.max(),
                     "Top10_overlap_with_base": len(top10), "Spearman_vs_base": spearmanr(R[n], R[base], nan_policy="omit")[0],
                     "Districts_changing_category": int((C[n] != C[base]).sum())})
    corr = R.corr(method="spearman")
    changed = C[(C.nunique(axis=1) > 1)]
    return S, R, C, pd.DataFrame(rows), corr, changed


def contribution_summary(table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [c for c in table.columns if c.endswith("_contribution")]
    return table.head(n)[["Rank", "District_Label", "state_name", "UIPS"] + cols]
