"""Correlation, hypothesis-test and ANOVA helpers (SciPy only)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].dropna().corr(method=method)


def pair_correlation(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    d = df[[x, y]].dropna()
    pr, pp = stats.pearsonr(d[x], d[y])
    sr, sp = stats.spearmanr(d[x], d[y])
    return pd.DataFrame({"Method": ["Pearson", "Spearman"], "Coefficient": [pr, sr], "P-value": [pp, sp], "n": len(d)})


def shapiro(x) -> float:
    x = pd.Series(x).dropna()
    return float(stats.shapiro(x).pvalue) if 3 <= len(x) <= 5000 else float("nan")


def two_group_test(df: pd.DataFrame, value: str, group_col: str, alpha: float = ALPHA) -> dict:
    """Compare `value` between exactly two groups. Welch t-test if both groups look normal,
    otherwise Mann-Whitney U. Reports statistic, p, effect size and interpretation."""
    d = df[[value, group_col]].dropna()
    levels = sorted(d[group_col].unique())
    if len(levels) != 2:
        raise ValueError("group_col must have exactly two levels")
    a, b = (d.loc[d[group_col] == l, value] for l in levels)
    if min(len(a), len(b)) < 5:
        raise ValueError("Each group needs at least 5 observations")
    normal = shapiro(a) > alpha and shapiro(b) > alpha
    if normal:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        eff, eff_name, test, stat = (a.mean() - b.mean()) / sp, "Cohen's d", "Welch t-test", t
    else:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        eff, eff_name, test, stat = 1 - 2 * u / (len(a) * len(b)), "rank-biserial r", "Mann-Whitney U", u
    verdict = "reject H0" if p < alpha else "fail to reject H0"
    return {"test": test, "groups": levels, "n": (len(a), len(b)), "medians": (a.median(), b.median()),
            "statistic": float(stat), "p_value": float(p), "alpha": alpha, "effect_size": float(eff),
            "effect_size_name": eff_name, "decision": verdict,
            "interpretation": f"{test}: {verdict} at alpha={alpha} (p={p:.4g}, {eff_name}={eff:.3f}). "
                              "This indicates an association between groups, not causation."}


def welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    mw = np.sum(w * m) / np.sum(w)
    A = np.sum(w * (m - mw) ** 2) / (k - 1)
    lam = np.sum((1 - w / w.sum()) ** 2 / (n - 1))
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = A / B
    df2 = (k ** 2 - 1) / (3 * lam)
    return float(F), float(stats.f.sf(F, k - 1, df2))


def one_way_anova(df: pd.DataFrame, value: str, group_col: str, min_group_size: int = 5, alpha: float = ALPHA) -> dict:
    """One-way comparison across groups with assumption checks.
    Groups with fewer than `min_group_size` observations are excluded (documented in output).
    If residual normality or equal variance fails, Kruskal-Wallis is the primary result."""
    d = df[[value, group_col]].dropna()
    sizes = d.groupby(group_col)[value].size()
    keep = sizes[sizes >= min_group_size].index
    excluded = sizes[sizes < min_group_size].index.tolist()
    d = d[d[group_col].isin(keep)]
    names = sorted(d[group_col].unique())
    if len(names) < 3:
        raise ValueError("Need at least 3 groups of adequate size for a one-way ANOVA")
    groups = [d.loc[d[group_col] == g, value].to_numpy() for g in names]
    resid = np.concatenate([g - g.mean() for g in groups])
    sh_p = shapiro(resid)
    lev_p = float(stats.levene(*groups, center="median").pvalue)
    F, p_classic = stats.f_oneway(*groups)
    Fw, p_welch = welch_anova(groups)
    H, p_kw = stats.kruskal(*groups)
    n, k = len(d), len(names)
    grand = d[value].mean()
    eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((d[value] - grand) ** 2).sum()
    eps2 = (H - k + 1) / (n - k)
    ok = (sh_p > alpha) and (lev_p > alpha)
    primary = "Classic ANOVA" if ok else "Kruskal-Wallis (assumptions violated)"
    pprim = p_classic if ok else p_kw
    return {"groups_used": k, "groups_excluded_small": excluded, "n": n,
            "residual_shapiro_p": sh_p, "levene_p": lev_p,
            "classic_F": float(F), "classic_p": float(p_classic), "eta_squared": float(eta2),
            "welch_F": Fw, "welch_p": p_welch,
            "kruskal_H": float(H), "kruskal_p": float(p_kw), "epsilon_squared": float(eps2),
            "primary_test": primary, "primary_p": float(pprim),
            "independence_note": "Districts are nested within states and may be spatially clustered, so the "
                                 "independence assumption is only approximately satisfied; treat p-values cautiously.",
            "interpretation": f"Primary: {primary}, p={pprim:.4g} (alpha={alpha}). "
                              "A significant result shows that at least one group differs, not which one, and not why."}
