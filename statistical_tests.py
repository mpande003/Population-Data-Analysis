"""Statistical tests strictly conforming to the SI Lab Practical Syllabus.

Allowed tests (Practical 5, Notebooks 1 & 2):
- Z-test (One-sample, Two-sample for independent groups)
- T-test (One-sample, Two-sample / Welch's t-test)
- Chi-square test of independence
- One-way ANOVA
- Two-way ANOVA
- Related measures: means, standard errors, confidence intervals, Cohen's d, eta-squared, Cramer's V

Non-syllabus non-parametric tests (Mann-Whitney U, Kruskal-Wallis) are strictly excluded.
When assumptions are violated, warnings and limitations are reported without substituting unlisted tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

ALPHA = 0.05


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].dropna().corr(method=method)


def pair_correlation(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    d = df[[x, y]].dropna()
    pr, pp = stats.pearsonr(d[x], d[y])
    sr, sp = stats.spearmanr(d[x], d[y])
    return pd.DataFrame({"Method": ["Pearson", "Spearman"], "Coefficient": [pr, sr], "P-value": [pp, sp], "n": len(d)})


def shapiro(x) -> float:
    """Normality assumption check via Shapiro-Wilk test."""
    x = pd.Series(x).dropna()
    return float(stats.shapiro(x).pvalue) if 3 <= len(x) <= 5000 else float("nan")


def levene_test(*groups) -> float:
    """Homogeneity of variance check via Levene's test (centered at median)."""
    clean_groups = [np.asarray(g[~np.isnan(g)]) for g in groups if len(g) > 1]
    if len(clean_groups) < 2:
        return float("nan")
    return float(stats.levene(*clean_groups, center="median").pvalue)


# -----------------------------------------------------------------------------
# 1. Two-Sample T-Test / Welch's T-Test (Practical 5 — Notebooks 1 & 2)
# -----------------------------------------------------------------------------
def two_sample_t_test(df: pd.DataFrame, value: str, group_col: str, alpha: float = ALPHA) -> dict:
    """Independent Two-Sample t-test / Welch's t-test comparing group means.

    H0: mu_1 = mu_2 (Group means are equal)
    H1: mu_1 != mu_2 (Group means differ, two-sided)

    Assumption checks:
    - Normality: Shapiro-Wilk test on each group.
    - Homogeneity of variance: Levene's test.
    If equal variance fails, Welch's t-test is used (degrees of freedom adjusted).
    If normality fails, a warning is raised and limitations are reported without switching to unlisted tests.
    """
    d = df[[value, group_col]].dropna()
    levels = sorted(d[group_col].unique())
    if len(levels) != 2:
        raise ValueError(f"group_col must have exactly two levels, found {len(levels)}: {levels}")

    a = d.loc[d[group_col] == levels[0], value].to_numpy(dtype=float)
    b = d.loc[d[group_col] == levels[1], value].to_numpy(dtype=float)
    n1, n2 = len(a), len(b)
    if min(n1, n2) < 5:
        raise ValueError(f"Each group needs at least 5 observations (got {n1}, {n2})")

    # Assumption checks
    sh_p1 = shapiro(a)
    sh_p2 = shapiro(b)
    lev_p = levene_test(a, b)

    warnings = []
    if sh_p1 < alpha or sh_p2 < alpha:
        warnings.append(
            f"Normality assumption violated: Shapiro-Wilk p={sh_p1:.4e} ({levels[0]}), p={sh_p2:.4e} ({levels[1]}). "
            f"With large sample sizes (n1={n1}, n2={n2}), the t-test is asymptotically robust via the Central Limit Theorem, "
            f"but p-values should be interpreted with caution. (Note: Non-parametric tests are excluded per syllabus constraints.)"
        )
    equal_var = lev_p >= alpha
    if not equal_var:
        warnings.append(
            f"Equal variance assumption violated: Levene's p={lev_p:.4e} < {alpha}. "
            "Welch's t-test (Satterthwaite approximation) is applied to handle unequal group variances."
        )

    # Perform t-test (Student's if equal_var, Welch's if unequal)
    test_name = "Two-Sample Student's t-test" if equal_var else "Two-Sample Welch's t-test"
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df_deg = float(res.df) if hasattr(res, "df") else float(n1 + n2 - 2)

    # Group statistics
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    med1, med2 = float(np.median(a)), float(np.median(b))
    mean_diff = m1 - m2

    # Standard error of difference and 95% Confidence Interval
    if equal_var:
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        se_diff = sp * np.sqrt(1 / n1 + 1 / n2)
        cohens_d = (m1 - m2) / sp if sp > 0 else 0.0
    else:
        se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        sp = np.sqrt((s1**2 + s2**2) / 2)
        cohens_d = (m1 - m2) / sp if sp > 0 else 0.0

    t_crit = float(stats.t.ppf(1 - alpha / 2, df_deg))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    decision = "reject H0" if p_val < alpha else "fail to reject H0"
    interp = (
        f"{test_name}: {decision} at alpha={alpha} (t={t_stat:.3f}, df={df_deg:.1f}, p={p_val:.4e}). "
        f"Group '{levels[0]}' mean={m1:.2f} vs Group '{levels[1]}' mean={m2:.2f} (mean diff = {mean_diff:.2f}, "
        f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}], Cohen's d={cohens_d:.3f}). "
        "This indicates a statistically significant difference between the groups, but does NOT establish causation."
    )

    return {
        "test": test_name,
        "purpose": "Compare the population means of two independent groups.",
        "groups": levels,
        "n": (n1, n2),
        "means": (m1, m2),
        "stds": (s1, s2),
        "medians": (med1, med2),
        "mean_difference": mean_diff,
        "ci_95": (ci_lower, ci_upper),
        "statistic": t_stat,
        "df": df_deg,
        "p_value": p_val,
        "alpha": alpha,
        "effect_size": cohens_d,
        "effect_size_name": "Cohen's d",
        "normality_p": (sh_p1, sh_p2),
        "levene_p": lev_p,
        "assumption_warnings": warnings,
        "decision": decision,
        "interpretation": interp,
    }


# Backward-compatible alias
two_group_test = two_sample_t_test


# -----------------------------------------------------------------------------
# 2. One-Sample T-Test (Practical 5 — Notebook 1)
# -----------------------------------------------------------------------------
def one_sample_t_test(df: pd.DataFrame, value: str, test_value: float, alpha: float = ALPHA) -> dict:
    """One-Sample t-test comparing sample mean to a specified reference value.

    H0: mu = mu_0 (Sample mean equals reference benchmark)
    H1: mu != mu_0 (Sample mean differs from reference benchmark)
    """
    x = df[value].dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        raise ValueError(f"Need at least 5 observations, got {n}")

    sh_p = shapiro(x)
    warnings = []
    if sh_p < alpha:
        warnings.append(
            f"Normality assumption violated: Shapiro-Wilk p={sh_p:.4e} < {alpha}. "
            "For n >= 30, the t-test remains robust due to the Central Limit Theorem."
        )

    res = stats.ttest_1samp(x, test_value)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df_deg = int(res.df) if hasattr(res, "df") else n - 1

    sample_mean = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1))
    se = sample_std / np.sqrt(n)
    cohens_d = (sample_mean - test_value) / sample_std if sample_std > 0 else 0.0

    t_crit = float(stats.t.ppf(1 - alpha / 2, df_deg))
    ci_lower = sample_mean - t_crit * se
    ci_upper = sample_mean + t_crit * se

    decision = "reject H0" if p_val < alpha else "fail to reject H0"
    interp = (
        f"One-Sample t-test: {decision} at alpha={alpha} (t={t_stat:.3f}, df={df_deg}, p={p_val:.4e}). "
        f"Sample mean = {sample_mean:.2f} vs benchmark = {test_value:.2f} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}], Cohen's d={cohens_d:.3f})."
    )

    return {
        "test": "One-Sample t-test",
        "purpose": "Compare sample mean against a known reference benchmark mu_0.",
        "n": n,
        "sample_mean": sample_mean,
        "test_value": test_value,
        "mean_diff": sample_mean - test_value,
        "sample_std": sample_std,
        "std_error": se,
        "ci_95": (ci_lower, ci_upper),
        "statistic": t_stat,
        "df": df_deg,
        "p_value": p_val,
        "alpha": alpha,
        "effect_size": cohens_d,
        "effect_size_name": "Cohen's d",
        "normality_p": sh_p,
        "assumption_warnings": warnings,
        "decision": decision,
        "interpretation": interp,
    }


# -----------------------------------------------------------------------------
# 3. One-Sample and Two-Sample Z-Tests (Practical 5 — Notebook 1)
# -----------------------------------------------------------------------------
def one_sample_z_test(df: pd.DataFrame, value: str, test_value: float, pop_std: float | None = None, alpha: float = ALPHA) -> dict:
    """One-Sample Z-test for large samples (n >= 30) or known population standard deviation.

    H0: mu = mu_0
    H1: mu != mu_0
    """
    x = df[value].dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 30 and pop_std is None:
        raise ValueError(f"One-sample Z-test requires large sample (n >= 30) when population std is unknown (got n={n})")

    sample_mean = float(np.mean(x))
    std = pop_std if pop_std is not None else float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    z_stat = (sample_mean - test_value) / se
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = sample_mean - z_crit * se
    ci_upper = sample_mean + z_crit * se

    decision = "reject H0" if p_val < alpha else "fail to reject H0"
    interp = (
        f"One-Sample Z-test: {decision} at alpha={alpha} (Z={z_stat:.3f}, p={p_val:.4e}). "
        f"Sample mean = {sample_mean:.2f} vs benchmark = {test_value:.2f} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}])."
    )

    return {
        "test": "One-Sample Z-test",
        "purpose": "Test hypothesis about population mean when sample size is large (Central Limit Theorem).",
        "n": n,
        "sample_mean": sample_mean,
        "test_value": test_value,
        "mean_diff": sample_mean - test_value,
        "std_used": std,
        "std_error": se,
        "ci_95": (ci_lower, ci_upper),
        "statistic": float(z_stat),
        "p_value": p_val,
        "alpha": alpha,
        "assumption_warnings": [],
        "decision": decision,
        "interpretation": interp,
    }


def two_sample_z_test(df: pd.DataFrame, value: str, group_col: str, alpha: float = ALPHA) -> dict:
    """Two-Sample Z-test for large independent samples (n1, n2 >= 30).

    H0: mu_1 = mu_2
    H1: mu_1 != mu_2
    """
    d = df[[value, group_col]].dropna()
    levels = sorted(d[group_col].unique())
    if len(levels) != 2:
        raise ValueError("group_col must have exactly two levels")

    a = d.loc[d[group_col] == levels[0], value].to_numpy(dtype=float)
    b = d.loc[d[group_col] == levels[1], value].to_numpy(dtype=float)
    n1, n2 = len(a), len(b)
    if min(n1, n2) < 30:
        raise ValueError(f"Two-sample Z-test requires large samples (n1, n2 >= 30, got {n1}, {n2})")

    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    mean_diff = m1 - m2
    se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    z_stat = mean_diff / se_diff
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = mean_diff - z_crit * se_diff
    ci_upper = mean_diff + z_crit * se_diff

    decision = "reject H0" if p_val < alpha else "fail to reject H0"
    interp = (
        f"Two-Sample Z-test: {decision} at alpha={alpha} (Z={z_stat:.3f}, p={p_val:.4e}). "
        f"Mean diff = {mean_diff:.2f} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}])."
    )

    return {
        "test": "Two-Sample Z-test",
        "purpose": "Compare means between two large independent groups using normal approximation.",
        "groups": levels,
        "n": (n1, n2),
        "means": (m1, m2),
        "mean_diff": mean_diff,
        "std_error": float(se_diff),
        "ci_95": (ci_lower, ci_upper),
        "statistic": float(z_stat),
        "p_value": p_val,
        "alpha": alpha,
        "decision": decision,
        "interpretation": interp,
    }


# -----------------------------------------------------------------------------
# 4. One-Way ANOVA (Practical 5 — Notebook 2)
# -----------------------------------------------------------------------------
def welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Welch's ANOVA for unequal group variances. Returns (F_welch, df2, p_value)."""
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
    return float(F), float(df2), float(stats.f.sf(F, k - 1, df2))


def one_way_anova(df: pd.DataFrame, value: str, group_col: str, min_group_size: int = 5, alpha: float = ALPHA) -> dict:
    """One-Way ANOVA testing for equality of group means across 3+ groups.

    H0: mu_1 = mu_2 = ... = mu_k (All group population means are equal)
    H1: At least one group population mean differs

    Assumptions checked:
    - Residual normality: Shapiro-Wilk test on ANOVA residuals.
    - Equal variances: Levene's test across groups.
    If assumptions are violated, warnings are issued and Welch's ANOVA is reported.
    No unapproved non-parametric test (e.g. Kruskal-Wallis) is substituted.
    """
    d = df[[value, group_col]].dropna()
    sizes = d.groupby(group_col)[value].size()
    keep = sizes[sizes >= min_group_size].index
    excluded = sizes[sizes < min_group_size].index.tolist()
    d = d[d[group_col].isin(keep)]
    names = sorted(d[group_col].unique())
    if len(names) < 3:
        raise ValueError(f"Need at least 3 groups of adequate size for a one-way ANOVA (found {len(names)})")

    groups = [d.loc[d[group_col] == g, value].to_numpy(dtype=float) for g in names]
    n, k = len(d), len(names)

    # Residuals & Assumption Checks
    resid = np.concatenate([g - g.mean() for g in groups])
    sh_p = shapiro(resid)
    lev_p = levene_test(*groups)

    warnings = []
    if sh_p < alpha:
        warnings.append(
            f"Residual normality assumption violated: Shapiro-Wilk p={sh_p:.4e} < {alpha}. "
            "ANOVA is moderately robust to mild departures from normality with large N, but results should be interpreted cautiously."
        )
    if lev_p < alpha:
        warnings.append(
            f"Homogeneity of variance violated: Levene's p={lev_p:.4e} < {alpha}. "
            "Welch's ANOVA is reported as the variance-robust parametric alternative."
        )

    # Classic ANOVA
    F_classic, p_classic = stats.f_oneway(*groups)
    grand = float(d[value].mean())
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((d[value] - grand) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

    # Welch's ANOVA
    F_welch, df2_welch, p_welch = welch_anova(groups)

    primary_test = "One-Way ANOVA (Classic)" if lev_p >= alpha else "One-Way ANOVA (Welch's robust test)"
    primary_p = float(p_classic if lev_p >= alpha else p_welch)
    decision = "reject H0" if primary_p < alpha else "fail to reject H0"

    interp = (
        f"{primary_test}: {decision} at alpha={alpha} (F={float(F_classic):.3f}, p={float(p_classic):.4e}, "
        f"Welch F={F_welch:.3f}, Welch p={p_welch:.4e}, eta_squared={eta2:.3f}). "
        "A significant result demonstrates that at least one group mean differs from the others. "
        "It does not identify which specific group differs and does not establish a causal relationship."
    )

    return {
        "test": "One-Way ANOVA",
        "purpose": "Test whether the population means of three or more independent groups are equal.",
        "groups_used": k,
        "groups_excluded_small": excluded,
        "n": n,
        "df_between": k - 1,
        "df_within": n - k,
        "classic_F": float(F_classic),
        "classic_p": float(p_classic),
        "eta_squared": eta2,
        "welch_F": float(F_welch),
        "welch_df2": float(df2_welch),
        "welch_p": float(p_welch),
        "residual_shapiro_p": sh_p,
        "levene_p": lev_p,
        "primary_test": primary_test,
        "primary_p": primary_p,
        "alpha": alpha,
        "assumption_warnings": warnings,
        "independence_note": (
            "Districts are nested within states and may exhibit spatial auto-correlation; "
            "the independent observation assumption is only approximately satisfied."
        ),
        "decision": decision,
        "interpretation": interp,
    }


# -----------------------------------------------------------------------------
# 5. Two-Way ANOVA (Practical 5 — Notebook 2)
# -----------------------------------------------------------------------------
def two_way_anova(df: pd.DataFrame, value: str, factor1: str, factor2: str, alpha: float = ALPHA) -> dict:
    """Two-Way ANOVA testing for main effects and interaction effect.

    H0_A: No main effect of Factor 1
    H0_B: No main effect of Factor 2
    H0_AB: No interaction effect between Factor 1 and Factor 2
    """
    d = df[[value, factor1, factor2]].dropna().copy()
    d[factor1] = d[factor1].astype(str)
    d[factor2] = d[factor2].astype(str)

    formula = f"Q('{value}') ~ C(Q('{factor1}')) + C(Q('{factor2}')) + C(Q('{factor1}')):C(Q('{factor2}'))"
    model = ols(formula, data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)

    # Clean up index names
    f1_key = f"C(Q('{factor1}'))"
    f2_key = f"C(Q('{factor2}'))"
    inter_key = f"C(Q('{factor1}')):C(Q('{factor2}'))"

    resid = model.resid
    sh_p = shapiro(resid)
    warnings = []
    if sh_p < alpha:
        warnings.append(f"Residual normality violated: Shapiro-Wilk p={sh_p:.4e} < {alpha}.")

    ss_total = table["sum_sq"].sum()
    results_summary = []
    for factor_label, key in [(factor1, f1_key), (factor2, f2_key), (f"{factor1} x {factor2} (Interaction)", inter_key)]:
        if key in table.index:
            row = table.loc[key]
            ss = float(row["sum_sq"])
            df_val = float(row["df"])
            f_val = float(row["F"])
            p_val = float(row["PR(>F)"])
            part_eta = ss / (ss + float(table.loc["Residual", "sum_sq"])) if (ss + float(table.loc["Residual", "sum_sq"])) > 0 else 0.0
            dec = "reject H0" if p_val < alpha else "fail to reject H0"
            results_summary.append({
                "Source": factor_label,
                "Sum_of_Squares": ss,
                "df": df_val,
                "F_statistic": f_val,
                "p_value": p_val,
                "partial_eta_squared": part_eta,
                "Decision": dec,
            })

    res_df = pd.DataFrame(results_summary)
    interp = (
        f"Two-Way ANOVA on '{value}' across '{factor1}' and '{factor2}': "
        + "; ".join([f"{r['Source']} (F={r['F_statistic']:.2f}, p={r['p_value']:.4e}, {r['Decision']})" for r in results_summary])
        + ". Results indicate statistical associations, not causal links."
    )

    return {
        "test": "Two-Way ANOVA",
        "purpose": "Evaluate main effects and interaction between two categorical factors on a continuous outcome.",
        "factor1": factor1,
        "factor2": factor2,
        "value": value,
        "n": len(d),
        "anova_table": res_df,
        "residual_shapiro_p": sh_p,
        "assumption_warnings": warnings,
        "interpretation": interp,
    }


# -----------------------------------------------------------------------------
# 6. Chi-Square Test of Independence (Practical 5 — Notebook 2)
# -----------------------------------------------------------------------------
def chi_square_test(df: pd.DataFrame, cat_col1: str, cat_col2: str, alpha: float = ALPHA) -> dict:
    """Chi-Square Test of Independence for two categorical variables.

    H0: cat_col1 and cat_col2 are independent.
    H1: cat_col1 and cat_col2 are dependent (associated).
    """
    d = df[[cat_col1, cat_col2]].dropna()
    ct = pd.crosstab(d[cat_col1], d[cat_col2])
    res = stats.chi2_contingency(ct)

    chi2_stat = float(res.statistic)
    p_val = float(res.pvalue)
    dof = int(res.dof)
    expected = pd.DataFrame(res.expected_freq, index=ct.index, columns=ct.columns)

    # Cochran's assumption check: expected counts >= 5
    sparse_cells = int((expected < 5).sum().sum())
    total_cells = int(expected.size)
    warnings = []
    if sparse_cells > 0:
        warnings.append(
            f"Sparse cells warning: {sparse_cells} out of {total_cells} expected cell counts are < 5. "
            "Chi-square approximation may be less reliable."
        )

    # Effect size: Cramer's V
    n = float(ct.to_numpy().sum())
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) if (n > 0 and min_dim > 0) else 0.0

    decision = "reject H0" if p_val < alpha else "fail to reject H0"
    interp = (
        f"Chi-Square Test of Independence: {decision} at alpha={alpha} (chi2={chi2_stat:.3f}, df={dof}, p={p_val:.4e}, "
        f"Cramer's V={cramers_v:.3f}). "
        f"There is a statistically significant association between '{cat_col1}' and '{cat_col2}'."
        if p_val < alpha else
        f"Chi-Square Test of Independence: {decision} at alpha={alpha} (chi2={chi2_stat:.3f}, df={dof}, p={p_val:.4e}). "
        f"No statistically significant association observed between '{cat_col1}' and '{cat_col2}'."
    )

    return {
        "test": "Chi-Square Test of Independence",
        "purpose": "Test whether two categorical variables are statistically independent.",
        "variable_1": cat_col1,
        "variable_2": cat_col2,
        "contingency_table": ct,
        "expected_frequencies": expected,
        "statistic": chi2_stat,
        "df": dof,
        "p_value": p_val,
        "alpha": alpha,
        "cramers_v": cramers_v,
        "sparse_cells": sparse_cells,
        "assumption_warnings": warnings,
        "decision": decision,
        "interpretation": interp,
    }
