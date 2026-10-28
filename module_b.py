"""Module B - education and social development indicators and syllabus-approved statistical tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statistical_tests as st

B_VARS = ["Literacy_Rate", "Sex_Ratio", "Child_Population_Share", "SC_Population_Share",
          "ST_Population_Share", "Workforce_Participation_Rate"]


def literacy_summary(df: pd.DataFrame) -> pd.DataFrame:
    from module_a import describe_series
    return pd.DataFrame({c: describe_series(df[c]) for c in B_VARS if c in df})


def rank_districts(df: pd.DataFrame, col: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    return df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)[["District_Label", "state_name", col]].reset_index(drop=True)


def state_literacy(df: pd.DataFrame, basis: str = "source") -> pd.DataFrame:
    """Population-weighted state literacy from numerators/denominators.
    basis='source'   : sum(literate) / sum(total population)  (matches the dataset's crude definition)
    basis='effective': sum(literate) / sum(total population - age 0-6)
    Also returns the unweighted mean of district rates for comparison (labelled as such)."""
    g = df.groupby("state_name")
    out = g.agg(Districts=("District_Label", "count"), Literate=("literate_total", "sum"),
                Population=("pop_total", "sum"), Child=("pop_under6_total", "sum"),
                Unweighted_Mean_District_Literacy=("Literacy_Rate", "mean"))
    denom = out["Population"] if basis == "source" else out["Population"] - out["Child"]
    out["Weighted_Literacy_Rate"] = out["Literate"] / denom * 100
    return out.drop(columns=["Child"]).sort_values("Weighted_Literacy_Rate", ascending=False).reset_index()


def sc_st_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    d = df.assign(SC_ST_Share=df["SC_Population_Share"] + df["ST_Population_Share"])
    return d.sort_values("SC_ST_Share", ascending=False).head(n)[["District_Label", "state_name", "SC_Population_Share", "ST_Population_Share", "SC_ST_Share"]].reset_index(drop=True)


def default_hypothesis(df: pd.DataFrame) -> dict:
    """RQ: Do districts with an above-median SC+ST share differ in mean literacy rate from the rest?
    Syllabus-approved test: Two-Sample t-test / Welch's t-test (Practical 5)."""
    d = df.copy()
    share = d["SC_Population_Share"] + d["ST_Population_Share"]
    d["SC_ST_Group"] = np.where(share > share.median(), "Above-median SC+ST share", "At/below-median SC+ST share")
    res = st.two_sample_t_test(d, "Literacy_Rate", "SC_ST_Group")
    res["research_question"] = "Do districts with an above-median SC+ST share differ in mean Literacy_Rate from other districts?"
    res["H0"] = "Mean district Literacy_Rate is equal in both groups (mu_above = mu_below)."
    res["H1"] = "Mean district Literacy_Rate differs between the two groups (mu_above != mu_below, two-sided)."
    return res


def state_anova(df: pd.DataFrame, min_group_size: int = 5) -> dict:
    """Compare district literacy across STATES with at least `min_group_size` districts.
    Syllabus-approved test: One-Way ANOVA with assumption checks (Practical 5 — Notebook 2)."""
    return st.one_way_anova(df, "Literacy_Rate", "state_name", min_group_size)


def two_way_anova_literacy(df: pd.DataFrame) -> dict:
    """Two-Way ANOVA: Literacy Rate across SC+ST Concentration Tier and Workforce Participation Tier.
    Syllabus-approved test: Two-Way ANOVA (Practical 5 — Notebook 2)."""
    d = df.copy()
    share = d["SC_Population_Share"] + d["ST_Population_Share"]
    d["SC_ST_Tier"] = np.where(share > share.median(), "High SC+ST Share", "Low SC+ST Share")
    d["Workforce_Tier"] = np.where(d["Workforce_Participation_Rate"] > d["Workforce_Participation_Rate"].median(), "High Workforce", "Low Workforce")
    return st.two_way_anova(d, "Literacy_Rate", "SC_ST_Tier", "Workforce_Tier")


def chi_square_literacy_sc_st(df: pd.DataFrame, low: float = 60.0, high: float = 75.0) -> dict:
    """Chi-Square Test of Independence between Literacy Tier and SC+ST Share Group.
    Syllabus-approved test: Chi-square test (Practical 5 — Notebook 2)."""
    d = df.copy()
    share = d["SC_Population_Share"] + d["ST_Population_Share"]
    d["SC_ST_Tier"] = np.where(share > share.median(), "Above-median SC+ST", "At/below-median SC+ST")
    lit = d["Literacy_Rate"]
    d["Literacy_Tier"] = pd.Categorical(
        np.select([lit < low, lit < high, lit >= high], ["Low", "Medium", "High"], default="Low"),
        categories=["Low", "Medium", "High"], ordered=True
    )
    return st.chi_square_test(d, "Literacy_Tier", "SC_ST_Tier")


def one_sample_test_literacy(df: pd.DataFrame, test_type: str = "t", benchmark: float = 65.0) -> dict:
    """One-sample T-test or Z-test comparing sample mean literacy to a benchmark (Practical 5 — Notebook 1)."""
    if test_type.lower() == "z":
        return st.one_sample_z_test(df, "Literacy_Rate", benchmark)
    return st.one_sample_t_test(df, "Literacy_Rate", benchmark)
