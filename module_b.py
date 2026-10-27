"""Module B - education and social development indicators."""
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
    """RQ: Do districts with an above-median SC+ST share differ in literacy rate from the rest?
    H0: the distributions of Literacy_Rate are equal. H1: they differ (two-sided)."""
    d = df.copy()
    share = d["SC_Population_Share"] + d["ST_Population_Share"]
    d["SC_ST_Group"] = np.where(share > share.median(), "Above-median SC+ST share", "At/below-median SC+ST share")
    res = st.two_group_test(d, "Literacy_Rate", "SC_ST_Group")
    res["research_question"] = "Do districts with an above-median SC+ST share differ in Literacy_Rate from other districts?"
    res["H0"] = "Literacy_Rate has the same distribution in both groups."
    res["H1"] = "Literacy_Rate differs between the groups (two-sided)."
    return res


def state_anova(df: pd.DataFrame, min_group_size: int = 5) -> dict:
    """Compare district literacy across STATES with at least `min_group_size` districts."""
    return st.one_way_anova(df, "Literacy_Rate", "state_name", min_group_size)
