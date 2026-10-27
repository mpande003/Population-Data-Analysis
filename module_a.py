"""Module A - population pressure analysis (descriptive)."""
from __future__ import annotations
import pandas as pd


def describe_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    return pd.Series({"count": s.size, "sum": s.sum(), "mean": s.mean(), "median": s.median(),
                      "std": s.std(), "min": s.min(), "Q1": q1, "Q3": q3, "IQR": q3 - q1, "max": s.max()})


def population_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Total_Population", "Child_Population", "Child_Population_Share"]
    return pd.DataFrame({c: describe_series(df[c]) for c in cols})


def top_n(df: pd.DataFrame, col: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    keep = ["District_Label", "state_name", col]
    return df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)[keep].reset_index(drop=True)


def state_population(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("state_name").agg(Districts=("District_Label", "count"),
                                     Total_Population=("Total_Population", "sum"),
                                     Child_Population=("Child_Population", "sum"))
    g["Child_Population_Share"] = g["Child_Population"] / g["Total_Population"] * 100
    g["Share_of_Included_Population_%"] = g["Total_Population"] / g["Total_Population"].sum() * 100
    return g.sort_values("Total_Population", ascending=False).reset_index()


def concentration(df: pd.DataFrame, k: int = 10) -> dict:
    tot = df["Total_Population"].sum()
    top = df.nlargest(k, "Total_Population")["Total_Population"].sum()
    return {"top_k": k, "top_k_share_%": top / tot * 100, "total_population": tot}
