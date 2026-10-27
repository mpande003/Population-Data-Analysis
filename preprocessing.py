"""Cleaning, schema mapping and derived variables (shared by notebook and app)."""
from __future__ import annotations
import numpy as np
import pandas as pd

# Conceptual variable -> source column (verified against the repository schema)
SOURCE_MAP = {
    "State": "state_name", "State_Code": "state_code",
    "District": "district_name", "District_Code": "district_code",
    "Total_Population": "pop_total", "Male_Population": "pop_male", "Female_Population": "pop_female",
    "Child_Population": "pop_under6_total",   # Census age group 0-6 years
    "SC_Population": "pop_sc", "ST_Population": "pop_st",
    "Literate_Total": "literate_total", "Illiterate_Total": "illiterate_total",
    "Workers_Total": "workers_total", "Non_Workers_Total": "non_workers_total",
    "Source_Literacy_Rate": "literacy_rate", "Sex_Ratio": "sex_ratio",
    "Workforce_Participation_Rate": "workforce_participation",
}
COUNT_COLS = ["pop_total", "pop_male", "pop_female", "pop_under6_total", "pop_sc", "pop_st",
              "literate_total", "illiterate_total", "workers_total", "non_workers_total"]
RATE_COLS = ["literacy_rate", "sex_ratio", "workforce_participation"]

DERIVED = {
    "Child_Population_Share": "pop_under6_total / pop_total * 100 (child = age 0-6)",
    "SC_Population_Share": "pop_sc / pop_total * 100",
    "ST_Population_Share": "pop_st / pop_total * 100",
    "Literacy_Rate": "Analysis literacy rate. 'source' = dataset column literacy_rate; "
                     "'effective' = literate_total / (pop_total - pop_under6_total) * 100",
    "Literacy_Rate_Source": "dataset column literacy_rate (unchanged)",
    "Literacy_Rate_Crude_Check": "literate_total / pop_total * 100 (used to verify the source denominator)",
    "Literacy_Rate_Effective": "literate_total / (pop_total - pop_under6_total) * 100",
    "Literacy_Gap": "100 - Literacy_Rate",
    "Sex_Ratio": "dataset column sex_ratio (females per 1000 males, verified vs pop_female/pop_male*1000)",
    "Workforce_Participation_Rate": "dataset column workforce_participation "
                                    "(verified vs workers_total / pop_total * 100)",
    "Child_Sex_Ratio": "pop_under6_female / pop_under6_male * 1000 (females per 1000 males, age 0-6)",
    "Female_Workforce_Rate": "workers_female / pop_female * 100 (female workforce participation %)",
    "Dependency_Ratio": "(pop_under6_total + non_workers_total) / workers_total (dependents per worker)",
    "Log_Population": "log10(pop_total) (logarithmic population scale)",
}


def schema_report(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept, col in SOURCE_MAP.items():
        rows.append({"Conceptual variable": concept, "Source column": col,
                     "Present": col in raw.columns,
                     "Dtype": str(raw[col].dtype) if col in raw.columns else "-"})
    return pd.DataFrame(rows)


def module_availability(raw: pd.DataFrame) -> pd.DataFrame:
    has = lambda *c: all(x in raw.columns for x in c)
    req = {
        "A: Population analysis": has("state_name", "district_name", "pop_total", "pop_under6_total"),
        "B: Education & social": has("literacy_rate", "sex_ratio", "pop_sc", "pop_st", "workforce_participation", "pop_under6_total"),
        "C: Literacy classification": has("literacy_rate", "sex_ratio", "pop_under6_total", "pop_sc", "pop_st", "workforce_participation", "pop_total"),
        "D: UIPS": has("pop_total", "pop_under6_total", "literacy_rate", "workforce_participation"),
    }
    return pd.DataFrame({"Module": list(req), "All required fields present": list(req.values())})


def data_dictionary(raw: pd.DataFrame) -> pd.DataFrame:
    inv = {v: k for k, v in SOURCE_MAP.items()}
    return pd.DataFrame({"Source column": raw.columns, "Dtype": [str(t) for t in raw.dtypes],
                         "Mapped concept": [inv.get(c, "(not used / identifier)") for c in raw.columns]})


def clean_data(raw: pd.DataFrame, literacy_basis: str = "source"):
    """Return (clean_df, quality_report). The raw DataFrame is never modified.

    literacy_basis: 'source' (dataset literacy_rate, default) or 'effective' (7+ population).
    Column names of the working copy are standardised; SOURCE_MAP keeps the link to original names.
    """
    if literacy_basis not in ("source", "effective"):
        raise ValueError("literacy_basis must be 'source' or 'effective'")
    missing = [c for c in COUNT_COLS + RATE_COLS + ["state_name", "district_name"] if c not in raw.columns]
    q: dict = {"missing_source_columns": missing, "notes": []}
    df = raw.copy(deep=True)
    for c in COUNT_COLS + RATE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    q["missing_values"] = raw.isna().sum()[lambda s: s > 0].to_dict()
    q["duplicate_rows"] = int(raw.duplicated().sum())
    q["duplicate_district_codes"] = int(raw["district_code"].duplicated().sum()) if "district_code" in raw else None
    if "state_name" in df:
        df["state_name"] = df["state_name"].astype(str).str.strip()
    if "district_name" in df:
        df["district_name"] = df["district_name"].astype(str).str.strip()
    # unique label (district names repeat across states)
    df["District_Label"] = df["district_name"] + " (" + df["state_name"] + ")"
    dup = df["District_Label"].duplicated(keep=False)
    q["labels_made_unique"] = int(dup.sum())
    if dup.any() and "district_code" in df:  # keep labels unique so tables never silently collide
        df.loc[dup, "District_Label"] += " #" + df.loc[dup, "district_code"].astype(str)

    def n_bad(mask): return int(mask.fillna(False).sum())
    inv = {}
    inv["non_positive_population"] = n_bad(~(df["pop_total"] > 0))
    inv["negative_counts"] = n_bad((df[[c for c in COUNT_COLS if c in df]] < 0).any(axis=1))
    inv["child_pop_exceeds_total"] = n_bad(df["pop_under6_total"] > df["pop_total"])
    inv["sc_st_exceed_total"] = n_bad((df["pop_sc"] > df["pop_total"]) | (df["pop_st"] > df["pop_total"]))
    inv["male_plus_female_ne_total"] = n_bad((df["pop_male"] + df["pop_female"]) != df["pop_total"])
    inv["literate_plus_illiterate_ne_total"] = n_bad((df["literate_total"] + df["illiterate_total"]) != df["pop_total"])
    inv["workers_plus_nonworkers_ne_total"] = n_bad((df["workers_total"] + df["non_workers_total"]) != df["pop_total"])
    inv["rates_outside_0_100"] = n_bad((df["literacy_rate"] < 0) | (df["literacy_rate"] > 100)
                                       | (df["workforce_participation"] < 0) | (df["workforce_participation"] > 100))
    q["invalid_counts"] = inv

    # denominator verification (median absolute difference, percentage points)
    crude = df["literate_total"] / df["pop_total"] * 100
    eff = df["literate_total"] / (df["pop_total"] - df["pop_under6_total"]) * 100
    wfp = df["workers_total"] / df["pop_total"] * 100
    sr = df["pop_female"] / df["pop_male"] * 1000
    q["denominator_checks"] = {
        "literacy_rate vs literate/pop_total (median abs diff, pp)": float((df["literacy_rate"] - crude).abs().median()),
        "literacy_rate vs literate/(pop_total-under6) (median abs diff, pp)": float((df["literacy_rate"] - eff).abs().median()),
        "workforce_participation vs workers/pop_total (median abs diff, pp)": float((df["workforce_participation"] - wfp).abs().median()),
        "sex_ratio vs female/male*1000 (median abs diff)": float((df["sex_ratio"] - sr).abs().median()),
    }
    d = q["denominator_checks"]
    k1, k2 = list(d)[0], list(d)[1]
    if d[k1] < d[k2]:
        q["notes"].append("The source literacy_rate matches literate_total / TOTAL population (all ages), i.e. a CRUDE "
                          "literacy rate, not the Census 'effective' rate (age 7+). Child population (age 0-6) therefore "
                          "mechanically lowers it. Use literacy_basis='effective' for the age 7+ rate.")
    else:
        q["notes"].append("The source literacy_rate matches literate_total / (total - age 0-6 population), i.e. an effective rate.")
    q["notes"].append("Child population = age group 0-6 years (columns pop_under6_*).")
    q["notes"].append("Sex ratio = females per 1000 males. Workforce participation = workers / total population * 100.")

    # name consistency (informational; no merge on names is performed)
    if "district_name_lgd" in df:
        q["district_name_vs_lgd_mismatch"] = int((df["district_name"].str.lower() != df["district_name_lgd"].astype(str).str.lower().str.strip()).sum())
        q["notes"].append("district_name and district_name_lgd differ in spelling for some rows; district_code is used as the key, never names.")
    q["district_names_repeated_across_states"] = int(df["district_name"].duplicated(keep=False).sum())

    # exclusions
    bad = ~(df["pop_total"] > 0) | df["pop_total"].isna()
    q["excluded_rows"] = int(bad.sum())
    df = df.loc[~bad].copy()

    # derived variables (invalid denominators -> NaN, never inf)
    pt = df["pop_total"].where(df["pop_total"] > 0)
    df["Total_Population"] = df["pop_total"]
    df["Child_Population"] = df["pop_under6_total"]
    df["Child_Population_Share"] = df["pop_under6_total"] / pt * 100
    df["SC_Population_Share"] = df["pop_sc"] / pt * 100
    df["ST_Population_Share"] = df["pop_st"] / pt * 100
    df["Literacy_Rate_Source"] = df["literacy_rate"]
    df["Literacy_Rate_Crude_Check"] = df["literate_total"] / pt * 100
    den = (df["pop_total"] - df["pop_under6_total"]).where(lambda s: s > 0)
    df["Literacy_Rate_Effective"] = df["literate_total"] / den * 100
    df["Literacy_Rate"] = df["Literacy_Rate_Source"] if literacy_basis == "source" else df["Literacy_Rate_Effective"]
    df["Literacy_Gap"] = 100 - df["Literacy_Rate"]
    df["Sex_Ratio"] = df["sex_ratio"]
    df["Workforce_Participation_Rate"] = df["workforce_participation"]
    # Enhanced non-literacy demographic features (zero target leakage)
    if "pop_under6_female" in df and "pop_under6_male" in df:
        den_u6m = df["pop_under6_male"].where(df["pop_under6_male"] > 0)
        df["Child_Sex_Ratio"] = df["pop_under6_female"] / den_u6m * 1000
    if "workers_female" in df and "pop_female" in df:
        den_f = df["pop_female"].where(df["pop_female"] > 0)
        df["Female_Workforce_Rate"] = df["workers_female"] / den_f * 100
    if "non_workers_total" in df and "workers_total" in df and "pop_under6_total" in df:
        den_w = df["workers_total"].where(df["workers_total"] > 0)
        df["Dependency_Ratio"] = (df["pop_under6_total"] + df["non_workers_total"]) / den_w
    df["Log_Population"] = np.log10(pt)
    df = df.replace([np.inf, -np.inf], np.nan)
    q["literacy_basis"] = literacy_basis
    q["rows_after_cleaning"] = len(df)
    return df.reset_index(drop=True), q


def derived_variable_table() -> pd.DataFrame:
    return pd.DataFrame({"Derived variable": list(DERIVED), "Definition": list(DERIVED.values())})
