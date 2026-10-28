"""Module C - AI Literacy Classification (Random Forest, Gradient Boosting, Soft-Voting Ensemble)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["Low", "Medium", "High"]

BASE_FEATURES = [
    "Sex_Ratio", "Child_Population_Share", "Workforce_Participation_Rate",
    "SC_Population_Share", "ST_Population_Share",
]

ENHANCED_FEATURES = BASE_FEATURES + [
    "Child_Sex_Ratio", "Female_Workforce_Rate", "Dependency_Ratio", "Log_Population",
]

CANDIDATE_FEATURES = ENHANCED_FEATURES

FORBIDDEN = {
    "Literacy_Rate", "Literacy_Gap", "Literacy_Category", "literacy_rate", "literate_total", "illiterate_total",
    "Literacy_Rate_Source", "Literacy_Rate_Effective", "Literacy_Rate_Crude_Check",
    "district_code", "district_name", "District", "District_Label", "state_name", "state_code", "lgd_code", "district_name_lgd",
}


def make_target(df: pd.DataFrame, low: float = 60.0, high: float = 75.0) -> pd.Series:
    """Low: <low; Medium: low <= x < high; High: >= high. Project-defined, not official."""
    if not low < high:
        raise ValueError("low threshold must be smaller than high threshold")
    r = df["Literacy_Rate"]
    cat = pd.Series(np.select([r < low, r < high, r >= high], LABELS, default=""), index=df.index)
    cat[r.isna()] = np.nan
    return pd.Categorical(cat, categories=LABELS, ordered=True)


def class_distribution(df: pd.DataFrame, low: float = 60.0, high: float = 75.0) -> pd.DataFrame:
    y = pd.Series(make_target(df, low, high))
    c = y.value_counts().reindex(LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Count": c, "Percent": c / max(c.sum(), 1) * 100}).assign(Missing_target=int(y.isna().sum()))


def available_features(df: pd.DataFrame, feature_set: str = "enhanced") -> list[str]:
    cands = BASE_FEATURES if feature_set == "base" else ENHANCED_FEATURES
    feats = [f for f in cands if (f in df.columns) and bool(df[f].notna().any())]
    leak = FORBIDDEN.intersection(feats)
    assert not leak, f"Target leakage: {leak}"
    return feats


def train_evaluate(
    df: pd.DataFrame,
    low: float = 60.0,
    high: float = 75.0,
    test_size: float = 0.2,
    seed: int = 42,
    n_estimators: int = 300,
    cv: bool = True,
    model_type: str = "rf",
    feature_set: str = "enhanced",
    class_weight: str | None = "balanced",
) -> dict:
    """Train and evaluate classification model.

    Parameters
    ----------
    model_type: 'rf' (Random Forest), 'gradient_boosting' (Gradient Boosting), or 'ensemble' (Soft Voting)
    feature_set: 'enhanced' (9 demographic predictors) or 'base' (5 original predictors)
    class_weight: 'balanced' or None
    """
    feats = available_features(df, feature_set=feature_set)
    if not feats:
        raise ValueError("No valid predictor columns available.")
    d = df.copy()
    d["Literacy_Category"] = make_target(d, low, high)
    n_missing = int(d["Literacy_Category"].isna().sum())
    d = d.dropna(subset=["Literacy_Category"])
    y = d["Literacy_Category"].astype(str)
    counts = y.value_counts()
    present = [l for l in LABELS if l in counts.index]
    warnings = []
    if len(present) < 2:
        raise ValueError(f"Only one class present ({present}); classification is not meaningful with these thresholds.")
    if counts.min() < 2:
        raise ValueError(
            f"A class has fewer than 2 observations ({counts.to_dict()}); stratified split impossible. "
            "Thresholds were NOT changed automatically; adjust them or interpret with care."
        )
    n_test = int(np.ceil(len(d) * test_size))
    stratify = y if (counts.min() >= 2 and n_test >= len(present)) else None
    if stratify is None:
        warnings.append("Test set too small for stratification; unstratified split used.")
    if counts.min() < 10:
        warnings.append(f"Smallest class has only {counts.min()} districts; metrics for it are unreliable.")

    X = d[feats]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    def pre():
        return ColumnTransformer([("num", SimpleImputer(strategy="median"), feats)])

    base = Pipeline([("prep", pre()), ("clf", DummyClassifier(strategy="most_frequent"))]).fit(Xtr, ytr)

    # Select model architecture
    if model_type == "gradient_boosting":
        clf_step = GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=seed
        )
        model_name = "Gradient Boosting"
    elif model_type == "ensemble":
        clf1 = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10, min_samples_leaf=2,
            class_weight=class_weight, random_state=seed, n_jobs=-1
        )
        clf2 = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=seed
        )
        clf_step = VotingClassifier(estimators=[("rf", clf1), ("gb", clf2)], voting="soft")
        model_name = "Soft-Voting Ensemble (RF + GB)"
    else:  # default 'rf'
        clf_step = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10, class_weight=class_weight,
            random_state=seed, n_jobs=-1
        )
        model_name = f"Random Forest ({'Balanced' if class_weight else 'Standard'})"

    model = Pipeline([("prep", pre()), ("clf", clf_step)]).fit(Xtr, ytr)

    def metrics(m):
        p = m.predict(Xte)
        pr, rc, f1, sup = precision_recall_fscore_support(yte, p, labels=present, zero_division=0)
        return {
            "pred": p,
            "accuracy": accuracy_score(yte, p),
            "balanced_accuracy": balanced_accuracy_score(yte, p),
            "macro_f1": f1_score(yte, p, labels=present, average="macro", zero_division=0),
            "per_class": pd.DataFrame({"Precision": pr, "Recall": rc, "F1": f1, "Support": sup}, index=present),
            "confusion": pd.DataFrame(
                confusion_matrix(yte, p, labels=present),
                index=[f"True {l}" for l in present],
                columns=[f"Pred {l}" for l in present],
            ),
            "report": classification_report(yte, p, labels=present, zero_division=0),
        }

    mb = metrics(base)
    mr = metrics(model)

    # Calculate feature importances
    raw_step = model.named_steps["clf"]
    if hasattr(raw_step, "feature_importances_"):
        raw_imp = raw_step.feature_importances_
    elif hasattr(raw_step, "estimators_"):
        imps = [e.feature_importances_ for e in raw_step.estimators_ if hasattr(e, "feature_importances_")]
        raw_imp = np.mean(imps, axis=0) if imps else np.zeros(len(feats))
    else:
        raw_imp = np.zeros(len(feats))

    imp = pd.DataFrame({"Feature": feats, "Impurity_Importance": raw_imp})
    pi = permutation_importance(model, Xte, yte, n_repeats=20, random_state=seed, scoring="balanced_accuracy")
    imp["Permutation_Importance"] = pi.importances_mean
    imp = imp.sort_values("Impurity_Importance", ascending=False).reset_index(drop=True)

    cv_scores = None
    k = min(5, int(ytr.value_counts().min()))
    if cv and k >= 2:
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        cv_pipe = Pipeline([("prep", pre()), ("clf", clf_step)])
        cv_scores = cross_val_score(cv_pipe, Xtr, ytr, cv=skf, scoring="f1_macro")

    table = pd.DataFrame({
        "District": d.loc[Xte.index, "District_Label"],
        "Actual": yte,
        "Predicted": mr["pred"],
        "Predicted_RF": mr["pred"],  # Alias for backward compatibility
        "Predicted_Baseline": mb["pred"],
    }).assign(Correct=lambda t: t.Actual == t.Predicted)

    return {
        "features": feats,
        "feature_set": feature_set,
        "model_type": model_type,
        "model_name": model_name,
        "thresholds": (low, high),
        "seed": seed,
        "test_size": test_size,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "n_missing_target": n_missing,
        "class_counts": counts.reindex(LABELS).fillna(0).astype(int),
        "stratified": stratify is not None,
        "baseline": mb,
        "rf": mr,  # Key 'rf' preserved for full backward compatibility
        "model_metrics": mr,
        "importance": imp,
        "cv_macro_f1": cv_scores,
        "test_table": table.reset_index(drop=True),
        "warnings": warnings,
        "labels": present,
        "model": model,
    }


def correlation_with_literacy_note(df: pd.DataFrame, feature_set: str = "enhanced") -> pd.Series:
    """Spearman correlation of each predictor with Literacy_Rate (diagnostic only, not a feature)."""
    feats = available_features(df, feature_set=feature_set)
    return df[feats + ["Literacy_Rate"]].corr(method="spearman")["Literacy_Rate"].drop("Literacy_Rate")
