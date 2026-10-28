# RAMDEOBABA UNIVERSITY, NAGPUR
## DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
### B.Tech. CSE (Artificial Intelligence and Data Science) (AIDS)

---

### SEMESTER V  
### SI LAB PROJECT REPORT  

# Population Analysis, Education and Social Development Planning, Literacy Classification, and District Infrastructure Priority Scoring Using Census 2011 Data

**Submitted by:**

| Student Name | Roll No. | Enrollment No. |
| :--- | :--- | :--- |
| *[Student Name]* | *[Roll No.]* | *[Enrollment No.]* |

**Academic Year:** 2026–27  

---

<div style="page-break-after: always;"></div>

# Population Analysis, Education and Social Development Planning, Literacy Classification, and District Infrastructure Priority Scoring Using Census 2011 Data

## SI Lab Project Report

---

### 1. PROJECT OBJECTIVE

The primary objective of this project is to provide a transparent, empirical, and reproducible computational framework for district-level demographic analysis, educational planning, literacy classification, and infrastructure screening using the official IndiaSet Census 2011 dataset.

The specific objectives achieved in this project include:
1. **Load and explore** the official 640-district Census 2011 dataset from verified repositories without synthetic records or fabricated variables.
2. **Preprocess and validate** district administrative records, verify denominator relationships, check mathematical total consistencies, and engineer derived indicators.
3. **Analyze demographic patterns and execute syllabus-approved statistical tests** (Two-Sample Welch's t-test, One-Sample t/Z-tests, One-Way ANOVA, Two-Way ANOVA, and Chi-Square test of independence) while evaluating distributional assumptions without substituting unlisted tests.
4. **Extract and select non-literacy demographic features** strictly eliminating target leakage.
5. **Train multiple machine learning classification models** (Dummy Majority Baseline, Standard Random Forest, Balanced Cost-Sensitive Random Forest, Gradient Boosting, and Soft-Voting Ensemble) to categorize districts into Low, Medium, and High literacy tiers.
6. **Evaluate and benchmark model performance** using multi-class accuracy, precision, recall, macro-averaged F1-score, and confusion matrices.
7. **Select the best classification architecture** based on empirical evaluation metrics and generalization capability.
8. **Construct a transparent Urban Infrastructure Planning Priority Score (UIPS)** using normalized multi-criteria weighting with sensitivity analysis.
9. **Develop an interactive Graphical User Interface (GUI)** using Streamlit to facilitate parameter exploration, interactive "What-If" demographic simulation, and district benchmarking.

---

### 2. INTRODUCTION

District-level planning in India requires systematic screening of administrative units to identify demographic pressures, educational disparities, and infrastructure development needs. India's Census 2011 provides complete administrative coverage across 640 districts, encapsulating population counts, gender ratios, early childhood cohorts (ages 0–6), marginalized communities (Scheduled Castes and Scheduled Tribes), workforce participation, and literacy counts.

This project implements an end-to-end analytical pipeline structured into four interconnected modules:
* **Module A (Population Pressure):** Analyzes aggregate population distributions, skewness, right-tail concentration, and early-childhood demographic structure.
* **Module B (Education & Social Indicators):** Evaluates gender equity (sex ratio), marginalized community representation, bivariate relationships, and executes rigorous hypothesis testing.
* **Module C (Literacy Level Classification):** Employs supervised machine learning to classify districts into literacy tiers based strictly on non-literacy demographic predictors.
* **Module D (Infrastructure Planning Priority - UIPS):** Synthesizes demographic pressure, early childhood share, educational deficit, and dependency burden into a composite screening index (0–100) with multi-scenario sensitivity testing.

#### 2.1 Problem Statement
Administrative planners require an objective, data-driven screening methodology to prioritize districts for socio-economic assessment without confusing demographic pressure with verified physical infrastructure deficits. Furthermore, educational classification must avoid data leakage from proxy variables, and statistical hypothesis testing must adhere strictly to verified syllabus methodologies without substituting unapproved non-parametric tests when parametric assumptions encounter empirical skewness.

#### 2.2 Scope of the Project
* **Included:**
  * Complete coverage of all 640 districts across 35 States and Union Territories of India from Census 2011.
  * Rigorous verification of crude vs. effective literacy rate definitions.
  * Implementation of strictly syllabus-approved statistical hypothesis tests (Z-test, T-test, One-Way ANOVA, Two-Way ANOVA, Chi-Square test) with formal assumption reporting (Shapiro-Wilk normality and Levene's variance homogeneity).
  * Supervised multi-class classification using five distinct model configurations.
  * Interactive Streamlit web application providing live simulators and data exports.
* **Limitations:**
  * Census 2011 data is historical; subsequent district bifurcations and recent demographic shifts are not reflected.
  * The dataset contains no spatial boundary polygons or land area records; physical population density and choropleth maps cannot be calculated.
  * Demographic indicators serve as screening proxies; they do not measure actual physical classroom shortages, hospital bed deficits, or road connectivity.

---

### 3. PROJECT METHODOLOGY

#### 3.1 Dataset Loading and Exploration
* **Dataset Name & Source:** IndiaSet Census 2011 District Dataset (`indiaset/census-2011`, Hugging Face).
* **Dimensions:** 640 district rows, 29 original attributes.
* **Data Types:** Numeric integer counts for demographic cohorts, floating-point rates, and text identifiers (`district_name`, `state_name`, `district_code`, `state_code`).
* **Missing Values & Duplicates:**
  * Total missing values across all raw columns: **0**.
  * Duplicate district records: **0** (verified across `district_code`).
  * Cross-state name duplication: Several district names repeat across different states (e.g., Bilaspur, Aurangabad, Hamirpur); `district_code` is maintained as the primary key and compound labels (`District (State)`) are created for unambiguous display.
* **Class Distribution (Target: Literacy Tier):**
  * Thresholds: Low ($<60\%$), Medium ($60\%\text{ to }<75\%$), High ($\ge 75\%$).
  * Low Literacy: 261 districts (40.78%).
  * Medium Literacy: 301 districts (47.03%).
  * High Literacy: 78 districts (12.19%).
  * The target demonstrates pronounced class imbalance, particularly in the minority "High" class.

#### 3.2 Data Preprocessing
1. **Numeric Coercion:** Standardized all count and rate columns to 64-bit numeric data types.
2. **Denominators & Consistency Checks:**
   * Validated that $\text{Male} + \text{Female} = \text{Total Population}$ (0 violations).
   * Validated that $\text{Workers} + \text{Non-Workers} = \text{Total Population}$ (0 violations).
   * Verified that the raw `literacy_rate` column matches $\frac{\text{literate\_total}}{\text{pop\_total}} \times 100$ (median absolute difference $= 0.00\%$), proving the raw column is a crude all-ages rate.
   * Formulated the Census "effective" literacy rate: $\frac{\text{literate\_total}}{\text{pop\_total} - \text{pop\_under6\_total}} \times 100$.
3. **Data Splitting:** Applied an 80:20 stratified train-test split ($n_{\text{train}} = 512$, $n_{\text{test}} = 128$) with a fixed random seed (`42`) to ensure reproducibility.
4. **Pipeline Imputation:** Fitted a median imputer within a scikit-learn `Pipeline` strictly on training folds to prevent data snooping.

#### 3.3 Feature Extraction / Selection
To maintain strict scientific validity and prevent **target leakage**, all literacy counts (`literate_total`, `literate_male`, `literate_female`, `illiterate_total`), literacy rates, and geographic identifiers were excluded from the feature space. 

Nine demographic predictors were extracted from raw counts:
1. `Sex_Ratio`: Females per 1,000 males.
2. `Child_Population_Share`: Percentage of population aged 0–6 ($\frac{\text{pop\_under6\_total}}{\text{pop\_total}} \times 100$).
3. `Workforce_Participation_Rate`: Percentage of active workers ($\frac{\text{workers\_total}}{\text{pop\_total}} \times 100$).
4. `SC_Population_Share`: Scheduled Caste population share ($\frac{\text{pop\_sc}}{\text{pop\_total}} \times 100$).
5. `ST_Population_Share`: Scheduled Tribe population share ($\frac{\text{pop\_st}}{\text{pop\_total}} \times 100$).
6. `Child_Sex_Ratio`: Females per 1,000 males in the 0–6 age cohort ($\frac{\text{pop\_under6\_female}}{\text{pop\_under6\_male}} \times 1000$).
7. `Female_Workforce_Rate`: Female labor force participation ($\frac{\text{workers\_female}}{\text{pop\_female}} \times 100$).
8. `Dependency_Ratio`: Ratio of dependents to active workers ($\frac{\text{pop\_under6\_total} + \text{non\_workers\_total}}{\text{workers\_total}}$).
9. `Log_Population`: Logarithmic population scale ($\log_{10}(\text{pop\_total})$).

#### 3.4 Training Multiple Classification Models
Five distinct classification models were implemented to benchmark predictive capacity:

| Sr. No. | Model Name | Purpose / Notes |
| :---: | :--- | :--- |
| 1 | **Dummy Majority Baseline** | Establishes the performance floor by uniformly predicting the most frequent training class (Medium). |
| 2 | **Standard Random Forest** | An ensemble of 300 decorrelated decision trees using uniform class weighting and the 5 base demographic features. |
| 3 | **Balanced Random Forest** | Random Forest with cost-sensitive `class_weight="balanced"` and 9 engineered features to penalize minority-class errors. |
| 4 | **Gradient Boosting Classifier** | Sequential ensemble of 180 shallow trees optimizing deviance loss with shrinkage (`learning_rate=0.08`, `subsample=0.85`). |
| 5 | **Soft-Voting Ensemble (RF + GB)** | Combines the predicted probability distributions of Balanced Random Forest and Gradient Boosting. |

#### 3.5 Performance Evaluation and Model Comparison
Evaluated on the held-out test set ($n = 128$ districts, stratified across Low, Medium, and High literacy classes):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy Majority Baseline** | 46.88% | 0.1562 | 0.3333 | 0.2128 |
| **Standard Random Forest** | 72.66% | 0.7303 | 0.6264 | 0.6498 |
| **Balanced Random Forest** | 74.22% | 0.7418 | 0.6375 | 0.6606 |
| **Gradient Boosting** | 75.78% | 0.7614 | 0.6656 | 0.6911 |
| **Soft-Voting Ensemble (RF + GB)** | **76.56%** | **0.7661** | **0.6720** | **0.6969** |

*Note: All values are empirical results generated from the actual Census 2011 test dataset.*

#### 3.6 Best Model Selection
The **Soft-Voting Ensemble (Random Forest + Gradient Boosting)** was selected as the champion model. It achieved the highest test accuracy (**76.56%**, a $+29.68\%$ improvement over the dummy baseline) and the highest macro-averaged F1-score (**0.6969**). The soft probability voting mechanism balances the low variance of Random Forest bagging with the low bias of Gradient Boosting, resulting in superior identification of the difficult minority "High" literacy class.

---

### 4. PROJECT IMPLEMENTATION

#### 4.1 Tools and Technologies

| Tool / Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.12.5 (3.10+ compatible) | Core programming language |
| **Jupyter / Colab** | 1.1.1 / nbformat 5.10 | Notebook research and model verification environment |
| **Pandas / NumPy** | Pandas 3.0+, NumPy 2.1+ | Data manipulation, matrix operations, and cleaning |
| **Scikit-learn** | 1.8.0 / 1.9+ | Pipelines, cross-validation, and classification models |
| **SciPy / Statsmodels** | SciPy 1.14+, Statsmodels 0.15 | Hypothesis testing (Z-test, T-test, ANOVA, Chi-Square) |
| **Plotly** | 6.7+ | Interactive visualizations, heatmaps, radar and gauge charts |
| **Streamlit** | 1.56+ / 1.66+ | Web-based graphical user interface (GUI) |

#### 4.2 Module-wise Implementation
1. **`data_loader.py`:** Handles automated dataset retrieval from Hugging Face (`indiaset/census-2011`) with local Parquet fallback caching in `./data/`.
2. **`preprocessing.py`:** Implements numeric coercion, denominator checks, crude vs. effective rate computations, zero-leakage feature engineering, and data dictionary mappings.
3. **`module_a.py`:** Computes population descriptive summaries, top-10/bottom-10 rankings, skewness metrics, and state aggregations.
4. **`module_b.py`:** Implements syllabus-compliant hypothesis tests, bivariate scatter relations, and weighted state literacy aggregations.
5. **`module_c.py`:** Houses machine learning pipeline architectures, cross-validation runners, permutation importance routines, and prediction tables.
6. **`module_d.py`:** Implements min-max normalized multi-criteria UIPS formulation, percentile winsorization, and 5-scenario sensitivity simulations.
7. **`statistical_tests.py`:** Reusable backend implementing strictly syllabus-authorized statistical tests (Z-test, T-test, One-Way ANOVA, Two-Way ANOVA, Chi-Square) with formal assumption checking.
8. **`visualizations.py`:** Reusable Plotly graphing library providing styled horizontal bars, distributions, correlation matrices, confusion heatmaps, gauge indicators, and demographic radar charts.

#### 4.3 GUI Implementation
The graphical user interface is implemented as a multi-page Streamlit application (`app.py`) organized into eight functional sections:
* **Sidebar Controls:** Houses global literacy basis toggles (source vs. effective), dynamic multi-state filters, and navigation controls.
* **Module C Dashboard:** Provides dynamic sliders for low/high literacy thresholds, test split percentages, model architecture selectors (Ensemble, Gradient Boosting, Balanced RF, Standard RF), and feature set toggles.
* **Interactive What-If Simulator:** Accepts user inputs for all 9 demographic features or prefills values from any of the 640 districts, producing live model predictions and probability distribution charts.
* **Module D Priority Panel:** Features interactive weight sliders for Population, Child Share, Literacy Gap, and Workforce Participation with auto-normalization, district leaderboard, and single-district score waterfall breakdowns.

---

### 5. COMPLETE CODE

The complete, modular source code is maintained in the project repository and Jupyter Notebook. Below are the core algorithmic implementations.

#### 5.1 Machine Learning Classification Pipeline (`module_c.py`)
```python
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

LABELS = ["Low", "Medium", "High"]
BASE_FEATURES = ["Sex_Ratio", "Child_Population_Share", "Workforce_Participation_Rate", "SC_Population_Share", "ST_Population_Share"]
ENHANCED_FEATURES = BASE_FEATURES + ["Child_Sex_Ratio", "Female_Workforce_Rate", "Dependency_Ratio", "Log_Population"]

def train_evaluate(df, low=60.0, high=75.0, test_size=0.2, seed=42, model_type="ensemble", feature_set="enhanced", class_weight="balanced"):
    feats = BASE_FEATURES if feature_set == "base" else ENHANCED_FEATURES
    d = df.copy()
    r = d["Literacy_Rate"]
    d["Literacy_Category"] = pd.Series(np.select([r < low, r < high, r >= high], LABELS, default=""), index=d.index)
    d = d.dropna(subset=["Literacy_Category"])
    X, y = d[feats], d["Literacy_Category"].astype(str)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    def pre(): return ColumnTransformer([("num", SimpleImputer(strategy="median"), feats)])

    base = Pipeline([("prep", pre()), ("clf", DummyClassifier(strategy="most_frequent"))]).fit(Xtr, ytr)

    if model_type == "gradient_boosting":
        clf_step = GradientBoostingClassifier(n_estimators=180, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=seed)
    elif model_type == "ensemble":
        clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight=class_weight, random_state=seed, n_jobs=-1)
        clf2 = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=seed)
        clf_step = VotingClassifier(estimators=[("rf", clf1), ("gb", clf2)], voting="soft")
    else:
        clf_step = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight=class_weight, random_state=seed, n_jobs=-1)

    model = Pipeline([("prep", pre()), ("clf", clf_step)]).fit(Xtr, ytr)
    p = model.predict(Xte)

    return {
        "model": model, "accuracy": accuracy_score(yte, p),
        "balanced_accuracy": balanced_accuracy_score(yte, p),
        "macro_f1": f1_score(yte, p, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(yte, p, labels=LABELS, zero_division=0)
    }
```

#### 5.2 Syllabus-Approved Statistical Testing (`statistical_tests.py`)
```python
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

def two_sample_t_test(df, value, group_col, alpha=0.05):
    d = df[[value, group_col]].dropna()
    levels = sorted(d[group_col].unique())
    a = d.loc[d[group_col] == levels[0], value].to_numpy(dtype=float)
    b = d.loc[d[group_col] == levels[1], value].to_numpy(dtype=float)
    sh_a, sh_b = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
    lev_p = stats.levene(a, b, center="median").pvalue
    equal_var = lev_p >= alpha
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    return {
        "test": "Two-Sample Student's t-test" if equal_var else "Two-Sample Welch's t-test",
        "statistic": float(res.statistic), "df": float(res.df), "p_value": float(res.pvalue),
        "decision": "reject H0" if res.pvalue < alpha else "fail to reject H0",
        "shapiro_p": (sh_a, sh_b), "levene_p": lev_p
    }

def one_way_anova(df, value, group_col, min_group_size=5, alpha=0.05):
    d = df[[value, group_col]].dropna()
    sizes = d.groupby(group_col)[value].size()
    d = d[d[group_col].isin(sizes[sizes >= min_group_size].index)]
    groups = [d.loc[d[group_col] == g, value].to_numpy(dtype=float) for g in sorted(d[group_col].unique())]
    F_classic, p_classic = stats.f_oneway(*groups)
    resid = np.concatenate([g - g.mean() for g in groups])
    sh_p = stats.shapiro(resid).pvalue
    lev_p = stats.levene(*groups, center="median").pvalue
    return {
        "test": "One-Way ANOVA", "classic_F": float(F_classic), "classic_p": float(p_classic),
        "shapiro_p": sh_p, "levene_p": lev_p, "decision": "reject H0" if p_classic < alpha else "fail to reject H0"
    }
```

---

### 6. RESULTS AND DISCUSSION

#### 6.1 Summary of Actual Outputs
1. **Data Preprocessing & Validation:** Verified 640 complete district records. Confirmed that crude literacy rate uses total population as denominator ($|\text{Rate} - \text{Crude}| = 0.00\%$), proving children aged 0–6 mechanically lower the crude rate.
2. **Two-Sample Welch's T-Test Output:**
   * Test: Two-Sample Welch's t-test ($H_0: \mu_{\text{above}} = \mu_{\text{below}}$)
   * Sample sizes: $n_{\text{above}} = 320$, $n_{\text{below}} = 320$.
   * Group means: Above-median SC+ST mean $= 60.74\%$; At/below-median SC+ST mean $= 64.19\%$.
   * $t = -4.201$, $\text{df} = 628.6$, $p = 3.04 \times 10^{-5}$, Cohen's $d = -0.332$.
   * Decision: **Reject $H_0$** at $\alpha = 0.05$.
3. **One-Way ANOVA across States Output:**
   * Groups evaluated: 26 states with $\ge 5$ districts ($n = 618$ districts).
   * Classic $F = 25.019$, $p = 2.44 \times 10^{-76}$, $\eta^2 = 0.514$.
   * Welch's robust $F = 51.520$, $p = 2.60 \times 10^{-60}$.
   * Decision: **Reject $H_0$** at $\alpha = 0.05$.
4. **Two-Way ANOVA Output:**
   * Factor A (SC+ST Tier): $F = 20.91$, $p = 6.0 \times 10^{-6}$ (Significant).
   * Factor B (Workforce Tier): $F = 2.98$, $p = 0.085$ (Non-significant).
   * Interaction ($A \times B$): $F = 14.88$, $p = 0.000126$ (Significant interaction).
5. **Chi-Square Test of Independence Output:**
   * $\chi^2 = 20.10$, $\text{df} = 2$, $p = 4.33 \times 10^{-5}$, Cramér's $V = 0.177$.
   * Decision: **Reject $H_0$** at $\alpha = 0.05$.
6. **Classification Model Performance Output:**
   * Champion Soft-Voting Ensemble achieved **76.56% Accuracy** and **0.6969 Macro F1-Score**.
   * Feature importance ranks `Child_Population_Share` ($38.2\%$), `Sex_Ratio` ($19.4\%$), and `Female_Workforce_Rate` ($14.1\%$) as the top predictors.

#### 6.2 Result Analysis
* **Statistical Significance vs. Causation:** The hypothesis tests confirm that districts with higher proportions of marginalized communities (SC+ST) exhibit lower average literacy rates. However, this statistical association does not prove causation; it reflects historical and socio-economic disparities.
* **Model Differences:** The Dummy Baseline achieves only 46.88% accuracy by always guessing the majority class, demonstrating severe failure on minority categories. Standard Random Forest improves accuracy to 72.66% but struggles on the "High" class (Recall = 0.31). The Soft-Voting Ensemble significantly mitigates this imbalance, raising High-class recall to 0.50 and overall accuracy to 76.56%.
* **UIPS Planning Utility:** The UIPS score successfully identifies priority districts (e.g., Purba Champaran, Mewat, Purnia) driven by high child shares and large literacy deficits. Sensitivity analysis demonstrates that 257 districts shift priority categories across weighting scenarios, emphasizing that UIPS is an exploratory screening index rather than a rigid construction mandate.

---

### 7. CONCLUSION

This project successfully developed an integrated demographic and educational analysis suite on Census 2011 data. By strictly enforcing zero target leakage, nine demographic predictors were extracted to train five classification architectures. The **Soft-Voting Ensemble** achieved superior performance with **76.56% test accuracy** and **0.6969 Macro-F1**, substantially exceeding the 46.88% baseline. 

The analytical pipeline was strictly aligned with the SI Lab practical syllabus, restricting significance testing to Z-tests, Student's/Welch's t-tests, ANOVA, and Chi-square tests, while completely removing unapproved non-parametric substitutes (Mann-Whitney U and Kruskal-Wallis) and replacing them with transparent assumption violation reporting. The interactive Streamlit application provides accessible exploration, What-If simulation, and multi-scenario priority screening for educational and demographic planners.

---

### 8. GITHUB LINK

**GitHub Repository URL:**  
[https://github.com/mpande003/Population-Data-Analysis.git](https://github.com/mpande003/Population-Data-Analysis.git)

The repository contains:
1. `SI_Lab_Modules_A_B_C_D_Census_2011.ipynb` — Executed Colab-compatible notebook with all cell outputs.
2. `app.py` — Streamlit web application.
3. `data_loader.py`, `preprocessing.py`, `statistical_tests.py`, `visualizations.py` — Core backend modules.
4. `module_a.py`, `module_b.py`, `module_c.py`, `module_d.py` — Analytical and ML logic.
5. `data/census_2011_districts_final.parquet` — Verified district dataset.
6. `requirements.txt` — Environment specifications.

---

### 9. REFERENCES

[1] Office of the Registrar General & Census Commissioner, India (ORGI), *Census of India 2011: Primary Census Abstract*, Ministry of Home Affairs, Government of India. Dataset hosted on Hugging Face: `indiaset/census-2011`. URL: https://huggingface.co/datasets/indiaset/census-2011  
[2] Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. URL: https://scikit-learn.org/stable/  
[3] Virtanen, P. et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020. URL: https://scipy.org/  
[4] Seabold, S. and Perktold, J., "Statsmodels: Econometric and statistical modeling with python," *Proceedings of the 9th Python in Science Conference*, 2010. URL: https://www.statsmodels.org/stable/  
[5] Streamlit Inc., *Streamlit Documentation: The fastest way to build and share data apps*, 2024. URL: https://docs.streamlit.io/  
[6] Plotly Technologies Inc., *Collaborative Data Science: Plotly Express and Graph Objects*, 2024. URL: https://plotly.com/python/  
