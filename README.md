# Census 2011: Population, Education, Literacy Classification and UIPS (SI Lab)

Population Analysis, Education and Social Development Planning, Literacy Classification, and District Infrastructure Priority Scoring Using Census 2011 Data.
Data source (only): https://huggingface.co/datasets/indiaset/census-2011 (640 district rows, 29 columns, per the repository viewer; the code re-verifies on load).

## Contents
| File | Purpose |
|---|---|
| `SI_Lab_Modules_A_B_C_D_Census_2011.ipynb` | Complete notebook, Modules A-D |
| `app.py` | Streamlit prototype (7 pages) |
| `data_loader.py` | Repo file listing, download, local fallback |
| `preprocessing.py` | Cleaning, schema mapping, derived variables, quality checks |
| `module_a.py` ... `module_d.py` | Population, education/statistics, Random Forest, UIPS |
| `statistical_tests.py`, `visualizations.py` | Shared helpers |

## Setup and run
1. Install Python 3.10+.
2. Extract the ZIP and open a terminal in `si_lab_census_project/`.
3. Create and activate a virtual environment: `python -m venv .venv` then `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
4. `pip install -r requirements.txt`
5. Launch the app: `streamlit run app.py`
6. Open the local URL printed by Streamlit (usually http://localhost:8501).
7. If the download fails, download the Parquet file from the Hugging Face page and place it (or a CSV/Excel export) in `data/`. The loader falls back to it automatically; reload the app.

**Notebook:** run `jupyter notebook`, open `SI_Lab_Modules_A_B_C_D_Census_2011.ipynb` from this folder, and use *Kernel > Restart & Run All* (cells run top to bottom; Cell 1 installs dependencies).

## Methodology (summary)
- Child population = age 0-6; sex ratio = females per 1000 males; workforce participation = workers / total population.
- Statistical tests: Strictly conforms to SI Lab Practical 5 (Notebooks 1 & 2): Z-tests, Student's/Welch's Two-Sample t-tests, One-Sample t-tests, One-Way ANOVA, Two-Way ANOVA, and Chi-Square tests of independence. Unapproved non-parametric tests (Mann-Whitney U, Kruskal-Wallis) are excluded; assumption violations are explicitly reported with limitations rather than test substitutions.
- The loader code tests which denominator `literacy_rate` uses. In the repository preview it matches `literate_total / pop_total` (a crude all-ages rate). Child share therefore mechanically relates to literacy. The sidebar/`literacy_basis="effective"` option gives the age 7+ rate.
- Module C: Low < 60, Medium 60 to < 75, High >= 75 (project thresholds); features are sex ratio, child share, workforce participation, SC share, ST share; literacy variables/identifiers excluded; majority-class baseline; 80:20 stratified split; seed 42.
- Module D: min-max normalisation, UIPS = 100 x (0.35P + 0.25C + 0.25L + 0.15W); omitted components are removed and weights rescaled; categories <=33.33 / <=66.67 / above; five sensitivity scenarios.

## Limitations
Census 2011 is historical. No land area or boundary geometry exists in the dataset, so no density or maps. District aggregates hide internal variation. UIPS and the Random Forest are screening/association tools: they do not verify school, hospital, road or housing shortages, nor justify new construction. Weights are provisional assumptions.
