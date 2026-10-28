"""Streamlit prototype - Census 2011 SI Lab project (Modules A-D).
Run: streamlit run app.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

import data_loader
import preprocessing as pp
import module_a as A
import module_b as B
import module_c as C
import module_d as D
import statistical_tests as ST
import visualizations as V

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Modern Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Census 2011 Analytics & Planning Suite | SI Lab",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for an ultra-clean, modern, and attractive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #312E81 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.6rem;
        background: linear-gradient(90deg, #FFFFFF, #E2E8F0, #A5B4FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        max-width: 900px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-indigo { background: rgba(99, 102, 241, 0.2); color: #C7D2FE; border: 1px solid rgba(99, 102, 241, 0.4); }
    .badge-emerald { background: rgba(16, 185, 129, 0.2); color: #A7F3D0; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #FDE68A; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-rose { background: rgba(244, 63, 94, 0.2); color: #FECDD3; border: 1px solid rgba(244, 63, 94, 0.4); }

    /* Polished Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Module Highlights */
    .module-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .module-card h4 {
        margin: 0 0 0.4rem 0;
        color: #1E293B;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .module-card p {
        margin: 0;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Disclaimer Callout */
    .disclaimer-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #92400E;
        line-height: 1.5;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.6rem 0 1rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #F1F5F9;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Sidebar polish */
    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1E293B;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

NOTICE = (
    "Census 2011 is historical data. Demographic indicators (population size, child share, literacy, workforce) "
    "are screening proxies and do NOT measure physical infrastructure shortages, classroom overcrowding, "
    "hospital bed deficits, or road connectivity."
)


# -----------------------------------------------------------------------------
# 2. Data Caching & Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Connecting to IndiaSet Census 2011 repository...")
def get_raw() -> tuple[pd.DataFrame, str]:
    return data_loader.load_raw()


@st.cache_data(show_spinner="Cleaning and engineering derived demographic features...")
def get_clean(basis: str) -> tuple[pd.DataFrame, dict]:
    raw_df, _ = get_raw()
    return pp.clean_data(raw_df, basis)


@st.cache_data(show_spinner="Training machine learning model...")
def run_model(basis: str, low: float, high: float, test_size: float, seed: int, n_est: int = 300,
              model_type: str = "ensemble", feature_set: str = "enhanced", class_weight: str | None = "balanced") -> dict:
    clean_df, _ = get_clean(basis)
    return C.train_evaluate(clean_df, low, high, test_size, seed, n_estimators=n_est,
                            model_type=model_type, feature_set=feature_set, class_weight=class_weight)


# Load dataset safely
try:
    raw, source_desc = get_raw()
except Exception as exc:
    st.error("⚠️ Dataset could not be loaded automatically from Hugging Face.")
    st.info("Place `census_2011_districts_final.parquet` in the `./data/` folder and reload the page.")
    st.code(str(exc))
    st.stop()


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Global Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🏛️ Census 2011 Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">SI Lab • Ramdeobaba University, Nagpur</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview & Dashboard",
            "👥 Module A: Population Analysis",
            "🎓 Module B: Education & Social Indicators",
            "🌲 Module C: AI Literacy Classifier",
            "🎯 Module D: Infrastructure Priority (UIPS)",
            "🔎 District Intelligence Dossier",
            "📁 Dataset Explorer & Audit",
            "📚 Methodology & Limitations",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Analytical Settings")

    basis = st.selectbox(
        "Literacy rate basis",
        ["source", "effective"],
        index=0,
        help=(
            "'source': crude all-ages rate (matches dataset column `literacy_rate = literate_total / pop_total`).\n"
            "'effective': age 7+ rate (`literate_total / (pop_total - age_0_6)`)."
        ),
    )

    df, quality = get_clean(basis)

    # State filter
    all_states = sorted(df["state_name"].unique())
    selected_states = st.multiselect(
        "Filter by State / UT",
        all_states,
        placeholder="All States & UTs (640 Districts)",
    )

    filtered_df = df[df["state_name"].isin(selected_states)] if selected_states else df

    st.markdown("---")
    st.caption(f"**Data source:** `{data_loader.REPO_ID}`")
    st.caption(f"**Districts Active:** {len(filtered_df)} / {len(df)}")
    st.caption("🔒 Strict No Data Leakage Standard")


# -----------------------------------------------------------------------------
# 4. Page 1: Overview & Executive Dashboard
# -----------------------------------------------------------------------------
if page.startswith("🏠"):
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">India Census 2011 Demographic & Infrastructure Analytics</div>
        <div class="hero-subtitle">
            An end-to-end analytical suite spanning demographic concentration analysis, bivariate & hypothesis testing,
            leakage-free Random Forest literacy tier classification, and transparent weighted infrastructure planning priority scoring (UIPS).
        </div>
        <div class="badge-container">
            <span class="badge badge-indigo">IndiaSet Census 2011 Verified</span>
            <span class="badge badge-emerald">640 Districts Mapped</span>
            <span class="badge badge-amber">Modules A, B, C & D Included</span>
            <span class="badge badge-rose">Zero Synthetic Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Critical Analytical Note:</strong> {NOTICE}</div>', unsafe_allow_html=True)

    # Top Macro KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    tot_pop = df["Total_Population"].sum()
    med_pop = df["Total_Population"].median()
    mean_lit = df["Literacy_Rate"].mean()
    med_sex = df["Sex_Ratio"].median()
    mean_child = df["Child_Population_Share"].mean()

    k1.markdown(f'<div class="stat-card"><div class="stat-label">Total Population</div><div class="stat-val">{tot_pop/1e9:.2f}B</div><div class="stat-sub">{tot_pop:,.0f} persons</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="stat-card"><div class="stat-label">Total Districts</div><div class="stat-val">{len(df)}</div><div class="stat-sub">Across 35 States & UTs</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="stat-card"><div class="stat-label">Average Literacy</div><div class="stat-val">{mean_lit:.1f}%</div><div class="stat-sub">Basis: {basis.title()}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="stat-card"><div class="stat-label">Median Sex Ratio</div><div class="stat-val">{med_sex:.0f}</div><div class="stat-sub">Females per 1000 Males</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="stat-card"><div class="stat-label">Mean Child Share</div><div class="stat-val">{mean_child:.1f}%</div><div class="stat-sub">Age 0-6 population</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h3>📌 Module Architecture & Core Objectives</h3></div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="module-card" style="border-left-color: #4F46E5;">
            <h4>Module A: Population Pressure Analysis</h4>
            <p>Evaluates demographic size, right-skewed population distributions, and age-structure variation (child share, age 0-6). Demonstrates how aggregate size contrasts with demographic shares.</p>
        </div>
        <div class="module-card" style="border-left-color: #06B6D4;">
            <h4>Module B: Education & Social Development</h4>
            <p>Explores literacy rates, sex ratios, and marginalized community representations (SC & ST). Features Pearson/Spearman correlations, two-sample t-test / Welch's t-test, Z-tests, one-way and two-way ANOVA, and Chi-square tests of independence (strictly syllabus-approved).</p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="module-card" style="border-left-color: #10B981;">
            <h4>Module C: AI Literacy Classification (Random Forest)</h4>
            <p>Classifies districts into Low (&lt;60%), Medium (60-75%), and High (&ge;75%) literacy tiers using only non-literacy demographic predictors. Built with strict leakage prevention, pipeline imputation, and dummy baseline benchmarking.</p>
        </div>
        <div class="module-card" style="border-left-color: #F59E0B;">
            <h4>Module D: Urban Infrastructure Planning Priority Score (UIPS)</h4>
            <p>Transparent composite screening index (0-100) combining normalized Population, Child Share, Literacy Gap, and Workforce Participation. Includes 5 multi-scenario weight sensitivity simulations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h3>⚡ District Snapshot: Top 5 Highlights</h3></div>', unsafe_allow_html=True)

    # Quick comparative preview
    uips_preview, _ = D.compute_uips(df)
    c_pop, c_lit, c_uips = st.columns(3)

    with c_pop:
        st.markdown("##### 👥 Most Populous Districts")
        top_p = A.top_n(df, "Total_Population", 5)[["District_Label", "Total_Population"]]
        top_p["Population"] = top_p["Total_Population"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(top_p[["District_Label", "Population"]], hide_index=True, use_container_width=True)

    with c_lit:
        st.markdown("##### 📖 Highest Literacy Districts")
        top_l = B.rank_districts(df, "Literacy_Rate", 5)[["District_Label", "Literacy_Rate"]]
        top_l["Literacy"] = top_l["Literacy_Rate"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(top_l[["District_Label", "Literacy"]], hide_index=True, use_container_width=True)

    with c_uips:
        st.markdown("##### 🎯 Highest UIPS Priority Districts")
        top_u = uips_preview.head(5)[["District_Label", "UIPS", "Priority_Category"]]
        top_u["Score"] = top_u["UIPS"].apply(lambda x: f"{x:.2f}")
        st.dataframe(top_u[["District_Label", "Score", "Priority_Category"]], hide_index=True, use_container_width=True)

    # Verification status
    with st.expander("🔍 Verified Data Pipeline & Schema Audit"):
        st.dataframe(pp.module_availability(raw), hide_index=True, use_container_width=True)
        st.caption(f"Loaded source path: `{source_desc}` | Raw Dimensions: {raw.shape[0]} rows × {raw.shape[1]} columns.")


# -----------------------------------------------------------------------------
# 5. Page 2: Module A - Population Pressure Analysis
# -----------------------------------------------------------------------------
elif page.startswith("👥"):
    st.title("Module A — Demographic & Population Pressure Analysis")
    st.markdown(
        "Descriptive profiling of aggregate district population and early-childhood demographic structure (ages 0-6). "
        "*Note: Census 2011 district datasets lack land area and spatial boundary polygons, so geographic density cannot be calculated.*"
    )

    d = filtered_df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subset Population", f"{d['Total_Population'].sum():,.0f}")
    m2.metric("Districts in View", len(d))
    m3.metric("Median Population", f"{d['Total_Population'].median():,.0f}")
    m4.metric("Mean Child Share (0-6)", f"{d['Child_Population_Share'].mean():.2f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Rankings (Top & Bottom)",
        "📊 Distribution & Skewness",
        "🏛️ State-Level Totals",
        "👶 Child Demographics (0-6)",
        "📋 Summary Statistics Table",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                V.bar_h(A.top_n(d, "Total_Population", 10), "Total_Population", title="Top 10 Districts by Total Population"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                V.bar_h(A.top_n(d, "Total_Population", 10, ascending=True), "Total_Population", title="Bottom 10 Districts by Total Population"),
                use_container_width=True,
            )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.histogram(d, "Total_Population", title="Distribution of Total District Population"), use_container_width=True)
        with c2:
            st.plotly_chart(V.box(d, "Total_Population", title="Outlier Detection: Total District Population"), use_container_width=True)

        s = A.population_summary(d)["Total_Population"]
        st.info(
            f"**Right Skewness:** Mean population ({s['mean']:,.0f}) exceeds Median ({s['median']:,.0f}) by "
            f"{(s['mean'] - s['median']):,.0f} persons. The top 10 districts hold {A.concentration(d, 10)['top_k_share_%']:.2f}% of the total population."
        )

    with tab3:
        sp = A.state_population(d)
        st.plotly_chart(
            V.state_bar(sp, "Total_Population", title="Total Population by State / UT (Aggregated across Districts)"),
            use_container_width=True,
        )
        st.dataframe(sp.style.format({"Total_Population": "{:,.0f}", "Child_Population": "{:,.0f}", "Child_Population_Share": "{:.2f}%", "Share_of_Included_Population_%": "{:.2f}%"}), hide_index=True, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                V.bar_h(A.top_n(d, "Child_Population", 10), "Child_Population", title="Top 10 Districts by Child Population (Ages 0-6)"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                V.scatter(d, "Total_Population", "Child_Population_Share", color="state_name", title="Total Population vs Child Population Share (%)"),
                use_container_width=True,
            )
        st.caption("Notice that while absolute child count tracks total population volume, child percentage share reveals age-structure differences across states.")

    with tab5:
        st.dataframe(A.population_summary(d).style.format("{:,.2f}"), use_container_width=True)


# -----------------------------------------------------------------------------
# 6. Page 3: Module B - Education & Social Development
# -----------------------------------------------------------------------------
elif page.startswith("🎓"):
    st.title("Module B — Education & Social Development Indicators")
    st.markdown(
        "Rigorous statistical evaluation of district literacy rates, gender balance (sex ratio), marginalized community presence (SC & ST), "
        "bivariate correlations, hypothesis testing, and inter-state ANOVA."
    )

    d = filtered_df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Literacy Rate", f"{d['Literacy_Rate'].mean():.2f}%")
    m2.metric("Median Literacy Rate", f"{d['Literacy_Rate'].median():.2f}%")
    m3.metric("Median Sex Ratio", f"{d['Sex_Ratio'].median():.0f}")
    m4.metric("Mean SC+ST Share", f"{(d['SC_Population_Share'] + d['ST_Population_Share']).mean():.2f}%")

    if basis == "source":
        st.warning(
            "⚠️ Currently using the **source (crude)** literacy rate where total population (all ages including infants) is the denominator. "
            "To view the Census 'effective' rate for age 7+, select **effective** in the sidebar."
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📖 Literacy Profiles",
        "⚖️ Sex Ratio & SC / ST",
        "🔍 Bivariate Explorer",
        "🧬 Correlation Heatmaps",
        "🧪 T-Tests & Z-Tests (Practical 5.1)",
        "📐 ANOVA & Chi-Square (Practical 5.2)",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.bar_h(B.rank_districts(d, "Literacy_Rate", 10), "Literacy_Rate", title="Top 10 Districts by Literacy Rate (%)"), use_container_width=True)
        with c2:
            st.plotly_chart(V.bar_h(B.rank_districts(d, "Literacy_Rate", 10, ascending=True), "Literacy_Rate", title="Bottom 10 Districts by Literacy Rate (%)"), use_container_width=True)

        sl = B.state_literacy(d, basis)
        st.plotly_chart(V.state_bar(sl, "Weighted_Literacy_Rate", title="State Literacy Rate (Population-Weighted)", xlabel="Weighted Literacy Rate (%)"), use_container_width=True)
        with st.expander("View State Literacy Data Table"):
            st.dataframe(sl.style.format({"Weighted_Literacy_Rate": "{:.2f}%", "Unweighted_Mean_District_Literacy": "{:.2f}%"}), hide_index=True, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.bar_h(B.rank_districts(d, "Sex_Ratio", 10), "Sex_Ratio", title="Top 10 Districts by Sex Ratio"), use_container_width=True)
        with c2:
            st.plotly_chart(V.bar_h(B.rank_districts(d, "Sex_Ratio", 10, ascending=True), "Sex_Ratio", title="Bottom 10 Districts by Sex Ratio"), use_container_width=True)

        st.plotly_chart(V.scatter(d, "SC_Population_Share", "ST_Population_Share", color="state_name", title="SC Share (%) vs ST Share (%) across Districts"), use_container_width=True)

    with tab3:
        b_vars = B.B_VARS
        bx1, bx2 = st.columns(2)
        x_var = bx1.selectbox("Select Independent Variable (X-axis)", b_vars, index=2)
        y_var = bx2.selectbox("Select Dependent Variable (Y-axis)", b_vars, index=0)
        st.plotly_chart(V.scatter(d, x_var, y_var, color="state_name", trend=True, title=f"{V.lab(y_var)} vs {V.lab(x_var)} (with OLS Trendline)"), use_container_width=True)

        # Show pairwise stats
        pair_res = ST.pair_correlation(d, x_var, y_var)
        st.dataframe(pair_res.style.format({"Coefficient": "{:.4f}", "P-value": "{:.4e}"}), hide_index=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.corr_heatmap(ST.correlation_matrix(d, B.B_VARS, "pearson"), title="Pearson Correlation (Linear Association)"), use_container_width=True)
        with c2:
            st.plotly_chart(V.corr_heatmap(ST.correlation_matrix(d, B.B_VARS, "spearman"), title="Spearman Correlation (Monotonic Association)"), use_container_width=True)

        st.caption("Notice the strong negative correlation between Child Share and Literacy Rate (partly mechanical when using the crude denominator).")

    with tab5:
        st.markdown("### 🧪 Syllabus-Approved T-Tests & Z-Tests (Practical 5 — Notebook 1)")
        st.caption("Only Z-tests and t-tests specified in the practical syllabus are performed. Non-parametric alternatives (Mann-Whitney U) are strictly excluded.")

        # Sub-section 1: Two-Sample t-test on SC+ST vs Literacy
        st.markdown("#### 1. Independent Two-Sample T-Test (SC+ST Share vs Literacy Rate)")
        st.markdown(
            "**Purpose:** Test whether mean district literacy rate differs significantly between districts with an above-median SC+ST population share and those at or below the median.  \n"
            "**Null Hypothesis ($H_0$):** $\\mu_{\\text{above}} = \\mu_{\\text{below}}$ (Group mean literacy rates are equal)  \n"
            "**Alternative Hypothesis ($H_1$):** $\\mu_{\\text{above}} \\ne \\mu_{\\text{below}}$ (Group mean literacy rates differ, two-sided, $\\alpha = 0.05$)"
        )
        try:
            h_res = B.default_hypothesis(d)
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric("Applied Test", h_res["test"])
            col_h2.metric("t-Statistic", f"{h_res['statistic']:.3f}")
            col_h3.metric("p-value", f"{h_res['p_value']:.4e}", delta="Reject H0" if h_res["p_value"] < 0.05 else "Fail to Reject")
            col_h4.metric("Cohen's d (Effect)", f"{h_res['effect_size']:.3f}")

            # Group comparison metrics
            gm1, gm2, gm3 = st.columns(3)
            gm1.markdown(f"**Group 1 ({h_res['groups'][0]}):** Mean = `{h_res['means'][0]:.2f}%`, Std = `{h_res['stds'][0]:.2f}`, n = `{h_res['n'][0]}`")
            gm2.markdown(f"**Group 2 ({h_res['groups'][1]}):** Mean = `{h_res['means'][1]:.2f}%`, Std = `{h_res['stds'][1]:.2f}`, n = `{h_res['n'][1]}`")
            gm3.markdown(f"**Mean Difference:** `{h_res['mean_difference']:.2f}%` (95% CI: `[{h_res['ci_95'][0]:.2f}%, {h_res['ci_95'][1]:.2f}%]`)")

            # Assumption verification display
            st.markdown("##### Assumption Verification & Limitation Notes:")
            sh_a, sh_b = h_res["normality_p"]
            st.markdown(f"- **Normality (Shapiro-Wilk):** Group 1 p = `{sh_a:.4e}`, Group 2 p = `{sh_b:.4e}`")
            st.markdown(f"- **Homogeneity of Variance (Levene):** p = `{h_res['levene_p']:.4e}` (Unequal variance handled via Welch's Satterthwaite approximation)")

            if h_res.get("assumption_warnings"):
                for w in h_res["assumption_warnings"]:
                    st.warning(f"⚠️ {w}")

            if h_res["p_value"] < 0.05:
                st.success(f"**Statistical Decision:** {h_res['decision'].upper()} at α = 0.05. {h_res['interpretation']}")
            else:
                st.info(f"**Statistical Decision:** {h_res['decision'].upper()} at α = 0.05. {h_res['interpretation']}")

            st.caption("Critical Note: Rejecting H0 indicates a statistical association between marginalized community concentration and district literacy, NOT causation. It does not prove classroom deficits.")
        except Exception as e:
            st.error(f"Cannot perform two-sample t-test on current selection: {e}")

        st.markdown("---")
        # Sub-section 2: One-Sample T-test and Z-test against National Benchmark
        st.markdown("#### 2. One-Sample Hypothesis Tests against Policy Benchmark (T-test & Z-test)")
        st.markdown("Evaluate whether sample district indicators differ significantly from a known national benchmark $\\mu_0$.")

        t_cfg1, t_cfg2, t_cfg3 = st.columns(3)
        bench_var = t_cfg1.selectbox("Variable to Test", ["Literacy_Rate", "Sex_Ratio", "Workforce_Participation_Rate"], key="bench_var")
        bench_val = t_cfg2.number_input("Benchmark Value (mu_0)", value=65.0 if bench_var == "Literacy_Rate" else (950.0 if bench_var == "Sex_Ratio" else 40.0), key="bench_val")
        test_mode = t_cfg3.radio("Test Framework", ["One-Sample T-Test", "One-Sample Z-Test (Large Sample N>=30)"], horizontal=True)

        try:
            if "Z-Test" in test_mode:
                z_res = ST.one_sample_z_test(d, bench_var, float(bench_val))
                zc1, zc2, zc3 = st.columns(3)
                zc1.metric("Sample Mean", f"{z_res['sample_mean']:.2f}")
                zc2.metric("Z-Statistic", f"{z_res['statistic']:.3f}")
                zc3.metric("p-value", f"{z_res['p_value']:.4e}", delta="Reject H0" if z_res["p_value"] < 0.05 else "Fail to Reject")
                st.info(z_res["interpretation"])
            else:
                t1_res = ST.one_sample_t_test(d, bench_var, float(bench_val))
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Sample Mean", f"{t1_res['sample_mean']:.2f}")
                tc2.metric("t-Statistic", f"{t1_res['statistic']:.3f}")
                tc3.metric("p-value", f"{t1_res['p_value']:.4e}", delta="Reject H0" if t1_res["p_value"] < 0.05 else "Fail to Reject")
                tc4.metric("Cohen's d", f"{t1_res['effect_size']:.3f}")
                if t1_res.get("assumption_warnings"):
                    for w in t1_res["assumption_warnings"]:
                        st.warning(f"⚠️ {w}")
                st.info(t1_res["interpretation"])
        except Exception as e:
            st.error(f"Error performing one-sample test: {e}")

    with tab6:
        st.markdown("### 📐 Syllabus-Approved ANOVA & Chi-Square Tests (Practical 5 — Notebook 2)")
        st.caption("Only One-Way ANOVA, Two-Way ANOVA, and Chi-Square tests are performed. Kruskal-Wallis is strictly removed per syllabus requirements.")

        # Section 1: One-Way ANOVA across States
        st.markdown("#### 1. One-Way ANOVA across States")
        st.markdown(
            "**Purpose:** Test whether mean district literacy rates differ across states.  \n"
            "**Null Hypothesis ($H_0$):** $\\mu_1 = \\mu_2 = \\dots = \\mu_k$ (All included state means are equal)  \n"
            "**Alternative Hypothesis ($H_1$):** At least one state mean differs ($\\alpha = 0.05$)"
        )
        min_k = st.slider("Minimum districts per state to include", 3, 10, 5, key="anova_k")
        try:
            a_res = B.state_anova(d, min_k)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("States Evaluated", a_res["groups_used"])
            c2.metric("Districts (N)", a_res["n"])
            c3.metric("Classic F-Statistic", f"{a_res['classic_F']:.3f}")
            c4.metric("Classic p-value", f"{a_res['classic_p']:.4e}")

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Welch F-Statistic", f"{a_res['welch_F']:.3f}")
            wc2.metric("Welch p-value", f"{a_res['welch_p']:.4e}")
            wc3.metric("Effect Size (Eta-Squared)", f"{a_res['eta_squared']:.3f}")

            st.markdown("##### Assumption Checks & Reporting:")
            st.markdown(f"- **Residual Normality (Shapiro-Wilk):** p = `{a_res['residual_shapiro_p']:.4e}`")
            st.markdown(f"- **Equal Variances (Levene Test):** p = `{a_res['levene_p']:.4e}`")

            if a_res.get("assumption_warnings"):
                for w in a_res["assumption_warnings"]:
                    st.warning(f"⚠️ {w}")

            st.markdown(f"**Primary Test Selected:** `{a_res['primary_test']}` — Decision: **{a_res['decision'].upper()}**")
            st.info(a_res["interpretation"])
            st.caption(f"Spatial nestedness note: {a_res['independence_note']}")

            with st.expander("Full ANOVA Metrics Table (Syllabus Compliant)"):
                st.json({k: v for k, v in a_res.items() if k not in ("interpretation", "independence_note")})
        except Exception as e:
            st.error(f"ANOVA calculation error: {e}")

        st.markdown("---")
        # Section 2: Two-Way ANOVA (Factorial Interaction)
        st.markdown("#### 2. Two-Way ANOVA: SC+ST Share Group and Workforce Tier on Literacy")
        st.markdown(
            "**Purpose:** Evaluate the main effects of SC+ST Concentration Tier and Workforce Participation Tier, along with their interaction effect, on District Literacy Rate.  \n"
            "**Hypotheses:**  \n"
            "- $H_{0,A}$: No main effect of SC+ST concentration tier.  \n"
            "- $H_{0,B}$: No main effect of Workforce participation tier.  \n"
            "- $H_{0,AB}$: No interaction effect between SC+ST concentration and workforce participation."
        )
        try:
            aov2 = B.two_way_anova_literacy(d)
            st.dataframe(aov2["anova_table"].style.format({
                "Sum_of_Squares": "{:,.2f}",
                "df": "{:.0f}",
                "F_statistic": "{:.3f}",
                "p_value": "{:.4e}",
                "partial_eta_squared": "{:.4f}",
            }), hide_index=True, use_container_width=True)

            if aov2.get("assumption_warnings"):
                for w in aov2["assumption_warnings"]:
                    st.warning(f"⚠️ {w}")
            st.info(aov2["interpretation"])
        except Exception as e:
            st.error(f"Two-Way ANOVA calculation error: {e}")

        st.markdown("---")
        # Section 3: Chi-Square Test of Independence
        st.markdown("#### 3. Chi-Square Test of Independence (Literacy Tier vs SC+ST Concentration Group)")
        st.markdown(
            "**Purpose:** Test whether district literacy level tier (Low <60%, Medium 60-75%, High ≥75%) is statistically independent of SC+ST concentration (Above-median vs At/below-median).  \n"
            "**Null Hypothesis ($H_0$):** Literacy Tier and SC+ST Concentration Group are independent.  \n"
            "**Alternative Hypothesis ($H_1$):** Literacy Tier and SC+ST Concentration Group are dependent (statistically associated, $\\alpha = 0.05$)."
        )
        try:
            chi_res = B.chi_square_literacy_sc_st(d)
            chi_c1, chi_c2, chi_c3, chi_c4 = st.columns(4)
            chi_c1.metric("Chi-Square (chi2)", f"{chi_res['statistic']:.3f}")
            chi_c2.metric("Degrees of Freedom", chi_res["df"])
            chi_c3.metric("p-value", f"{chi_res['p_value']:.4e}", delta="Reject H0" if chi_res["p_value"] < 0.05 else "Fail to Reject")
            chi_c4.metric("Cramer's V (Effect)", f"{chi_res['cramers_v']:.3f}")

            st.markdown("##### Observed Contingency Table:")
            st.dataframe(chi_res["contingency_table"], use_container_width=True)

            st.markdown("##### Expected Frequencies (Cochran's Assumption Check):")
            st.dataframe(chi_res["expected_frequencies"].style.format("{:.2f}"), use_container_width=True)

            if chi_res.get("assumption_warnings"):
                for w in chi_res["assumption_warnings"]:
                    st.warning(f"⚠️ {w}")
            else:
                st.caption("✅ Cochran's condition satisfied: all expected cell counts are >= 5.")

            st.success(f"**Decision:** {chi_res['decision'].upper()} at α = 0.05. {chi_res['interpretation']}")
        except Exception as e:
            st.error(f"Chi-square test error: {e}")


# -----------------------------------------------------------------------------
# 7. Page 4: Module C - AI Literacy Level Classification (Random Forest)
# -----------------------------------------------------------------------------
elif page.startswith("🌲"):
    st.title("Module C — AI Literacy Classification & Accuracy Optimization")
    st.markdown(
        "Machine Learning classification of district literacy level into **Low**, **Medium**, and **High** tiers "
        "using strictly non-literacy demographic features. Benchmark your chosen model against a Dummy Majority Baseline."
    )

    # Advanced Model & Feature Configuration
    st.markdown("#### ⚙️ Model Architecture & Feature Engineering Settings")
    c_m1, c_m2, c_m3 = st.columns(3)
    model_choice = c_m1.selectbox(
        "Model Algorithm:",
        [
            "🤝 Soft-Voting Ensemble (RF + GB) [Highest Accuracy]",
            "⚡ Gradient Boosting (Iterative Residuals)",
            "🌲 Random Forest (Balanced Cost-Sensitive)",
            "🪵 Random Forest (Standard Uniform)",
        ],
        index=0,
    )
    feature_choice = c_m2.selectbox(
        "Demographic Feature Set:",
        [
            "🚀 Enhanced Demographics (9 Features: includes Child Sex Ratio, Female Workforce, Dependency, Log Pop)",
            "📁 Base Demographics (5 Features: Sex Ratio, Child Share, Workforce, SC, ST)",
        ],
        index=0,
    )
    t_size = c_m3.select_slider("Test Set Split", [0.15, 0.20, 0.25, 0.30], value=0.20)

    # Map settings
    if "Soft-Voting" in model_choice:
        m_type = "ensemble"
        cw = "balanced"
    elif "Gradient Boosting" in model_choice:
        m_type = "gradient_boosting"
        cw = None
    elif "Balanced" in model_choice:
        m_type = "rf"
        cw = "balanced"
    else:
        m_type = "rf"
        cw = None

    f_set = "enhanced" if "Enhanced" in feature_choice else "base"

    c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
    low_th = c_cfg1.number_input("Low Literacy Threshold (< %)", 40.0, 70.0, 60.0, 1.0)
    high_th = c_cfg2.number_input("High Literacy Threshold (≥ %)", 65.0, 90.0, 75.0, 1.0)
    rand_seed = c_cfg3.number_input("Random Seed", 0, 9999, 42)

    if low_th >= high_th:
        st.error("Low threshold must be strictly less than High threshold.")
        st.stop()

    # Model training trigger
    st.markdown("---")
    res = run_model(basis, low_th, high_th, t_size, int(rand_seed), model_type=m_type, feature_set=f_set, class_weight=cw)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance & Baseline",
        "🎯 Confusion Matrix & Importances",
        "🔮 Interactive 'What-If' Simulator",
        "📋 Test Set Predictions Table",
    ])

    with tab1:
        st.markdown(f"#### Performance of **{res['model_name']}** vs Dummy Majority Baseline")
        m_rf = res["rf"]
        m_base = res["baseline"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Test Accuracy", f"{m_rf['accuracy']*100:.2f}%", delta=f"{(m_rf['accuracy'] - m_base['accuracy'])*100:+.2f}% vs Baseline")
        col2.metric("Balanced Accuracy", f"{m_rf['balanced_accuracy']*100:.2f}%", delta=f"{(m_rf['balanced_accuracy'] - m_base['balanced_accuracy'])*100:+.2f}% vs Baseline")
        col3.metric("Macro-Averaged F1", f"{m_rf['macro_f1']:.4f}", delta=f"{m_rf['macro_f1'] - m_base['macro_f1']:+.4f} vs Baseline")

        if res["cv_macro_f1"] is not None:
            st.info(f"**Stratified 5-Fold Training CV Macro-F1:** {res['cv_macro_f1'].mean():.4f} ± {res['cv_macro_f1'].std():.4f}")

        st.markdown("##### Per-Class Performance Breakdown")
        st.dataframe(m_rf["per_class"].style.format("{:.3f}"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.confusion_heatmap(m_rf["confusion"]), use_container_width=True)
        with c2:
            st.plotly_chart(V.importance_bar(res["importance"], title=f"Feature Importance ({res['model_name']})"), use_container_width=True)

        st.dataframe(res["importance"].style.format({"Impurity_Importance": "{:.4f}", "Permutation_Importance": "{:.4f}"}), hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("#### 🔮 Interactive District Demographic What-If Simulator")
        st.markdown(f"Simulate demographic attributes using the trained **{res['model_name']}**.")

        # Preset or manual slider
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            preset_district = st.selectbox("Optionally load values from an existing district:", ["(Custom Values)"] + sorted(df["District_Label"].tolist()))

        row_match = df[df["District_Label"] == preset_district].iloc[0] if preset_district != "(Custom Values)" else None

        active_feats = res["features"]
        sim_inputs = {}

        st.markdown("##### Enter Feature Values:")
        sim_cols = st.columns(min(len(active_feats), 5))

        slider_meta = {
            "Sex_Ratio": ("Sex Ratio", 700.0, 1200.0, 940.0, 5.0),
            "Child_Population_Share": ("Child Share (%)", 5.0, 30.0, 13.5, 0.5),
            "Workforce_Participation_Rate": ("Workforce Rate (%)", 20.0, 70.0, 40.0, 0.5),
            "SC_Population_Share": ("SC Share (%)", 0.0, 60.0, 15.0, 0.5),
            "ST_Population_Share": ("ST Share (%)", 0.0, 100.0, 8.0, 0.5),
            "Child_Sex_Ratio": ("Child Sex Ratio", 700.0, 1100.0, 920.0, 5.0),
            "Female_Workforce_Rate": ("Female Workforce (%)", 5.0, 65.0, 25.0, 0.5),
            "Dependency_Ratio": ("Dependency Ratio", 0.8, 3.5, 1.8, 0.05),
            "Log_Population": ("Log10 Population", 4.0, 7.5, 6.2, 0.1),
        }

        for idx, feat in enumerate(active_feats):
            col_target = sim_cols[idx % len(sim_cols)]
            if feat in slider_meta:
                label, mn, mx, fallback, step = slider_meta[feat]
                val = float(row_match[feat]) if row_match is not None and feat in row_match else float(df[feat].median() if feat in df else fallback)
                sim_inputs[feat] = col_target.slider(label, mn, mx, float(np.clip(val, mn, mx)), step, key=f"sim_{feat}")
            else:
                val = float(row_match[feat]) if row_match is not None and feat in row_match else float(df[feat].median() if feat in df else 0.0)
                sim_inputs[feat] = col_target.number_input(feat, value=val, key=f"sim_{feat}")

        sample_in = pd.DataFrame([sim_inputs])

        model_pipeline = res["model"]
        pred_class = model_pipeline.predict(sample_in)[0]
        pred_probs = model_pipeline.predict_proba(sample_in)[0]
        classes = list(model_pipeline.classes_)

        st.markdown(f"### Predicted Literacy Tier: **`{pred_class}`**")
        st.plotly_chart(V.probability_bar(classes, pred_probs, title="Class Probabilities Breakdown"), use_container_width=True)

    with tab4:
        st.markdown("##### Test Set Sample Predictions")
        st.dataframe(res["test_table"], hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# 8. Page 5: Module D - Urban Infrastructure Planning Priority Score (UIPS)
# -----------------------------------------------------------------------------
elif page.startswith("🎯"):
    st.title("Module D — Demographic Infrastructure Planning Priority Score (UIPS)")
    st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Screening Caveat:</strong> {NOTICE}</div>', unsafe_allow_html=True)

    st.markdown("""
    **Mathematical Formulation:**
    $$\\text{UIPS} = 100 \\times \\left( w_P \\cdot P_{\\text{norm}} + w_C \\cdot C_{\\text{norm}} + w_L \\cdot L_{\\text{norm}} + w_W \\cdot W_{\\text{norm}} \\right)$$
    Where:
    - $P_{\\text{norm}}$: Normalized Total Population
    - $C_{\\text{norm}}$: Normalized Child Population Share (Age 0-6)
    - $L_{\\text{norm}}$: Normalized Literacy Gap ($100 - \\text{Literacy Rate}$)
    - $W_{\\text{norm}}$: Normalized Workforce Participation (lower participation treated as higher priority)
    """)

    # Interactive Weights Section
    st.markdown('<div class="section-header"><h3>⚙️ Interactive Weight Configuration & Scenarios</h3></div>', unsafe_allow_html=True)

    scenario_choice = st.selectbox(
        "Choose a Weight Scenario Preset:",
        list(D.SCENARIOS.keys()) + ["Custom Weights"],
        index=0,
    )

    if scenario_choice in D.SCENARIOS:
        preset_w = D.SCENARIOS[scenario_choice]
    else:
        preset_w = D.DEFAULT_WEIGHTS

    w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
    w_p = w_col1.slider("Population (P)", 0.0, 1.0, float(preset_w.get("Total_Population", 0.35)), 0.05)
    w_c = w_col2.slider("Child Share (C)", 0.0, 1.0, float(preset_w.get("Child_Population_Share", 0.25)), 0.05)
    w_l = w_col3.slider("Literacy Gap (L)", 0.0, 1.0, float(preset_w.get("Literacy_Gap", 0.25)), 0.05)
    w_w = w_col4.slider("Workforce (W)", 0.0, 1.0, float(preset_w.get("Workforce_Participation_Rate", 0.15)), 0.05)
    winsorize = w_col5.checkbox("Winsorize (1-99%)", value=False, help="Clip extreme 1% outliers before min-max scaling.")

    active_weights = {
        "Total_Population": w_p,
        "Child_Population_Share": w_c,
        "Literacy_Gap": w_l,
        "Workforce_Participation_Rate": w_w,
    }

    try:
        norm_weights = D.normalize_weights(active_weights)
        uips_df, uips_info = D.compute_uips(df, active_weights, winsor=(1, 99) if winsorize else None)
    except ValueError as ve:
        st.error(str(ve))
        st.stop()

    st.caption(f"Normalized Effective Weights: {', '.join([f'{k}: {v:.3f}' for k, v in norm_weights.items()])}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Priority Leaderboard",
        "📊 Score Distributions",
        "🔬 Sensitivity Analysis",
        "🎯 Single District Score Breakdown",
        "📥 Export Full Priority Dataset",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.bar_h(uips_df.head(10), "UIPS", title="Top 10 High Priority Screening Districts"), use_container_width=True)
        with c2:
            st.plotly_chart(V.bar_h(uips_df.dropna(subset=["UIPS"]).tail(10), "UIPS", title="Bottom 10 Low Priority Screening Districts"), use_container_width=True)

        st.plotly_chart(V.category_bar(uips_df), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(V.histogram(uips_df, "UIPS", title="UIPS Priority Score Distribution"), use_container_width=True)
        with c2:
            st.plotly_chart(V.box(uips_df, "UIPS", title="UIPS Box Plot"), use_container_width=True)

        cols_needed = ["Total_Population", "Literacy_Gap"]
        missing_in_uips = [c for c in cols_needed if c not in uips_df.columns]
        if missing_in_uips:
            full_uips = uips_df.merge(df[["District_Label"] + missing_in_uips], on="District_Label", how="left")
        else:
            full_uips = uips_df.copy()
        s1, s2 = st.columns(2)
        with s1:
            st.plotly_chart(V.scatter(full_uips, "Total_Population", "UIPS", title="Total Population vs UIPS"), use_container_width=True)
        with s2:
            st.plotly_chart(V.scatter(full_uips, "Literacy_Gap", "UIPS", title="Literacy Gap vs UIPS"), use_container_width=True)

    with tab3:
        st.markdown("#### Weight-Sensitivity Matrix across 5 Standard Scenarios")
        st.markdown("Assesses the robustness of district priority rankings across differing policy emphases.")
        S_, R_, C_, summ, corr, ch = D.sensitivity_analysis(df, winsor=(1, 99) if winsorize else None)

        st.dataframe(summ, hide_index=True, use_container_width=True)
        st.markdown("##### Spearman Rank Correlation between Scenarios")
        st.plotly_chart(V.corr_heatmap(corr, title="Spearman Rank Correlation across UIPS Scenarios"), use_container_width=True)

        st.info(f"**Category Volatility:** {len(ch)} out of 640 districts shift priority category (Low / Medium / High) depending on the weighting scenario.")
        with st.expander("Inspect Districts Shifting Categories"):
            st.dataframe(ch, use_container_width=True)

    with tab4:
        st.markdown("#### Single District UIPS Component Breakdown")
        pick_d = st.selectbox("Select District:", sorted(uips_df["District_Label"].unique()))
        d_row = uips_df[uips_df["District_Label"] == pick_d].iloc[0]

        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.plotly_chart(V.gauge_chart(d_row["UIPS"], title=pick_d, subtitle=f"Rank #{int(d_row['Rank'])} of {len(uips_df)} ({d_row['Priority_Category']} Priority)"), use_container_width=True)

        with d_col2:
            contrib_data = pd.DataFrame([
                {"Component": "Population", "Percentage Points": d_row.get("P_contribution", 0)},
                {"Component": "Child Share", "Percentage Points": d_row.get("C_contribution", 0)},
                {"Component": "Literacy Gap", "Percentage Points": d_row.get("L_contribution", 0)},
                {"Component": "Workforce Participation", "Percentage Points": d_row.get("W_contribution", 0)},
            ])
            st.dataframe(contrib_data.style.format({"Percentage Points": "{:.2f}"}), hide_index=True, use_container_width=True)

    with tab5:
        st.markdown("#### Complete Priority Dataset")
        st.dataframe(uips_df, hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Download UIPS Leaderboard (CSV)",
            uips_df.to_csv(index=False).encode(),
            "uips_priority_ranking.csv",
            "text/csv",
        )


# -----------------------------------------------------------------------------
# 9. Page 6: District Intelligence Dossier
# -----------------------------------------------------------------------------
elif page.startswith("🔎"):
    st.title("District Intelligence Dossier & Comparative Benchmarking")
    st.markdown("Inspect comprehensive demographic and priority scorecards for any individual district or benchmark two districts side-by-side.")

    mode = st.radio("Mode", ["Individual District Deep Dive", "Dual District Side-by-Side Comparison"], horizontal=True)

    uips_all, _ = D.compute_uips(df)
    merged = df.merge(uips_all[["District_Label", "UIPS", "Rank", "Priority_Category"]], on="District_Label")

    if mode.startswith("Individual"):
        selected_d = st.selectbox("Select District to Inspect:", sorted(merged["District_Label"].unique()))
        row = merged[merged["District_Label"] == selected_d].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><div class="stat-label">Total Population</div><div class="stat-val">{row["Total_Population"]:,.0f}</div><div class="stat-sub">State: {row["state_name"]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-label">Literacy Rate</div><div class="stat-val">{row["Literacy_Rate"]:.1f}%</div><div class="stat-sub">National Median: {df["Literacy_Rate"].median():.1f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-label">Sex Ratio</div><div class="stat-val">{row["Sex_Ratio"]:.0f}</div><div class="stat-sub">National Median: {df["Sex_Ratio"].median():.0f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-label">UIPS Priority Score</div><div class="stat-val">{row["UIPS"]:.1f}</div><div class="stat-sub">Rank #{int(row["Rank"])} ({row["Priority_Category"]})</div></div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(V.gauge_chart(row["UIPS"], title="UIPS Priority Level", subtitle=f"Rank #{int(row['Rank'])} of 640"), use_container_width=True)
        with g2:
            radar_dict = {
                "Population (norm)": float(np.clip(row["Total_Population"] / df["Total_Population"].max() * 100, 0, 100)),
                "Child Share (norm)": float(np.clip(row["Child_Population_Share"] / df["Child_Population_Share"].max() * 100, 0, 100)),
                "Literacy Rate": float(row["Literacy_Rate"]),
                "Sex Ratio (norm)": float(np.clip(row["Sex_Ratio"] / 1100 * 100, 0, 100)),
                "Workforce Rate": float(row["Workforce_Participation_Rate"]),
            }
            st.plotly_chart(V.radar_profile(selected_d, radar_dict), use_container_width=True)

    else:
        d1_col, d2_col = st.columns(2)
        d1 = d1_col.selectbox("District A:", sorted(merged["District_Label"].unique()), index=0)
        d2 = d2_col.selectbox("District B:", sorted(merged["District_Label"].unique()), index=1)

        r1 = merged[merged["District_Label"] == d1].iloc[0]
        r2 = merged[merged["District_Label"] == d2].iloc[0]

        comp_metrics = [
            ("State", r1["state_name"], r2["state_name"]),
            ("Total Population", f"{r1['Total_Population']:,.0f}", f"{r2['Total_Population']:,.0f}"),
            ("Child Population Share", f"{r1['Child_Population_Share']:.2f}%", f"{r2['Child_Population_Share']:.2f}%"),
            ("Literacy Rate", f"{r1['Literacy_Rate']:.2f}%", f"{r2['Literacy_Rate']:.2f}%"),
            ("Sex Ratio", f"{r1['Sex_Ratio']:.0f}", f"{r2['Sex_Ratio']:.0f}"),
            ("Workforce Participation", f"{r1['Workforce_Participation_Rate']:.2f}%", f"{r2['Workforce_Participation_Rate']:.2f}%"),
            ("SC Population Share", f"{r1['SC_Population_Share']:.2f}%", f"{r2['SC_Population_Share']:.2f}%"),
            ("ST Population Share", f"{r1['ST_Population_Share']:.2f}%", f"{r2['ST_Population_Share']:.2f}%"),
            ("UIPS Priority Score", f"{r1['UIPS']:.2f}", f"{r2['UIPS']:.2f}"),
            ("National Rank", f"#{int(r1['Rank'])}", f"#{int(r2['Rank'])}"),
            ("Priority Category", r1["Priority_Category"], r2["Priority_Category"]),
        ]

        comp_df = pd.DataFrame(comp_metrics, columns=["Indicator", d1, d2])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# 10. Page 7: Dataset Explorer & Audit
# -----------------------------------------------------------------------------
elif page.startswith("📁"):
    st.title("Dataset Explorer & Data Quality Audit")
    st.markdown("Verified against Hugging Face repository `indiaset/census-2011`. 640 districts, 29 original attributes.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Data Dictionary & Mappings")
        st.dataframe(pp.data_dictionary(raw), hide_index=True, use_container_width=True)
    with col_b:
        st.subheader("Schema Verification Report")
        st.dataframe(pp.schema_report(raw), hide_index=True, use_container_width=True)

    st.subheader("Data Quality Checks & Denominator Verification")
    st.json({k: v for k, v in quality.items() if k != "notes"})
    for n in quality["notes"]:
        st.info(f"💡 {n}")

    st.subheader("Searchable Working Dataset")
    search_q = st.text_input("Search by District Name:")
    view_df = df[df["District_Label"].str.contains(search_q, case=False, na=False)] if search_q else df
    st.dataframe(view_df, hide_index=True, use_container_width=True)

    st.download_button(
        "📥 Download Cleaned Working Dataset (CSV)",
        view_df.to_csv(index=False).encode(),
        "census_2011_cleaned_working_data.csv",
        "text/csv",
    )


# -----------------------------------------------------------------------------
# 11. Page 8: Methodology & Limitations
# -----------------------------------------------------------------------------
else:
    st.title("Methodology, Formulations & Limitations")
    st.markdown(
        "### 1. Data Provenance & Scope\n"
        "- **Source:** Hugging Face IndiaSet Census 2011 (`indiaset/census-2011`).\n"
        "- **Granularity:** 640 district administrative units.\n"
        f"- **Notice:** {NOTICE}\n\n"
        "### 2. Variable Definitions\n"
        "- **Child Population:** Age group 0-6 years (`pop_under6_total`).\n"
        "- **Sex Ratio:** Females per 1,000 males (`pop_female / pop_male * 1000`).\n"
        "- **Workforce Participation:** Total workers divided by total population (`workers_total / pop_total * 100`).\n"
        "- **Literacy Rate:**\n"
        "  - *Source Basis:* Matches repository crude rate: Literacy Rate = (Literate Total / Total Population) * 100.\n"
        "  - *Effective Basis:* Official census definition for age 7+: Literacy Rate = (Literate Total / (Total Population - Child Population)) * 100.\n\n"
        "### 3. Module Methodologies\n"
        "- **Module A:** Population descriptive summary, skewness evaluation, top-10 concentration metrics.\n"
        "- **Module B:** Bivariate correlation (Pearson & Spearman), Two-Sample t-test / Welch's t-test (SC+ST share median split), One-Sample t/Z-tests against benchmarks, One-Way ANOVA across states, Two-Way ANOVA (SC+ST tier and workforce tier interaction), and Chi-Square test of independence (Literacy tier vs SC+ST tier). All tests strictly adhere to the practical syllabus, displaying formal hypotheses, assumption checks (Shapiro-Wilk, Levene, Cochran's condition), and limitation warnings without substituting unapproved non-parametric tests.\n"
        "- **Module C:** Random Forest Classifier predicting Low (<60%), Medium (60-75%), and High (≥75%) literacy tiers using only non-literacy demographic features. Evaluated against a Dummy Majority Baseline.\n"
        "- **Module D:** Urban Infrastructure Planning Priority Score:\n\n"
        "  $$\\text{UIPS} = 100 \\times (0.35 P + 0.25 C + 0.25 L + 0.15 W)$$\n\n"
        "  Normalized using Min-Max scaling, with 5 multi-scenario weight sensitivity simulations.\n\n"
        "### 4. Limitations\n"
        "- Census 2011 reflects 2011 conditions. Subsequent district bifurcations and demographic shifts are not reflected.\n"
        "- No land area or geospatial boundary polygons are included in the dataset; geographic density and choropleth maps cannot be calculated without external GIS datasets.\n"
        "- District aggregates obscure intra-district rural/urban and neighborhood disparities.\n"
    )
