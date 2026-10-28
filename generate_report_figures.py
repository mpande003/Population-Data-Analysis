import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_loader import load_raw
from preprocessing import clean_data
import module_c
import module_d

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

raw, _ = load_raw()
df, _ = clean_data(raw)

# 1. Figure 1: Class Distribution
plt.figure(figsize=(6, 4))
cats = ['Low (<60%)', 'Medium (60-75%)', 'High (>=75%)']
counts = [261, 301, 78]
bars = plt.bar(cats, counts, color=['#4a5568', '#718096', '#2b6cb0'], edgecolor='black', width=0.55)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, y + 5, f'{y} ({y/640*100:.1f}%)', ha='center', va='bottom', fontsize=9)
plt.title('Figure 1: Distribution of Districts Across Literacy Tiers (N = 640)', fontsize=11, fontweight='bold', pad=12)
plt.ylabel('Number of Districts')
plt.ylim(0, 350)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/figure1_class_distribution.png', dpi=300)
plt.close()

# 2. Figure 2: Welch t-test
plt.figure(figsize=(6, 4.5))
med_scst = df['SC_Population_Share'] + df['ST_Population_Share']
df_t = df.copy()
df_t['SCST_Group'] = np.where(med_scst > med_scst.median(), 'Above Median (>28.4%)', 'At/Below Median (<=28.4%)')
sns.boxplot(x='SCST_Group', y='Literacy_Rate', hue='SCST_Group', data=df_t, palette=['#cbd5e0', '#4a5568'], width=0.4, boxprops=dict(edgecolor='black'), medianprops=dict(color='black', linewidth=1.5), legend=False)
plt.title("Figure 2: Literacy Rate Comparison by SC+ST Population Share\n(Welch's t = -4.201, df = 628.6, p = 3.04e-5, Cohen's d = -0.332)", fontsize=10, fontweight='bold', pad=10)
plt.ylabel('Effective Literacy Rate (%)')
plt.xlabel('SC+ST Population Proportion')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/figure2_welch_t_test.png', dpi=300)
plt.close()

# 3. Figure 3: Model Comparison
models = ['Dummy\nBaseline', 'Standard\nRandom Forest', 'Balanced\nRandom Forest', 'Gradient\nBoosting', 'Soft-Voting\nEnsemble']
acc = [46.88, 72.66, 74.22, 75.78, 76.56]
f1 = [21.28, 64.98, 66.06, 69.11, 69.69]
x = np.arange(len(models))
width = 0.35
plt.figure(figsize=(7, 4.5))
plt.bar(x - width/2, acc, width, label='Accuracy (%)', color='#2b6cb0', edgecolor='black')
plt.bar(x + width/2, f1, width, label='Macro F1 (%)', color='#718096', edgecolor='black')
for i in range(len(models)):
    plt.text(x[i] - width/2, acc[i] + 1.2, f'{acc[i]:.1f}%', ha='center', fontsize=8)
    plt.text(x[i] + width/2, f1[i] + 1.2, f'{f1[i]:.1f}%', ha='center', fontsize=8)
plt.ylabel('Score (%)')
plt.title('Figure 3: Classification Model Benchmark on Test Set (N = 128)', fontsize=11, fontweight='bold', pad=12)
plt.xticks(x, models, fontsize=9)
plt.ylim(0, 90)
plt.legend(frameon=True, facecolor='#f7fafc', edgecolor='#cbd5e0')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/figure3_model_comparison.png', dpi=300)
plt.close()

# 4. Figure 4: Confusion Matrix
res = module_c.train_evaluate(df, model_type='ensemble', feature_set='enhanced', class_weight='balanced')
cm = res['model_metrics']['confusion']
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 12, 'weight': 'bold'}, linewidths=1, linecolor='black')
plt.title('Figure 4: Confusion Matrix - Soft-Voting Ensemble (N = 128)', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Predicted Class', fontweight='bold')
plt.ylabel('True Class', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/figure4_confusion_matrix.png', dpi=300)
plt.close()

# 5. Figure 5: Feature Importance
imp = res['importance'].sort_values('Impurity_Importance', ascending=True)
plt.figure(figsize=(7, 4.5))
plt.barh(imp['Feature'], imp['Impurity_Importance'] * 100, color='#4a5568', edgecolor='black', height=0.6)
for i, val in enumerate(imp['Impurity_Importance'] * 100):
    plt.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8.5)
plt.title('Figure 5: Predictor Importance Ranking (Soft-Voting Ensemble)', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Relative Importance (%)')
plt.xlim(0, 45)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/figure5_feature_importance.png', dpi=300)
plt.close()

# 6. Figure 6: UIPS Distribution
uips_df, _ = module_d.compute_uips(df)
mean_val = float(uips_df['UIPS'].mean())
plt.figure(figsize=(6, 4))
plt.hist(uips_df['UIPS'], bins=25, color='#718096', edgecolor='black', alpha=0.8)
plt.axvline(mean_val, color='#c53030', linestyle='--', linewidth=1.5, label=f'Mean UIPS = {mean_val:.1f}')
plt.title('Figure 6: Urban Infrastructure Priority Score (UIPS) Distribution', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('UIPS Score (0 - 100)')
plt.ylabel('Number of Districts')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/figure6_uips_distribution.png', dpi=300)
plt.close()

print('SUCCESS: All 6 figures generated.')
