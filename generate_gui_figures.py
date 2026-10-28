import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'serif'

# Figure 7: GUI Interface Overview
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Outer container
ax.add_patch(patches.Rectangle((1, 1), 98, 58, facecolor='#f8fafc', edgecolor='#cbd5e0', linewidth=1.5))

# Sidebar
ax.add_patch(patches.Rectangle((1, 1), 24, 58, facecolor='#edf2f7', edgecolor='#cbd5e0', linewidth=1))
ax.text(3, 55, 'SI LAB DASHBOARD', fontsize=9, fontweight='bold', color='#1a202c')
ax.text(3, 51, 'Census 2011 Planning Suite', fontsize=7.5, color='#718096')
ax.text(3, 45, 'NAVIGATION', fontsize=8, fontweight='bold', color='#4a5568')
tabs = ['Executive Summary', 'Mod A: Population', 'Mod B: Statistical Tests', 'Mod C: Literacy ML', 'Mod D: UIPS Priority', 'Comprehensive Report']
for idx, tab in enumerate(tabs):
    bg = '#2b6cb0' if idx == 3 else '#ffffff'
    tc = '#ffffff' if idx == 3 else '#2d3748'
    ax.add_patch(patches.Rectangle((3, 38 - idx*4.5), 20, 3.8, facecolor=bg, edgecolor='#cbd5e0', linewidth=0.8))
    ax.text(4.5, 39.5 - idx*4.5, tab, fontsize=7.5, color=tc, fontweight='bold' if idx==3 else 'normal')

ax.text(3, 11, 'GLOBAL FILTERS', fontsize=8, fontweight='bold', color='#4a5568')
ax.text(3, 7.5, 'Literacy Basis: Effective', fontsize=7.5, color='#2d3748')
ax.text(3, 4, 'State: All 35 States / UTs', fontsize=7.5, color='#2d3748')

# Main Content Area - Module C
ax.text(28, 54, 'Module C: Supervised Literacy Tier Classification', fontsize=12, fontweight='bold', color='#1a202c')
ax.text(28, 50.5, 'Multi-model benchmark & zero-leakage demographic inference pipeline', fontsize=8.5, color='#718096')

# 3 Metric Cards
cards = [
    ('Ensemble Accuracy', '76.56%', '+29.68% over baseline', '#2b6cb0'),
    ('Macro F1-Score', '0.6969', 'Balanced class representation', '#2f855a'),
    ('Target Leakage', '0 Features', 'Strict scientific separation', '#4a5568')
]
for idx, (title, val, sub, col) in enumerate(cards):
    cx = 28 + idx * 23.5
    ax.add_patch(patches.Rectangle((cx, 38), 22, 10, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1))
    ax.text(cx + 2, 45.5, title, fontsize=8, color='#718096')
    ax.text(cx + 2, 41.5, val, fontsize=13, fontweight='bold', color=col)
    ax.text(cx + 2, 39, sub, fontsize=6.5, color='#a0aec0')

# Control Sliders / Dropdowns
ax.add_patch(patches.Rectangle((28, 24), 69, 12, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1))
ax.text(30, 33, 'Model Configuration & Threshold Tuners', fontsize=8.5, fontweight='bold', color='#2d3748')
ax.text(30, 29.5, 'Model Architecture: [ Soft-Voting Ensemble (Random Forest + Gradient Boosting) v ]', fontsize=7.5, color='#4a5568')
ax.text(30, 26.5, 'Feature Set: [ Enhanced 9 Predictors v ]    Class Weighting: [ Balanced Cost-Sensitive v ]', fontsize=7.5, color='#4a5568')

# Bottom preview table
ax.add_patch(patches.Rectangle((28, 3), 69, 19, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1))
ax.text(30, 19, 'Held-Out Test Set Predictions (First 4 of 128 Districts)', fontsize=8.5, fontweight='bold', color='#2d3748')
headers = ['District', 'Actual', 'Predicted', 'P(Low)', 'P(Med)', 'P(High)', 'Status']
for j, h in enumerate(headers):
    ax.text(30 + j*9.5, 16, h, fontsize=7, fontweight='bold', color='#4a5568')
rows = [
    ('Adilabad (Andhra Pradesh)', 'Medium', 'Medium', '0.18', '0.68', '0.14', 'Correct'),
    ('Agra (Uttar Pradesh)', 'Medium', 'Medium', '0.22', '0.71', '0.07', 'Correct'),
    ('Ahmedabad (Gujarat)', 'High', 'High', '0.04', '0.28', '0.68', 'Correct'),
    ('Aligarh (Uttar Pradesh)', 'Medium', 'Low', '0.54', '0.42', '0.04', 'Misclassified')
]
for r_i, row in enumerate(rows):
    y_pos = 13 - r_i * 2.8
    for c_i, val in enumerate(row):
        col_c = '#2f855a' if val == 'Correct' else ('#c53030' if val == 'Misclassified' else '#2d3748')
        ax.text(30 + c_i*9.5, y_pos, val, fontsize=6.8, color=col_c)

plt.tight_layout()
plt.savefig('figures/figure7_gui_interface.png', dpi=300)
plt.close()

# Figure 8: GUI Prediction & Simulator Results
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Outer container
ax.add_patch(patches.Rectangle((1, 1), 98, 58, facecolor='#ffffff', edgecolor='#cbd5e0', linewidth=1.5))
ax.text(4, 55, 'Interactive Demographic Simulator & "What-If" Analysis', fontsize=12, fontweight='bold', color='#1a202c')
ax.text(4, 52, 'Real-time inference using the Champion Soft-Voting Ensemble', fontsize=8.5, color='#718096')

# Left Column: User Input Sliders
ax.add_patch(patches.Rectangle((4, 4), 44, 46, facecolor='#f7fafc', edgecolor='#e2e8f0', linewidth=1))
ax.text(6, 46.5, 'Demographic Input Parameters', fontsize=9.5, fontweight='bold', color='#2d3748')
inputs = [
    ('Sex Ratio (Females / 1000 Males)', '945'),
    ('Child Population Share (Aged 0-6)', '14.2%'),
    ('Workforce Participation Rate', '39.8%'),
    ('Scheduled Caste (SC) Share', '16.5%'),
    ('Scheduled Tribe (ST) Share', '8.2%'),
    ('Child Sex Ratio (Aged 0-6)', '918'),
    ('Female Workforce Participation', '24.6%'),
    ('Dependency Burden Ratio', '1.51'),
    ('Total Population Scale (Log10)', '6.28 (1.9 Million)')
]
for idx, (param, val) in enumerate(inputs):
    yp = 42.5 - idx * 4.1
    ax.text(6, yp, param, fontsize=7.5, color='#4a5568')
    ax.text(40, yp, val, fontsize=7.5, fontweight='bold', color='#2b6cb0', ha='right')
    ax.add_patch(patches.Rectangle((6, yp - 1.2), 34, 0.4, facecolor='#cbd5e0'))
    ax.add_patch(patches.Rectangle((6, yp - 1.2), 22, 0.4, facecolor='#2b6cb0'))

# Right Column: Prediction Results & Probabilities
ax.add_patch(patches.Rectangle((50, 4), 46, 46, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1))
ax.text(52, 46.5, 'Live Prediction Output', fontsize=9.5, fontweight='bold', color='#2d3748')

# Output Verdict Box
ax.add_patch(patches.Rectangle((52, 36), 42, 8.5, facecolor='#ebf8ff', edgecolor='#bee3f8', linewidth=1))
ax.text(54, 41.5, 'PREDICTED LITERACY TIER:', fontsize=8, color='#2b6cb0', fontweight='bold')
ax.text(54, 38, 'MEDIUM LITERACY (60% - 75%)', fontsize=12, color='#2c5282', fontweight='bold')

# Probability Bars
ax.text(52, 32, 'Class Probability Distribution:', fontsize=8.5, fontweight='bold', color='#4a5568')
probs = [('Low Literacy (< 60%)', 0.18, '#718096'), ('Medium Literacy (60-75%)', 0.71, '#2b6cb0'), ('High Literacy (>= 75%)', 0.11, '#4a5568')]
for idx, (pname, pval, col) in enumerate(probs):
    yp = 27 - idx * 5.2
    ax.text(52, yp + 1.8, pname, fontsize=7.5, color='#2d3748')
    ax.text(92, yp + 1.8, f'{pval*100:.1f}%', fontsize=7.5, fontweight='bold', color='#2d3748', ha='right')
    ax.add_patch(patches.Rectangle((52, yp), 40, 1.4, facecolor='#edf2f7'))
    ax.add_patch(patches.Rectangle((52, yp), 40 * pval, 1.4, facecolor=col))

# Bottom Note
ax.add_patch(patches.Rectangle((52, 6), 42, 7, facecolor='#f7fafc', edgecolor='#e2e8f0', linewidth=0.8))
ax.text(54, 10.5, 'Model Confidence Interpretation:', fontsize=7.5, fontweight='bold', color='#4a5568')
ax.text(54, 7.5, 'High confidence prediction (71.0% probability).\nDemographic indicators align with national median profile.', fontsize=6.8, color='#718096')

plt.tight_layout()
plt.savefig('figures/figure8_gui_prediction.png', dpi=300)
plt.close()

print('SUCCESS: Figure 7 and Figure 8 generated.')
