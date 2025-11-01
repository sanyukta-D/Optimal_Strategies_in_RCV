import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Load the gap-based probabilities CSV
df = pd.read_csv('gap_based_probabilities.csv')

# Calculate strategy-to-exhaust ratio
df['strategy_to_exhaust_ratio'] = df['gap_to_win_pct'] / df['exhaust_pct']

# Setup plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with GridSpec for flexible layout
fig = plt.figure(figsize=(15, 14))
gs = GridSpec(3, 2, figure=fig)

# 1. Distribution of probabilities by model and region - Top row, left
ax1 = fig.add_subplot(gs[0, 0])
models = ['beta_probability', 'normal_probability', 'competitive_probability', 
         'rcv_probability', 'combined_probability']
model_names = ['Beta', 'Normal', 'Competitive', 'RCV', 'Combined']

for i, (model, name) in enumerate(zip(models, model_names)):
    sns.kdeplot(df[model], ax=ax1, label=name, fill=True, alpha=0.3)

ax1.set_xlabel('Probability')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Probabilities by Model')
ax1.legend()
ax1.set_xlim(0, 0.5)  # Focus on 0-50% range where most values lie

# 2. Probability by letter/candidate - Top row, right
ax2 = fig.add_subplot(gs[0, 1])
letters = sorted(df['letter'].unique())
box_data = [df[df['letter'] == letter]['combined_probability'] for letter in letters]
ax2.boxplot(box_data, labels=letters)
ax2.set_xlabel('Candidate Letter')
ax2.set_ylabel('Combined Probability')
ax2.set_title('Distribution of Combined Probability by Candidate')
ax2.axhline(0.5, color='r', linestyle='--', alpha=0.7)
for i, letter in enumerate(letters):
    count = len(df[df['letter'] == letter])
    ax2.text(i+1, -0.05, f'n={count}', ha='center')

# 3. Gap to win % vs. probability (scatter) - Middle row, left
ax3 = fig.add_subplot(gs[1, 0])
regions = df['region'].unique()
colors = {'NYC': 'blue', 'Alaska': 'red'}

for region in regions:
    region_df = df[df['region'] == region]
    ax3.scatter(region_df['gap_to_win_pct'], region_df['combined_probability'], 
                alpha=0.7, label=region, c=colors[region])

ax3.set_xlabel('Gap to Win %')
ax3.set_ylabel('Combined Probability')
ax3.set_title('Gap to Win % vs Combined Probability')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Exhaust % vs. probability (scatter) - Middle row, right
ax4 = fig.add_subplot(gs[1, 1])
for region in regions:
    region_df = df[df['region'] == region]
    ax4.scatter(region_df['exhaust_pct'], region_df['combined_probability'], 
                alpha=0.7, label=region, c=colors[region])

ax4.set_xlabel('Exhaust %')
ax4.set_ylabel('Combined Probability')
ax4.set_title('Exhaust % vs Combined Probability')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Strategy-to-exhaust ratio vs. probability - Bottom row, left
ax5 = fig.add_subplot(gs[2, 0])

# Remove extreme outliers for better visualization
df_filtered = df[df['strategy_to_exhaust_ratio'] < 1.0]  # Filter out extreme ratios

for region in regions:
    region_df = df_filtered[df_filtered['region'] == region]
    ax5.scatter(region_df['strategy_to_exhaust_ratio'], region_df['combined_probability'], 
                alpha=0.7, label=region, c=colors[region])

# Add trend line
sns.regplot(x='strategy_to_exhaust_ratio', y='combined_probability', 
           data=df_filtered, scatter=False, ax=ax5, color='black', line_kws={"linestyle": "--"})

ax5.set_xlabel('Strategy-to-Exhaust Ratio')
ax5.set_ylabel('Combined Probability')
ax5.set_title('Strategy-to-Exhaust Ratio vs Combined Probability')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Heatmap of probability by gap and exhaust percentiles - Bottom row, right
ax6 = fig.add_subplot(gs[2, 1])

# Create percentile bins for gap and exhaust
gap_bins = [0, 1, 2, 4, 10, 100]
exhaust_bins = [0, 8, 15, 20, 30, 100]

# Create bin labels
gap_labels = [f"{gap_bins[i]}-{gap_bins[i+1]}%" for i in range(len(gap_bins)-1)]
exhaust_labels = [f"{exhaust_bins[i]}-{exhaust_bins[i+1]}%" for i in range(len(exhaust_bins)-1)]

# Add bin categories
df['gap_bin'] = pd.cut(df['gap_to_win_pct'], bins=gap_bins, labels=gap_labels)
df['exhaust_bin'] = pd.cut(df['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels)

# Create pivot table
pivot = df.pivot_table(values='combined_probability', 
                       index='gap_bin', 
                       columns='exhaust_bin', 
                       aggfunc='mean')

# Create heatmap
sns.heatmap(pivot, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax6)
ax6.set_title('Average Combined Probability by Gap and Exhaust Bins')
ax6.set_xlabel('Exhaust Percentage Bin')
ax6.set_ylabel('Gap to Win Percentage Bin')

# Create another figure for required preference analysis
fig2 = plt.figure(figsize=(15, 6))

# 7. Required preference percentage vs probability - First plot
ax7 = fig2.add_subplot(121)
for region in regions:
    region_df = df[df['region'] == region]
    ax7.scatter(region_df['required_preference_pct'], region_df['combined_probability'], 
                alpha=0.7, label=region, c=colors[region])

ax7.set_xlabel('Required Preference % (among exhausted ballots)')
ax7.set_ylabel('Combined Probability')
ax7.set_title('Required Preference % vs Probability')
ax7.axvline(50, color='r', linestyle='--', alpha=0.7)
ax7.axhline(0.5, color='r', linestyle='--', alpha=0.7)
ax7.grid(True, alpha=0.3)
ax7.legend()

# 8. Required net advantage vs probability - Second plot
ax8 = fig2.add_subplot(122)
for region in regions:
    region_df = df[df['region'] == region]
    ax8.scatter(region_df['required_net_advantage'], region_df['combined_probability'], 
                alpha=0.7, label=region, c=colors[region])

ax8.set_xlabel('Required Net Advantage % (among exhausted ballots)')
ax8.set_ylabel('Combined Probability')
ax8.set_title('Required Net Advantage % vs Probability')
ax8.axvline(0, color='r', linestyle='--', alpha=0.7)
ax8.grid(True, alpha=0.3)
ax8.legend()

# Adjust layout
fig.tight_layout()
fig2.tight_layout()

# Save the plots
fig.savefig('probability_distributions_analysis.png', dpi=300, bbox_inches='tight')
fig2.savefig('required_preference_analysis.png', dpi=300, bbox_inches='tight')

# Create a third figure for model comparison
fig3 = plt.figure(figsize=(15, 10))
gs3 = GridSpec(2, 2, figure=fig3)

# 9. Model correlation matrix - Upper left
ax9 = fig3.add_subplot(gs3[0, 0])
corr_models = ['beta_probability', 'normal_probability', 'competitive_probability', 
              'rcv_probability', 'combined_probability']
corr_df = df[corr_models].corr()
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', ax=ax9)
ax9.set_title('Correlation Between Different Probability Models')

# 10. Distribution of probabilities by region - Upper right
ax10 = fig3.add_subplot(gs3[0, 1])
for region in regions:
    region_df = df[df['region'] == region]
    sns.kdeplot(region_df['combined_probability'], ax=ax10, 
                label=f"{region} (n={len(region_df)})", fill=True, alpha=0.4)
ax10.set_xlabel('Combined Probability')
ax10.set_ylabel('Density')
ax10.set_title('Distribution of Combined Probability by Region')
ax10.legend()
ax10.set_xlim(0, 0.5)

# 11. Scatter plot matrix of key metrics - Bottom row spanning both columns
ax11 = fig3.add_subplot(gs3[1, :])
plot_cols = ['gap_to_win_pct', 'exhaust_pct', 'strategy_to_exhaust_ratio', 'required_preference_pct', 'combined_probability']
pd.plotting.scatter_matrix(df[plot_cols], alpha=0.5, figsize=(10, 10), diagonal='kde', ax=ax11)
ax11.set_title('Relationships Between Key Metrics', pad=20)

# Adjust layout and save
fig3.tight_layout()
fig3.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Visualizations saved to:")
print("1. probability_distributions_analysis.png")
print("2. required_preference_analysis.png")
print("3. model_comparison_analysis.png") 