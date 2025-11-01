import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Read the data
df = pd.read_csv('ballot_exhastion/model_comparison_results/all_elections_analysis.csv')

# Define the probability model columns to average
prob_columns = ['beta_model_prob', 'posterior_beta_prob', 'prior_posterior_beta_prob', 
                'category_bootstrap_prob', 'limited_bootstrap_prob', 'unconditional_bootstrap_prob']

# Replace empty strings and NaN values with 0 for probability columns
for col in prob_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Calculate average probability across all models for each row
df['avg_probability'] = df[prob_columns].mean(axis=1)

# Create bins for gap_to_win_pct and exhaust_pct
def create_bins(values, bin_edges):
    """Create bins and return bin labels"""
    bin_labels = []
    for i in range(len(bin_edges)-1):
        if i == len(bin_edges)-2:  # Last bin
            bin_labels.append(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}%")
        else:
            bin_labels.append(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}%")
    return bin_labels

# Define bin edges
gap_bins = [0, 5, 10, 15, 25, 100]  # Based on the heatmap shown
exhaust_bins = [0, 10, 15, 25, 100]  # Based on the heatmap shown

# Create bin labels
gap_labels = create_bins(gap_bins, gap_bins)
exhaust_labels = create_bins(exhaust_bins, exhaust_bins)

# Assign bins
df['gap_bin'] = pd.cut(df['gap_to_win_pct'], bins=gap_bins, labels=gap_labels, include_lowest=True)
df['exhaust_bin'] = pd.cut(df['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels, include_lowest=True)

# Create pivot tables for each region
regions = ['NYC', 'Alaska']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, region in enumerate(regions):
    region_data = df[df['region'] == region]
    
    # Create pivot table with average probabilities
    pivot_table = region_data.groupby(['gap_bin', 'exhaust_bin'])['avg_probability'].agg(['mean', 'count']).reset_index()
    pivot_matrix = pivot_table.pivot(index='gap_bin', columns='exhaust_bin', values='mean')
    count_matrix = pivot_table.pivot(index='gap_bin', columns='exhaust_bin', values='count')
    
    # Fill NaN values with 0
    pivot_matrix = pivot_matrix.fillna(0)
    count_matrix = count_matrix.fillna(0)
    
    # Create the heatmap
    ax = axes[idx]
    
    # Create custom colormap (yellow to red like in original)
    colors = ['#FFFFCC', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot heatmap
    sns.heatmap(pivot_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                vmin=0, 
                vmax=0.5,
                ax=ax,
                cbar_kws={'label': 'Probability of Candidate B Winning'})
    
    # Add count annotations
    for i in range(len(pivot_matrix.index)):
        for j in range(len(pivot_matrix.columns)):
            count = count_matrix.iloc[i, j] if not pd.isna(count_matrix.iloc[i, j]) else 0
            if count > 0:
                ax.text(j + 0.5, i + 0.7, f'(n={int(count)})', 
                       ha='center', va='center', fontsize=8, color='black')
    
    ax.set_title(f'{region}: Average Combined Probability by Gap and Exhaust Bins')
    ax.set_xlabel('Exhausted Ballot Percentage Bin')
    ax.set_ylabel('Gap Needed for Candidate B to Win (Percentage Bin)')
    
    # Invert y-axis to match original format
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('corrected_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Summary of Data:")
print(f"Total number of elections: {len(df)}")
print(f"NYC elections: {len(df[df['region'] == 'NYC'])}")
print(f"Alaska elections: {len(df[df['region'] == 'Alaska'])}")
print("\nAverage probabilities by region:")
for region in regions:
    region_data = df[df['region'] == region]
    avg_prob = region_data['avg_probability'].mean()
    print(f"{region}: {avg_prob:.4f}")

# Show the probability model contributions
print("\nProbability model contributions (averages):")
for col in prob_columns:
    avg_val = df[col].mean()
    print(f"{col}: {avg_val:.6f}") 