import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'updated_figures')
os.makedirs(output_dir, exist_ok=True)

# Load the saved results data
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    results_df = pd.read_csv(os.path.join(script_dir, 'model_comparison_results/all_elections_analysis.csv'))
except FileNotFoundError:
    try:
        results_df = pd.read_csv(os.path.join(script_dir, 'probability_results_focused.csv'))
    except FileNotFoundError:
        print("Could not find results data. Please run the analysis script first.")
        exit()

# Filter for Alaska elections only
alaska_df = results_df[results_df['region'] == 'Alaska']

if alaska_df.empty:
    print("No Alaska election data available for visualizations.")
    exit()

# Updated model names and colors (excluding Rank-Restricted Bootstrap for Alaska)
model_names = {
    'beta_model_prob': 'Gap-Based Beta',
    'posterior_beta_prob': 'Similarity Beta',
    'prior_posterior_beta_prob': 'Prior-Posterior Beta',
    'unconditional_bootstrap_prob': 'Unconditional Bootstrap',
    'category_bootstrap_prob': 'Similarity Bootstrap'
}

# Updated model colors
model_colors = {
    'beta_model_prob': 'blue',
    'posterior_beta_prob': 'green',
    'prior_posterior_beta_prob': 'magenta',
    'unconditional_bootstrap_prob': 'orange',
    'category_bootstrap_prob': 'red'
}

# Updated model markers
model_markers = {
    'beta_model_prob': 'o',
    'posterior_beta_prob': '*',
    'prior_posterior_beta_prob': 's',
    'unconditional_bootstrap_prob': 'X',
    'category_bootstrap_prob': '^'
}

# Sort by ratio for better visualization
alaska_sorted = alaska_df.sort_values('strategy_exhaust_ratio')

# Create figure
plt.figure(figsize=(14, 9))

# Plot each model
for model, name in model_names.items():
    if model in alaska_sorted.columns:
        plt.plot(alaska_sorted['strategy_exhaust_ratio'], alaska_sorted[model], 
               marker=model_markers[model], linestyle='-', color=model_colors[model], label=name, 
               markersize=10, linewidth=2, alpha=0.8)
    
# Add horizontal line at 0.5 probability    
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

# Updated title and labels with increased font sizes
plt.title('Probability of Alternate Winners Vs Victory Gap/Exhaust Ratio', fontsize=24)
plt.xlabel('Victory Gap/Exhaust Ratio (Gap to Win % / Exhaust %)', fontsize=22)
plt.ylabel('Probability', fontsize=22)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=20)

# Increase tick label font sizes
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Save the figure as PDF for better quality
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'alaska_models_vs_ratio.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'alaska_models_vs_ratio.png'), dpi=300)
print("Created updated alaska_models_vs_ratio.png and .pdf with corrected model names and title")
plt.close()

