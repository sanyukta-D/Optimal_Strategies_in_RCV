import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs('updated_figures', exist_ok=True)

# Load the saved results data
try:
    results_df = pd.read_csv('model_comparison_results/all_elections_analysis.csv')
except FileNotFoundError:
    try:
        results_df = pd.read_csv('probability_results_focused.csv')
    except FileNotFoundError:
        print("Could not find results data. Please run the analysis script first.")
        exit()

# Filter for NYC elections only
nyc_df = results_df[results_df['region'] == 'NYC']

if nyc_df.empty:
    print("No NYC election data available for visualizations.")
    exit()

# Updated model names and colors
model_names = {
    'beta_model_prob': 'Gap-Based Beta',
    'posterior_beta_prob': 'Similarity Beta',
    'prior_posterior_beta_prob': 'Prior-Posterior Beta',
    'unconditional_bootstrap_prob': 'Unconditional Bootstrap',
    'category_bootstrap_prob': 'Similarity Bootstrap',
    'limited_bootstrap_prob': 'Rank-Restricted Bootstrap'
}

# Updated model colors
model_colors = {
    'beta_model_prob': 'blue',
    'posterior_beta_prob': 'green',
    'prior_posterior_beta_prob': 'magenta',
    'unconditional_bootstrap_prob': 'orange',
    'category_bootstrap_prob': 'red',
    'limited_bootstrap_prob': 'purple'
}

# Updated model markers
model_markers = {
    'beta_model_prob': 'o',
    'posterior_beta_prob': '*',
    'prior_posterior_beta_prob': 's',
    'unconditional_bootstrap_prob': 'X',
    'category_bootstrap_prob': '^',
    'limited_bootstrap_prob': 'D'
}

# Sort by ratio for better visualization
nyc_sorted = nyc_df.sort_values('strategy_exhaust_ratio')

# Create figure
plt.figure(figsize=(14, 9))

# Plot each model
for model, name in model_names.items():
    if model in nyc_sorted.columns:
        plt.plot(nyc_sorted['strategy_exhaust_ratio'], nyc_sorted[model], 
               marker=model_markers[model], linestyle='-', color=model_colors[model], label=name, 
               markersize=10, linewidth=2, alpha=0.8)
    
# Add horizontal line at 0.5 probability    
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

# Updated title and labels
plt.title('Probability Of Alternate Winners Vs Strategy-Exhaust Ratio', fontsize=16)
plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig('updated_figures/nyc_models_vs_ratio.png', dpi=300)
print("Created updated nyc_models_vs_ratio.png with corrected model names and title")
plt.close() 