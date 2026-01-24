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

# Filter for Alaska elections only
alaska_df = results_df[results_df['region'] == 'Alaska']

if alaska_df.empty:
    print("No Alaska election data available for visualizations.")
    exit()

# Updated model names and colors
model_names = {
    'beta_model_prob': 'Gap-Based Beta',
    'prior_posterior_beta_prob': 'Prior-Posterior Beta',
    'posterior_beta_prob': 'Similarity Beta',
    'category_bootstrap_prob': 'Similarity Bootstrap',
    'unconditional_bootstrap_prob': 'Unconditional Bootstrap'
}

model_colors = {
    'beta_model_prob': 'blue',
    'prior_posterior_beta_prob': 'magenta',
    'posterior_beta_prob': 'green',
    'category_bootstrap_prob': 'red',
    'unconditional_bootstrap_prob': 'orange'
}

model_markers = {
    'beta_model_prob': 'o',
    'prior_posterior_beta_prob': 's',
    'posterior_beta_prob': '*',
    'category_bootstrap_prob': '^',
    'unconditional_bootstrap_prob': 'X'
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

# Updated title and labels
plt.title('Probability Of Alternate Winners Vs Strategy-Exhaust Ratio', fontsize=16)
plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig('updated_figures/alaska_models_vs_ratio.png', dpi=300)
print("Created updated alaska_models_vs_ratio.png with corrected model names and title")
plt.close() 