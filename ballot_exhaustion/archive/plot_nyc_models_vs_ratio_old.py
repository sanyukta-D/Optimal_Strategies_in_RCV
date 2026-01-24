import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs('model_comparison_results/figures', exist_ok=True)

# Load the saved results data
results_df = pd.read_csv('model_comparison_results/all_elections_analysis.csv')

# Filter for NYC elections only
nyc_df = results_df[results_df['region'] == 'NYC']

if nyc_df.empty:
    print("No NYC election data available for visualizations.")
    exit()

# Focus on the three beta models
beta_models = ['beta_model_prob', 'prior_posterior_beta_prob', 'posterior_beta_prob']
model_labels = ['Theoretical Beta Distribution', 'Prior-Posterior Beta', 'Empirical Posterior Beta']
colors = ['blue', 'cyan', 'green']
markers = ['o', 's', '*']

# For the bootstrap models
bootstrap_models = ['category_bootstrap_prob', 'limited_bootstrap_prob', 'unconditional_bootstrap_prob']
bootstrap_labels = ['Category-Based Bootstrap', 'Ranking-Limited Bootstrap', 'Unconditioned Bootstrap']
bootstrap_colors = ['red', 'purple', 'orange']
bootstrap_markers = ['^', 's', 'D']

# Sort by ratio for better visualization
nyc_sorted = nyc_df.sort_values('strategy_exhaust_ratio')

# Create figure
plt.figure(figsize=(14, 9))

# Plot each beta model
for i, (model, label, color, marker) in enumerate(zip(beta_models, model_labels, colors, markers)):
    plt.plot(nyc_sorted['strategy_exhaust_ratio'], nyc_sorted[model], 
           marker=marker, linestyle='-', color=color, label=label, 
           markersize=10, linewidth=2, alpha=0.8)

# Plot each bootstrap model
for i, (model, label, color, marker) in enumerate(zip(bootstrap_models, bootstrap_labels, bootstrap_colors, bootstrap_markers)):
    plt.plot(nyc_sorted['strategy_exhaust_ratio'], nyc_sorted[model], 
           marker=marker, linestyle='-', color=color, label=label, 
           markersize=10, linewidth=2, alpha=0.8)
    
# Add horizontal line at 0.5 probability    
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

# Title and labels
plt.title('NYC Elections: Models vs Strategy-to-Exhaust Ratio', fontsize=16)
plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig('nyc_bootstrap_vs_ratio.png', dpi=300)
print("Created nyc_bootstrap_vs_ratio.png with updated Prior-Posterior Beta model")
plt.close() 