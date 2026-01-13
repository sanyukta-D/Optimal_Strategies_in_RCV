import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs('figures_focused', exist_ok=True)

# Load the saved probability data
probability_df = pd.read_csv('probability_results_focused.csv')

# Split data by region
nyc_prob = probability_df[probability_df['region'] == 'NYC']
alaska_prob = probability_df[probability_df['region'] == 'Alaska']

# Set up Bayesian models, labels, and colors
bayesian_models = ['bayesian_beta_probability', 'bayesian_normal_probability', 
                 'direct_posterior_probability', 'direct_posterior_beta_probability', 
                 'direct_posterior_normal_probability']
model_labels = ['Bayesian Beta', 'Bayesian Normal', 'Direct Posterior (Binary)',
               'Direct Posterior Beta', 'Direct Posterior Normal']
colors = ['purple', 'orange', 'cyan', 'blue', 'green']
line_styles = ['-', '--', ':', '-', '-.']
markers = ['o', 's', '^', 'D', 'x']

# NYC Bayesian models
if not nyc_prob.empty:
    plt.figure(figsize=(14, 9))
    
    # Sort by required preference percentage for smoother lines
    nyc_sorted = nyc_prob.sort_values('required_preference_pct')
    
    for i, (model, label, color, ls, marker) in enumerate(zip(bayesian_models, model_labels, colors, line_styles, markers)):
        plt.plot(nyc_sorted['required_preference_pct'], nyc_sorted[model], 
               marker=marker, linestyle=ls, color=color, label=label, 
               markersize=10, linewidth=3, alpha=0.8)
        
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    plt.axvline(50, color='k', linestyle='--', alpha=0.5)
    plt.title('NYC: Bayesian Probability Models vs Required Preference %', fontsize=16)
    plt.xlabel('Required % of Exhausted Ballots Preferring Candidate B over A', fontsize=14)
    plt.ylabel('Probability of Candidate B Winning', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.ylim(-0.05, 1.05)  # Set y-axis limits to show full probability range
    
    plt.tight_layout()
    plt.savefig('figures_focused/nyc_bayesian_models_only.png', dpi=300)
    print("Generated nyc_bayesian_models_only.png")
    plt.close()

print("Done creating plots") 