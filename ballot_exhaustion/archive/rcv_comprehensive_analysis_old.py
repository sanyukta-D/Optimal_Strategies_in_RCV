import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os
from matplotlib.gridspec import GridSpec

def extract_strategy_dict(strategy_str):
    """
    Extract the strategy dictionary from the strategy string.
    Returns a dictionary mapping candidates to their percentages.
    """
    try:
        strategy_dict = ast.literal_eval(strategy_str)
        if not isinstance(strategy_dict, dict):
            return {}
        
        # Extract the main percentage for each candidate
        result = {}
        for candidate, data_list in strategy_dict.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                result[candidate] = data_list[0]
        return result
    except (ValueError, SyntaxError, TypeError):
        return {}

def extract_exhaust_dict(exhaust_str):
    """
    Extract the exhaust dictionary from the exhaust string.
    Returns a dictionary mapping candidates to their exhaust percentages.
    """
    try:
        exhaust_dict = ast.literal_eval(exhaust_str)
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except (ValueError, SyntaxError, TypeError):
        return {}

def calculate_differences(df, region):
    """
    Calculate the difference between exhaust and strategy percentages for each letter.
    Returns a dictionary mapping letters to lists of differences.
    """
    letter_diffs = {}
    election_count = 0
    elections_with_exhaust_gt_strategy = 0
    
    for _, row in df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find candidates that appear in both dictionaries
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        # Skip if no common candidates
        if not common_candidates:
            continue
            
        election_count += 1
        has_exhaust_gt_strategy = False
        
        for letter in common_candidates:
            # Skip letter A (the winner)
            if letter == 'A':
                continue
                
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            diff = exhaust_val - strategy_val  # Positive means exhaust > strategy
            
            if exhaust_val > strategy_val:
                has_exhaust_gt_strategy = True
                
            if letter not in letter_diffs:
                letter_diffs[letter] = []
            
            letter_diffs[letter].append({
                'diff': diff,
                'strategy': strategy_val,
                'exhaust': exhaust_val,
                'region': region,
                'election_id': row.get('file_name', f"{region}_{_}")  # Use file_name or create an ID
            })
        
        if has_exhaust_gt_strategy:
            elections_with_exhaust_gt_strategy += 1
    
    return letter_diffs, election_count, elections_with_exhaust_gt_strategy

# Probability model functions
def beta_parameters(gap_to_win_pct):
    """
    Calculate reasonable Beta parameters based on gap size.
    
    For small gaps, stay close to Beta(50,50) which is centered at 0.5.
    As gap increases, shift toward Beta(40,60), Beta(30,70), etc.
    """
    # Base parameters for a perfectly tied race (centered at 0.5)
    base_param = 50.0
    
    # Maximum shift based on gap (capped at 40)
    max_shift = min(gap_to_win_pct * 0.5, 40)
    
    # Calculate parameters
    a = max(base_param - max_shift, 10.0)  # Parameter for B (challenger), minimum 10
    b = max(base_param + max_shift, 10.0)  # Parameter for A (leader), minimum 10
    
    return a, b

def beta_probability(required_preference_pct, gap_to_win_pct):
    """
    Calculate probability using a Beta distribution with reasonable parameters.
    """
    # Convert from percentage to proportion
    required_proportion = required_preference_pct / 100
    
    # Calculate Beta parameters
    a, b = beta_parameters(gap_to_win_pct)
    
    # Probability is P(X ≥ required_proportion) where X ~ Beta(a,b)
    return 1 - stats.beta.cdf(required_proportion, a, b)

def normal_parameters(gap_to_win_pct):
    """
    Calculate reasonable Normal parameters based on gap size.
    """
    # Mean: 50% minus a factor of the gap
    mean = 50 - gap_to_win_pct * 0.5
    
    # Standard deviation: higher for smaller gaps
    std_dev = max(5, 10 - gap_to_win_pct * 0.5)
    
    return mean, std_dev

def normal_probability(required_preference_pct, gap_to_win_pct):
    """
    Calculate probability using a Normal distribution with reasonable parameters.
    """
    # Calculate Normal parameters
    mean, std_dev = normal_parameters(gap_to_win_pct)
    
    # Probability is P(X ≥ required_preference_pct) where X ~ N(mean, std_dev)
    return 1 - stats.norm.cdf(required_preference_pct, mean, std_dev)

def uniform_favor_probability(required_preference_pct, gap_to_win_pct):
    """
    More intuitive model based on gap size that favors uniform distribution.
    """
    # Estimated preference for B based on gap
    estimated_preference = 50 
    
    # Standard deviation based on gap size
    std_dev = max(5, 10 - gap_to_win_pct)
    
    # Use normal distribution around estimated preference
    return 1 - stats.norm.cdf(required_preference_pct, estimated_preference, std_dev)

def bayesian_probability(exhaust_pct, required_preference_pct, gap_to_win_pct):
    """
    Bayesian model with reasonable priors.
    """
    # Handle cases where exhaust_pct is zero or very small
    if exhaust_pct <= 0.01:
        return 0.0  # No chance of winning if there are no exhausted ballots
    
    # Start with informative prior centered near 50%
    a, b = 30, 30  # Beta(30,30) is fairly concentrated around 0.5
    
    # Update based on gap size - use a reasonable scale for evidence strength
    # Cap the evidence strength to prevent negative parameters
    evidence_strength = min((1 - exhaust_pct/100), 0.8) * 100  # Cap at 80 virtual observations
    evidence_for_a = 0.5 + gap_to_win_pct/200  # Proportion favoring A
    evidence_for_b = 1 - evidence_for_a  # Proportion favoring B
    
    # Update parameters, ensuring they remain positive
    a_posterior = max(a + evidence_strength * evidence_for_b, 1.0)
    b_posterior = max(b + evidence_strength * evidence_for_a, 1.0)
    
    # Calculate probability
    return 1 - stats.beta.cdf(required_preference_pct/100, a_posterior, b_posterior)

def combined_probability(beta_prob, normal_prob, uniform_prob, bayesian_prob):
    """
    Calculate combined probability as weighted average of other models.
    """
    weights = {
        'beta': 0.30,
        'normal': 0.20,
        'uniform': 0.25,
        'bayesian': 0.25
    }
    
    return (weights['beta'] * beta_prob + 
            weights['normal'] * normal_prob + 
            weights['uniform'] * uniform_prob + 
            weights['bayesian'] * bayesian_prob)

def analyze_elections():
    """
    Analyze the elections and calculate probability metrics.
    """
    print("Loading election data...")
    
    # Load NYC data
    nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
    nyc_df = pd.read_excel(nyc_path)
    nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
    print(f"Loaded NYC data: {len(nyc_df)} rows")
    
    # Load Alaska data
    alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
    alaska_df = pd.read_excel(alaska_path)
    print(f"Loaded Alaska data: {len(alaska_df)} rows")
    
    # Calculate differences
    print("Calculating differences...")
    nyc_diffs, nyc_election_count, nyc_exhaust_gt_strategy = calculate_differences(nyc_df, "NYC")
    alaska_diffs, alaska_election_count, alaska_exhaust_gt_strategy = calculate_differences(alaska_df, "Alaska")
    
    # Print summary statistics
    print(f"NYC Elections: {nyc_election_count}")
    print(f"NYC Elections with exhaust > strategy: {nyc_exhaust_gt_strategy} ({nyc_exhaust_gt_strategy/nyc_election_count*100:.1f}%)")
    
    print(f"Alaska Elections: {alaska_election_count}")
    print(f"Alaska Elections with exhaust > strategy: {alaska_exhaust_gt_strategy} ({alaska_exhaust_gt_strategy/alaska_election_count*100:.1f}%)")
    
    # Convert to DataFrame for easier analysis - NYC
    print("Converting data to DataFrames...")
    nyc_data = []
    for letter, diffs in nyc_diffs.items():
        for diff_dict in diffs:
            nyc_data.append({
                'letter': letter,
                'diff': diff_dict['diff'],
                'strategy': diff_dict['strategy'],
                'exhaust': diff_dict['exhaust'],
                'region': diff_dict['region'],
                'election_id': diff_dict['election_id']
            })
    
    nyc_df_analysis = pd.DataFrame(nyc_data)
    print(f"NYC analysis DataFrame: {len(nyc_df_analysis)} rows")
    
    # Convert to DataFrame for easier analysis - Alaska
    alaska_data = []
    for letter, diffs in alaska_diffs.items():
        for diff_dict in diffs:
            alaska_data.append({
                'letter': letter,
                'diff': diff_dict['diff'],
                'strategy': diff_dict['strategy'],
                'exhaust': diff_dict['exhaust'],
                'region': diff_dict['region'],
                'election_id': diff_dict['election_id']
            })
    
    alaska_df_analysis = pd.DataFrame(alaska_data)
    print(f"Alaska analysis DataFrame: {len(alaska_df_analysis)} rows")
    
    # Combine for overall analysis
    combined_df = pd.concat([nyc_df_analysis, alaska_df_analysis])
    print(f"Combined DataFrame: {len(combined_df)} rows")
    
    # Calculate probability metrics for each candidate
    print("Calculating probability metrics...")
    probability_results = []
    skipped_rows = 0
    
    for idx, row in combined_df.iterrows():
        region = row['region']
        election_id = row['election_id']
        letter = row['letter']
        gap_to_win_pct = row['strategy']
        exhaust_pct = row['exhaust']
        
        # Skip rows with zero or very small exhaust percentage (can't calculate probability)
        if exhaust_pct <= 0.01:
            skipped_rows += 1
            continue
        
        try:
            # Calculate required net advantage
            required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
            
            # Calculate required preference percentage
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            # Calculate probabilities using different models
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            uniform_prob = uniform_favor_probability(required_preference_pct, gap_to_win_pct)
            bayesian_prob = bayesian_probability(exhaust_pct, required_preference_pct, gap_to_win_pct)
            
            # Calculate combined probability
            combined_prob = combined_probability(beta_prob, normal_prob, uniform_prob, bayesian_prob)
            
            # Strategy to exhaust ratio
            strategy_to_exhaust_ratio = gap_to_win_pct / exhaust_pct
            
            probability_results.append({
                'region': region,
                'election_id': election_id,
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_net_advantage': required_net_advantage,
                'required_preference_pct': required_preference_pct,
                'strategy_to_exhaust_ratio': strategy_to_exhaust_ratio,
                'beta_probability': beta_prob,
                'normal_probability': normal_prob,
                'uniform_probability': uniform_prob,
                'bayesian_probability': bayesian_prob,
                'combined_probability': combined_prob
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(f"  Region: {region}, Letter: {letter}, Gap: {gap_to_win_pct}, Exhaust: {exhaust_pct}")
    
    print(f"Skipped {skipped_rows} rows with zero or very small exhaust percentage")
    
    probability_df = pd.DataFrame(probability_results)
    print(f"Probability DataFrame: {len(probability_df)} rows")
    
    # Save the probability data
    probability_df.to_csv('rcv_probability_analysis.csv', index=False)
    print("Saved probability data to rcv_probability_analysis.csv")
    
    return nyc_df_analysis, alaska_df_analysis, probability_df

def visualize_results(nyc_df, alaska_df, probability_df):
    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12})
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # 1. Histogram of letter frequencies
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC Letters
    nyc_letters = nyc_df['letter'].value_counts().sort_index()
    ax1.bar(nyc_letters.index, nyc_letters.values, color='steelblue')
    ax1.set_title('NYC: Frequency of Candidates (Excluding A)', fontsize=16)
    ax1.set_xlabel('Candidate Letter', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add counts on top of bars
    for i, v in enumerate(nyc_letters.values):
        ax1.text(i, v + 0.5, str(v), ha='center')
    
    # Alaska Letters
    alaska_letters = alaska_df['letter'].value_counts().sort_index()
    ax2.bar(alaska_letters.index, alaska_letters.values, color='firebrick')
    ax2.set_title('Alaska: Frequency of Candidates (Excluding A)', fontsize=16)
    ax2.set_xlabel('Candidate Letter', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add counts on top of bars
    for i, v in enumerate(alaska_letters.values):
        ax2.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/letter_frequencies.png', dpi=300)
    
    # 2. Boxplot of exhaust - strategy differences by region and letter
    # NYC boxplot
    fig2a = plt.figure(figsize=(10, 8))
    ax3 = fig2a.add_subplot(111)
    
    nyc_letters_sorted = sorted(nyc_df['letter'].unique())
    sns.boxplot(x='letter', y='diff', data=nyc_df, order=nyc_letters_sorted, 
                palette='Blues', ax=ax3)
    ax3.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax3.set_title('NYC: Exhaust - Strategy Differences by Letter', fontsize=16)
    ax3.set_xlabel('Candidate Letter', fontsize=14)
    ax3.set_ylabel('Exhaust % - Strategy %', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add count labels
    for i, letter in enumerate(nyc_letters_sorted):
        count = nyc_df[nyc_df['letter'] == letter].shape[0]
        ax3.text(i, ax3.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/nyc_exhaust_strategy_diff_boxplot.png', dpi=300)
    
    # Alaska boxplot
    fig2b = plt.figure(figsize=(10, 8))
    ax4 = fig2b.add_subplot(111)
    
    alaska_letters_sorted = sorted(alaska_df['letter'].unique())
    sns.boxplot(x='letter', y='diff', data=alaska_df, order=alaska_letters_sorted, 
                palette='Reds', ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax4.set_title('Alaska: Exhaust - Strategy Differences by Letter', fontsize=16)
    ax4.set_xlabel('Candidate Letter', fontsize=14)
    ax4.set_ylabel('Exhaust % - Strategy %', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add count labels
    for i, letter in enumerate(alaska_letters_sorted):
        count = alaska_df[alaska_df['letter'] == letter].shape[0]
        ax4.text(i, ax4.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/alaska_exhaust_strategy_diff_boxplot.png', dpi=300)
    
    # 3. Scatter plots of exhaust vs strategy by region
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(18, 8))
    
    # NYC
    scatter1 = ax5.scatter(nyc_df['strategy'], nyc_df['exhaust'], 
                      c=nyc_df['letter'].astype('category').cat.codes, 
                      alpha=0.7, s=80, cmap='tab10')
    
    # Add identity line
    min_val = min(nyc_df['strategy'].min(), nyc_df['exhaust'].min())
    max_val = max(nyc_df['strategy'].max(), nyc_df['exhaust'].max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    ax5.set_title('NYC: Exhaust % vs Strategy %', fontsize=16)
    ax5.set_xlabel('Strategy %', fontsize=14)
    ax5.set_ylabel('Exhaust %', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = scatter1.legend_elements(prop="colors")
    legend1 = ax5.legend(handles, sorted(nyc_df['letter'].unique()), 
                     loc="upper left", title="Letter")
    
    # Alaska
    scatter2 = ax6.scatter(alaska_df['strategy'], alaska_df['exhaust'], 
                      c=alaska_df['letter'].astype('category').cat.codes, 
                      alpha=0.7, s=80, cmap='tab10')
    
    # Add identity line
    min_val = min(alaska_df['strategy'].min(), alaska_df['exhaust'].min())
    max_val = max(alaska_df['strategy'].max(), alaska_df['exhaust'].max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    ax6.set_title('Alaska: Exhaust % vs Strategy %', fontsize=16)
    ax6.set_xlabel('Strategy %', fontsize=14)
    ax6.set_ylabel('Exhaust %', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = scatter2.legend_elements(prop="colors")
    legend2 = ax6.legend(handles, sorted(alaska_df['letter'].unique()), 
                     loc="upper left", title="Letter")
    
    plt.tight_layout()
    plt.savefig('figures/exhaust_vs_strategy_scatter.png', dpi=300)
    
    # 4. Visualize probability distributions by model
    fig4 = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig4)
    
    # NYC Beta distribution visualization
    ax7 = fig4.add_subplot(gs[0, 0])
    
    # Get a range of gap values
    gap_values = [0.5, 1, 2, 5, 10]
    x = np.linspace(0, 1, 1000)
    
    for gap in gap_values:
        a, b = beta_parameters(gap)
        ax7.plot(x, stats.beta.pdf(x, a, b), 
                label=f'Gap: {gap}%, Beta({a:.1f}, {b:.1f})')
    
    ax7.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    ax7.set_title('Beta Distribution Models by Gap Size', fontsize=16)
    ax7.set_xlabel('Proportion Preferring B over A', fontsize=14)
    ax7.set_ylabel('Density', fontsize=14)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Normal distribution visualization
    ax8 = fig4.add_subplot(gs[0, 1])
    x = np.linspace(0, 100, 1000)
    
    for gap in gap_values:
        mean, std_dev = normal_parameters(gap)
        ax8.plot(x, stats.norm.pdf(x, mean, std_dev),
                label=f'Gap: {gap}%, N({mean:.1f}, {std_dev:.1f})')
    
    ax8.axvline(50, color='k', linestyle='--', alpha=0.5)
    ax8.set_title('Normal Distribution Models by Gap Size', fontsize=16)
    ax8.set_xlabel('Percentage Preferring B over A', fontsize=14)
    ax8.set_ylabel('Density', fontsize=14)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Probability by model comparison
    ax9 = fig4.add_subplot(gs[1, 0])
    
    # Split by region
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    # Plot NYC
    models = ['beta_probability', 'normal_probability', 'uniform_probability', 
             'bayesian_probability', 'combined_probability']
    model_labels = ['Beta', 'Normal', 'Uniform', 'Bayesian', 'Combined']
    
    for model, label in zip(models, model_labels):
        ax9.scatter(nyc_prob['required_preference_pct'], nyc_prob[model], 
                   label=f'NYC: {label}', s=25, alpha=0.4)
    
    ax9.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax9.axvline(50, color='r', linestyle='--', alpha=0.5)
    ax9.set_title('NYC: Required Preference % vs Model Probabilities', fontsize=16)
    ax9.set_xlabel('Required Preference %', fontsize=14)
    ax9.set_ylabel('Probability', fontsize=14)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Alaska probability model comparison
    ax10 = fig4.add_subplot(gs[1, 1])
    
    for model, label in zip(models, model_labels):
        ax10.scatter(alaska_prob['required_preference_pct'], alaska_prob[model], 
                    label=f'Alaska: {label}', s=25, alpha=0.4)
    
    ax10.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax10.axvline(50, color='r', linestyle='--', alpha=0.5)
    ax10.set_title('Alaska: Required Preference % vs Model Probabilities', fontsize=16)
    ax10.set_xlabel('Required Preference %', fontsize=14)
    ax10.set_ylabel('Probability', fontsize=14)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/probability_models.png', dpi=300)
    
    # 5. Gap-to-win vs Probability and Exhaust vs Probability
    fig5, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # NYC: Gap to win vs Probability
    for model, label in zip(models[-1:], model_labels[-1:]):  # Only use combined model for clarity
        sns.regplot(x='gap_to_win_pct', y=model, data=nyc_prob, 
                   scatter_kws={'alpha':0.5, 's':50}, 
                   line_kws={'color':'red'}, ax=ax11)
    
    ax11.set_title('NYC: Gap to Win % vs Combined Probability', fontsize=16)
    ax11.set_xlabel('Gap to Win %', fontsize=14)
    ax11.set_ylabel('Combined Probability', fontsize=14)
    ax11.grid(True, alpha=0.3)
    
    # Alaska: Gap to win vs Probability
    for model, label in zip(models[-1:], model_labels[-1:]):
        sns.regplot(x='gap_to_win_pct', y=model, data=alaska_prob, 
                   scatter_kws={'alpha':0.5, 's':50}, 
                   line_kws={'color':'red'}, ax=ax12)
    
    ax12.set_title('Alaska: Gap to Win % vs Combined Probability', fontsize=16)
    ax12.set_xlabel('Gap to Win %', fontsize=14)
    ax12.set_ylabel('Combined Probability', fontsize=14)
    ax12.grid(True, alpha=0.3)
    
    # NYC: Exhaust vs Probability
    for model, label in zip(models[-1:], model_labels[-1:]):
        sns.regplot(x='exhaust_pct', y=model, data=nyc_prob, 
                   scatter_kws={'alpha':0.5, 's':50}, 
                   line_kws={'color':'red'}, ax=ax13)
    
    ax13.set_title('NYC: Exhaust % vs Combined Probability', fontsize=16)
    ax13.set_xlabel('Exhaust %', fontsize=14)
    ax13.set_ylabel('Combined Probability', fontsize=14)
    ax13.grid(True, alpha=0.3)
    
    # Alaska: Exhaust vs Probability
    for model, label in zip(models[-1:], model_labels[-1:]):
        sns.regplot(x='exhaust_pct', y=model, data=alaska_prob, 
                   scatter_kws={'alpha':0.5, 's':50}, 
                   line_kws={'color':'red'}, ax=ax14)
    
    ax14.set_title('Alaska: Exhaust % vs Combined Probability', fontsize=16)
    ax14.set_xlabel('Exhaust %', fontsize=14)
    ax14.set_ylabel('Combined Probability', fontsize=14)
    ax14.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/gap_exhaust_vs_probability.png', dpi=300)
    
    # 6. Strategy to exhaust ratio vs probability
    fig6, (ax15, ax16) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Filter out extreme ratios for better visualization
    nyc_filtered = nyc_prob[nyc_prob['strategy_to_exhaust_ratio'] <= 2.0]
    alaska_filtered = alaska_prob[alaska_prob['strategy_to_exhaust_ratio'] <= 2.0]
    
    # NYC
    sns.regplot(x='strategy_to_exhaust_ratio', y='combined_probability', 
               data=nyc_filtered, scatter_kws={'alpha':0.7, 's':60}, 
               line_kws={'color':'red'}, ax=ax15)
    
    ax15.set_title('NYC: Strategy-to-Exhaust Ratio vs Combined Probability', fontsize=16)
    ax15.set_xlabel('Strategy-to-Exhaust Ratio', fontsize=14)
    ax15.set_ylabel('Combined Probability', fontsize=14)
    ax15.grid(True, alpha=0.3)
    
    # Alaska
    sns.regplot(x='strategy_to_exhaust_ratio', y='combined_probability', 
               data=alaska_filtered, scatter_kws={'alpha':0.7, 's':60}, 
               line_kws={'color':'red'}, ax=ax16)
    
    ax16.set_title('Alaska: Strategy-to-Exhaust Ratio vs Combined Probability', fontsize=16)
    ax16.set_xlabel('Strategy-to-Exhaust Ratio', fontsize=14)
    ax16.set_ylabel('Combined Probability', fontsize=14)
    ax16.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/strategy_to_exhaust_ratio.png', dpi=300)
    
    # 7. Heatmaps of probability by gap and exhaust bins
    fig7, (ax17, ax18) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create bins
    gap_bins = [0, 1, 2, 5, 10, 100]
    exhaust_bins = [0, 5, 10, 15, 25, 100]
    
    # Create bin labels
    gap_labels = [f"{gap_bins[i]}-{gap_bins[i+1]}%" for i in range(len(gap_bins)-1)]
    exhaust_labels = [f"{exhaust_bins[i]}-{exhaust_bins[i+1]}%" for i in range(len(exhaust_bins)-1)]
    
    # Add bin categories to NYC data
    nyc_prob['gap_bin'] = pd.cut(nyc_prob['gap_to_win_pct'], bins=gap_bins, labels=gap_labels)
    nyc_prob['exhaust_bin'] = pd.cut(nyc_prob['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels)
    
    # Create pivot table for NYC
    nyc_pivot = nyc_prob.pivot_table(values='combined_probability', 
                                    index='gap_bin', 
                                    columns='exhaust_bin', 
                                    aggfunc='mean')
    
    # Create heatmap for NYC
    sns.heatmap(nyc_pivot, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax17)
    ax17.set_title('NYC: Average Combined Probability by Gap and Exhaust Bins', fontsize=16)
    ax17.set_xlabel('Exhaust Percentage Bin', fontsize=14)
    ax17.set_ylabel('Gap to Win Percentage Bin', fontsize=14)
    
    # Add bin categories to Alaska data
    alaska_prob['gap_bin'] = pd.cut(alaska_prob['gap_to_win_pct'], bins=gap_bins, labels=gap_labels)
    alaska_prob['exhaust_bin'] = pd.cut(alaska_prob['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels)
    
    # Create pivot table for Alaska
    alaska_pivot = alaska_prob.pivot_table(values='combined_probability', 
                                         index='gap_bin', 
                                         columns='exhaust_bin', 
                                         aggfunc='mean')
    
    # Create heatmap for Alaska
    sns.heatmap(alaska_pivot, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax18)
    ax18.set_title('Alaska: Average Combined Probability by Gap and Exhaust Bins', fontsize=16)
    ax18.set_xlabel('Exhaust Percentage Bin', fontsize=14)
    ax18.set_ylabel('Gap to Win Percentage Bin', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/probability_heatmap.png', dpi=300)
    
    # 8. Model comparison and correlation matrix
    fig8, (ax19, ax20) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Distribution of probabilities by model - NYC
    for model, label in zip(models, model_labels):
        sns.kdeplot(nyc_prob[model], ax=ax19, label=label, fill=True, alpha=0.3)
    
    ax19.set_xlabel('Probability', fontsize=14)
    ax19.set_ylabel('Density', fontsize=14)
    ax19.set_title('NYC: Distribution of Probabilities by Model', fontsize=16)
    ax19.legend()
    ax19.set_xlim(0, 1)
    ax19.grid(True, alpha=0.3)
    
    # Model correlation matrix
    corr_models = ['beta_probability', 'normal_probability', 'uniform_probability', 
                  'bayesian_probability', 'combined_probability']
    
    # Combine NYC and Alaska for overall correlation
    corr_df = probability_df[corr_models].corr()
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', ax=ax20)
    ax20.set_title('Correlation Between Different Probability Models', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300)
    
    print("\nAnalysis complete. Visualizations saved to 'figures' directory:")
    print("1. letter_frequencies.png")
    print("2. nyc_exhaust_strategy_diff_boxplot.png")
    print("3. alaska_exhaust_strategy_diff_boxplot.png")
    print("4. exhaust_vs_strategy_scatter.png")
    print("5. probability_models.png")
    print("6. gap_exhaust_vs_probability.png")
    print("7. strategy_to_exhaust_ratio.png")
    print("8. probability_heatmap.png")
    print("9. model_comparison.png")

if __name__ == "__main__":
    # Run analysis
    nyc_df, alaska_df, probability_df = analyze_elections()
    
    # Create visualizations
    visualize_results(nyc_df, alaska_df, probability_df) 