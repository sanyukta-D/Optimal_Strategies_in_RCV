import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os

def extract_strategy_dict(strategy_str):
    """Extract strategy dictionary from string"""
    try:
        strategy_dict = ast.literal_eval(strategy_str)
        if not isinstance(strategy_dict, dict):
            return {}
        
        # Extract main percentage for each candidate
        result = {}
        for candidate, data_list in strategy_dict.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                result[candidate] = data_list[0]
        return result
    except:
        return {}

def extract_exhaust_dict(exhaust_str):
    """Extract exhaust dictionary from string"""
    try:
        exhaust_dict = ast.literal_eval(exhaust_str)
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except:
        return {}

def beta_parameters(gap_to_win_pct):
    """Calculate Beta parameters based on gap size"""
    base_param = 50.0
    max_shift = min(gap_to_win_pct * 0.5, 40)
    a = max(base_param - max_shift, 10.0)
    b = max(base_param + max_shift, 10.0)
    return a, b

def beta_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using Beta distribution"""
    required_proportion = required_preference_pct / 100
    a, b = beta_parameters(gap_to_win_pct)
    return 1 - stats.beta.cdf(required_proportion, a, b)

def normal_parameters(gap_to_win_pct):
    """Calculate Normal parameters based on gap size"""
    mean = 50 - gap_to_win_pct * 0.5
    std_dev = max(5, 10 - gap_to_win_pct * 0.5)
    return mean, std_dev

def normal_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using Normal distribution"""
    mean, std_dev = normal_parameters(gap_to_win_pct)
    return 1 - stats.norm.cdf(required_preference_pct, mean, std_dev)

def uniform_favor_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using uniform favor model"""
    estimated_preference = 50
    std_dev = max(5, 10 - gap_to_win_pct)
    return 1 - stats.norm.cdf(required_preference_pct, estimated_preference, std_dev)

def process_election_data():
    """Process NYC and Alaska election data"""
    print("Loading and processing election data...")
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    try:
        # Load NYC data
        nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
        nyc_df = pd.read_excel(nyc_path)
        nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
        print(f"Loaded NYC data: {len(nyc_df)} rows")
    except Exception as e:
        print(f"Error loading NYC data: {e}")
        nyc_df = pd.DataFrame()
    
    try:
        # Load Alaska data
        alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
        alaska_df = pd.read_excel(alaska_path)
        print(f"Loaded Alaska data: {len(alaska_df)} rows")
    except Exception as e:
        print(f"Error loading Alaska data: {e}")
        alaska_df = pd.DataFrame()
    
    # Process NYC data
    nyc_data = []
    for idx, row in nyc_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    diff = exhaust_val - strategy_val
                    
                    nyc_data.append({
                        'letter': letter,
                        'diff': diff,
                        'strategy': strategy_val,
                        'exhaust': exhaust_val,
                        'region': 'NYC',
                        'election_id': row.get('file_name', f"NYC_{idx}")
                    })
        except Exception as e:
            print(f"Error processing NYC row {idx}: {e}")
    
    # Process Alaska data
    alaska_data = []
    for idx, row in alaska_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    diff = exhaust_val - strategy_val
                    
                    alaska_data.append({
                        'letter': letter,
                        'diff': diff,
                        'strategy': strategy_val,
                        'exhaust': exhaust_val,
                        'region': 'Alaska',
                        'election_id': row.get('file_name', f"Alaska_{idx}")
                    })
        except Exception as e:
            print(f"Error processing Alaska row {idx}: {e}")
    
    # Create DataFrames
    nyc_df_analysis = pd.DataFrame(nyc_data)
    alaska_df_analysis = pd.DataFrame(alaska_data)
    
    print(f"NYC analysis DataFrame: {len(nyc_df_analysis)} rows")
    print(f"Alaska analysis DataFrame: {len(alaska_df_analysis)} rows")
    
    # Save preprocessed data
    nyc_df_analysis.to_csv('nyc_analysis.csv', index=False)
    alaska_df_analysis.to_csv('alaska_analysis.csv', index=False)
    
    return nyc_df_analysis, alaska_df_analysis

def create_letter_frequency_plot(nyc_df, alaska_df):
    """Create letter frequency histograms"""
    print("Creating letter frequency plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC Letters
    if not nyc_df.empty:
        nyc_letters = nyc_df['letter'].value_counts().sort_index()
        ax1.bar(nyc_letters.index, nyc_letters.values, color='steelblue')
        ax1.set_title('NYC: Frequency of Candidates (Excluding A)', fontsize=16)
        ax1.set_xlabel('Candidate Letter', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add counts on top of bars
        for i, (letter, v) in enumerate(nyc_letters.items()):
            ax1.text(i, v + 0.5, str(v), ha='center')
    
    # Alaska Letters
    if not alaska_df.empty:
        alaska_letters = alaska_df['letter'].value_counts().sort_index()
        ax2.bar(alaska_letters.index, alaska_letters.values, color='firebrick')
        ax2.set_title('Alaska: Frequency of Candidates (Excluding A)', fontsize=16)
        ax2.set_xlabel('Candidate Letter', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add counts on top of bars
        for i, (letter, v) in enumerate(alaska_letters.items()):
            ax2.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/letter_frequencies.png', dpi=300)
    plt.close()

def create_diff_boxplots(nyc_df, alaska_df):
    """Create boxplots of exhaust - strategy differences"""
    print("Creating exhaust-strategy difference boxplots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # NYC
    if not nyc_df.empty:
        nyc_letters_sorted = sorted(nyc_df['letter'].unique())
        sns.boxplot(x='letter', y='diff', data=nyc_df, order=nyc_letters_sorted, 
                    palette='Blues', ax=ax1)
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        ax1.set_title('NYC: Exhaust - Strategy Differences by Letter', fontsize=16)
        ax1.set_xlabel('Candidate Letter', fontsize=14)
        ax1.set_ylabel('Exhaust % - Strategy %', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add count labels
        for i, letter in enumerate(nyc_letters_sorted):
            count = nyc_df[nyc_df['letter'] == letter].shape[0]
            ax1.text(i, ax1.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    # Alaska
    if not alaska_df.empty:
        alaska_letters_sorted = sorted(alaska_df['letter'].unique())
        sns.boxplot(x='letter', y='diff', data=alaska_df, order=alaska_letters_sorted, 
                    palette='Reds', ax=ax2)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        ax2.set_title('Alaska: Exhaust - Strategy Differences by Letter', fontsize=16)
        ax2.set_xlabel('Candidate Letter', fontsize=14)
        ax2.set_ylabel('Exhaust % - Strategy %', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add count labels
        for i, letter in enumerate(alaska_letters_sorted):
            count = alaska_df[alaska_df['letter'] == letter].shape[0]
            ax2.text(i, ax2.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/exhaust_strategy_diff_boxplot.png', dpi=300)
    plt.close()

def create_scatter_plots(nyc_df, alaska_df):
    """Create scatter plots of exhaust vs strategy"""
    print("Creating scatter plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # NYC
    if not nyc_df.empty:
        scatter1 = ax1.scatter(nyc_df['strategy'], nyc_df['exhaust'], 
                          c=nyc_df['letter'].astype('category').cat.codes, 
                          alpha=0.7, s=80, cmap='tab10')
        
        # Add identity line
        min_val = min(nyc_df['strategy'].min(), nyc_df['exhaust'].min())
        max_val = max(nyc_df['strategy'].max(), nyc_df['exhaust'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax1.set_title('NYC: Exhaust % vs Strategy %', fontsize=16)
        ax1.set_xlabel('Strategy %', fontsize=14)
        ax1.set_ylabel('Exhaust %', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = scatter1.legend_elements(prop="colors")
        legend1 = ax1.legend(handles, sorted(nyc_df['letter'].unique()), 
                         loc="upper left", title="Letter")
    
    # Alaska
    if not alaska_df.empty:
        scatter2 = ax2.scatter(alaska_df['strategy'], alaska_df['exhaust'], 
                          c=alaska_df['letter'].astype('category').cat.codes, 
                          alpha=0.7, s=80, cmap='tab10')
        
        # Add identity line
        min_val = min(alaska_df['strategy'].min(), alaska_df['exhaust'].min())
        max_val = max(alaska_df['strategy'].max(), alaska_df['exhaust'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        ax2.set_title('Alaska: Exhaust % vs Strategy %', fontsize=16)
        ax2.set_xlabel('Strategy %', fontsize=14)
        ax2.set_ylabel('Exhaust %', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = scatter2.legend_elements(prop="colors")
        legend2 = ax2.legend(handles, sorted(alaska_df['letter'].unique()), 
                         loc="upper left", title="Letter")
    
    plt.tight_layout()
    plt.savefig('figures/exhaust_vs_strategy_scatter.png', dpi=300)
    plt.close()

def create_distribution_plots():
    """Create probability distribution visualizations"""
    print("Creating probability distribution plots...")
    
    # Visualization of Beta and Normal distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Beta distribution visualization
    gap_values = [0.5, 1, 2, 5, 10]
    x_beta = np.linspace(0, 1, 1000)
    
    for gap in gap_values:
        a, b = beta_parameters(gap)
        ax1.plot(x_beta, stats.beta.pdf(x_beta, a, b), 
                label=f'Gap: {gap}%, Beta({a:.1f}, {b:.1f})')
    
    ax1.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    ax1.set_title('Beta Distribution Models by Gap Size', fontsize=16)
    ax1.set_xlabel('Proportion Preferring B over A', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Normal distribution visualization
    x_norm = np.linspace(0, 100, 1000)
    
    for gap in gap_values:
        mean, std_dev = normal_parameters(gap)
        ax2.plot(x_norm, stats.norm.pdf(x_norm, mean, std_dev),
                label=f'Gap: {gap}%, N({mean:.1f}, {std_dev:.1f})')
    
    ax2.axvline(50, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Normal Distribution Models by Gap Size', fontsize=16)
    ax2.set_xlabel('Percentage Preferring B over A', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/probability_distributions.png', dpi=300)
    plt.close()

def calculate_probabilities(nyc_df, alaska_df):
    """Calculate probabilities for each candidate"""
    print("Calculating probabilities...")
    
    probability_results = []
    
    # Process NYC data
    if not nyc_df.empty:
        for idx, row in nyc_df.iterrows():
            letter = row['letter']
            gap_to_win_pct = row['strategy']
            exhaust_pct = row['exhaust']
            
            # Skip if exhaust is too small
            if exhaust_pct <= 0.01:
                continue
                
            # Calculate required net advantage
            required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
            
            # Calculate required preference percentage
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            # Calculate model probabilities
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            uniform_prob = uniform_favor_probability(required_preference_pct, gap_to_win_pct)
            
            # Combined probability (simple average)
            combined_prob = (beta_prob + normal_prob + uniform_prob) / 3
            
            probability_results.append({
                'region': 'NYC',
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_preference_pct': required_preference_pct,
                'beta_probability': beta_prob,
                'normal_probability': normal_prob,
                'uniform_probability': uniform_prob,
                'combined_probability': combined_prob
            })
    
    # Process Alaska data
    if not alaska_df.empty:
        for idx, row in alaska_df.iterrows():
            letter = row['letter']
            gap_to_win_pct = row['strategy']
            exhaust_pct = row['exhaust']
            
            # Skip if exhaust is too small
            if exhaust_pct <= 0.01:
                continue
                
            # Calculate required net advantage
            required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
            
            # Calculate required preference percentage
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            # Calculate model probabilities
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            uniform_prob = uniform_favor_probability(required_preference_pct, gap_to_win_pct)
            
            # Combined probability (simple average)
            combined_prob = (beta_prob + normal_prob + uniform_prob) / 3
            
            probability_results.append({
                'region': 'Alaska',
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_preference_pct': required_preference_pct,
                'beta_probability': beta_prob,
                'normal_probability': normal_prob,
                'uniform_probability': uniform_prob,
                'combined_probability': combined_prob
            })
    
    probability_df = pd.DataFrame(probability_results)
    probability_df.to_csv('probability_results.csv', index=False)
    
    return probability_df

def create_probability_plots(probability_df):
    """Create plots visualizing probabilities"""
    print("Creating probability plots...")
    
    if probability_df.empty:
        print("No probability data available for plotting")
        return
    
    # Split by region
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    # 1. Probability model comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # NYC
    if not nyc_prob.empty:
        models = ['beta_probability', 'normal_probability', 'uniform_probability', 'combined_probability']
        model_labels = ['Beta', 'Normal', 'Uniform', 'Combined']
        
        for model, label in zip(models, model_labels):
            ax1.scatter(nyc_prob['required_preference_pct'], nyc_prob[model], 
                      label=label, alpha=0.5, s=30)
        
        ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(50, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('NYC: Required Preference % vs Probability', fontsize=16)
        ax1.set_xlabel('Required Preference %', fontsize=14)
        ax1.set_ylabel('Probability', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Alaska
    if not alaska_prob.empty:
        for model, label in zip(models, model_labels):
            ax2.scatter(alaska_prob['required_preference_pct'], alaska_prob[model], 
                      label=label, alpha=0.5, s=30)
        
        ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(50, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Alaska: Required Preference % vs Probability', fontsize=16)
        ax2.set_xlabel('Required Preference %', fontsize=14)
        ax2.set_ylabel('Probability', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300)
    plt.close()
    
    # 2. Gap vs Probability and Exhaust vs Probability
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    
    # NYC: Gap vs Probability
    if not nyc_prob.empty:
        sns.regplot(x='gap_to_win_pct', y='combined_probability', data=nyc_prob, 
                  scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red'}, ax=ax1)
        
        ax1.set_title('NYC: Gap to Win % vs Combined Probability', fontsize=16)
        ax1.set_xlabel('Gap to Win %', fontsize=14)
        ax1.set_ylabel('Combined Probability', fontsize=14)
        ax1.grid(True, alpha=0.3)
    
    # Alaska: Gap vs Probability
    if not alaska_prob.empty:
        sns.regplot(x='gap_to_win_pct', y='combined_probability', data=alaska_prob, 
                  scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red'}, ax=ax2)
        
        ax2.set_title('Alaska: Gap to Win % vs Combined Probability', fontsize=16)
        ax2.set_xlabel('Gap to Win %', fontsize=14)
        ax2.set_ylabel('Combined Probability', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    # NYC: Exhaust vs Probability
    if not nyc_prob.empty:
        sns.regplot(x='exhaust_pct', y='combined_probability', data=nyc_prob, 
                  scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red'}, ax=ax3)
        
        ax3.set_title('NYC: Exhaust % vs Combined Probability', fontsize=16)
        ax3.set_xlabel('Exhaust %', fontsize=14)
        ax3.set_ylabel('Combined Probability', fontsize=14)
        ax3.grid(True, alpha=0.3)
    
    # Alaska: Exhaust vs Probability
    if not alaska_prob.empty:
        sns.regplot(x='exhaust_pct', y='combined_probability', data=alaska_prob, 
                  scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red'}, ax=ax4)
        
        ax4.set_title('Alaska: Exhaust % vs Combined Probability', fontsize=16)
        ax4.set_xlabel('Exhaust %', fontsize=14)
        ax4.set_ylabel('Combined Probability', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/gap_exhaust_vs_probability.png', dpi=300)
    plt.close()

def process_district1_data():
    """Process NYC Council District 1 Primary (2021) data directly"""
    print("Processing NYC Council District 1 Primary (2021) data directly...")
    
    # Load NYC data to get actual exhaust percentages
    nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
    nyc_df = pd.read_excel(nyc_path)
    
    # Find the District 1 data
    district1_row = nyc_df[nyc_df['file_name'].str.contains("DEMCouncilMember1stCouncilDistrict", na=False)].iloc[0]
    
    # Extract exhaust percentages from the original data
    exhaust_dict = extract_exhaust_dict(district1_row['exhaust_percents'])
    
    # Data from the provided image
    district1_data = {
        'Candidate': [
            'Christopher Marte', 
            'Jenny L. Low', 
            'Gigi Li', 
            'Susan Lee', 
            'Maud Maron', 
            'T. Johnson-Winbush', 
            'Sean C. Hayes', 
            'Susan Damplo', 
            'Denny R. Salas'
        ],
        'ID': ['A', 'B', 'C', 'E', 'D', 'G', 'F', 'H', 'I'],
        'Victory': [0.00, 17.07, 20.13, 28.95, 37.49, 47.37, 47.45, 50.15, 53.47]
    }
    
    district1_df = pd.DataFrame(district1_data)
    
    # Calculate gap percentages (difference from winner A)
    district1_df['gap_to_win_pct'] = district1_df['Victory']
    
    # Use actual exhaust percentages from the Excel file
    district1_df['exhaust_pct'] = district1_df['ID'].map(lambda x: exhaust_dict.get(x, 0))
    
    # Create analysis dataframe in the same format as the nyc_df
    analysis_data = []
    for _, row in district1_df.iterrows():
        if row['ID'] == 'A':  # Skip the winner
            continue
            
        analysis_data.append({
            'letter': row['ID'],
            'diff': row['exhaust_pct'] - row['gap_to_win_pct'],
            'strategy': row['gap_to_win_pct'],
            'exhaust': row['exhaust_pct'],
            'region': 'NYC',
            'election_id': 'NYC_District1_2021'
        })
    
    district1_analysis_df = pd.DataFrame(analysis_data)
    print(f"District 1 analysis DataFrame: {len(district1_analysis_df)} rows")
    
    # Display the exhaust percentages
    print("\nExhaust percentages for NYC Council District 1 Primary (2021):")
    for _, row in district1_df.iterrows():
        if row['ID'] != 'A':  # Skip the winner
            print(f"Candidate {row['ID']} ({row['Candidate']}): {row['exhaust_pct']:.2f}%")
    
    return district1_analysis_df

def process_district2_data():
    """Process NYC Council District 2 Primary (2021) data directly"""
    print("Processing NYC Council District 2 Primary (2021) data directly...")
    
    # Load NYC data to get actual exhaust percentages
    nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
    nyc_df = pd.read_excel(nyc_path)
    
    # Find the District 2 data
    district2_row = nyc_df[nyc_df['file_name'].str.contains("DEMCouncilMember2ndCouncilDistrict", na=False)].iloc[0]
    
    # Extract exhaust percentages from the original data
    exhaust_dict = extract_exhaust_dict(district2_row['exhaust_percents'])
    
    # Data for District 2 (assuming similar format to District 1)
    district2_data = {
        'Candidate': [
            'Carlina Rivera', 
            'Allie Ryan', 
            'Juan Pagan', 
            'Erin Hussein', 
            'Momka Gurung'
        ],
        'ID': ['A', 'B', 'C', 'D', 'E'],
        'Victory': [0.00, 31.97, 40.26, 42.70, 47.11]
    }
    
    district2_df = pd.DataFrame(district2_data)
    
    # Calculate gap percentages (difference from winner A)
    district2_df['gap_to_win_pct'] = district2_df['Victory']
    
    # Use actual exhaust percentages from the Excel file
    district2_df['exhaust_pct'] = district2_df['ID'].map(lambda x: exhaust_dict.get(x, 0))
    
    # Create analysis dataframe in the same format as the nyc_df
    analysis_data = []
    for _, row in district2_df.iterrows():
        if row['ID'] == 'A':  # Skip the winner
            continue
            
        analysis_data.append({
            'letter': row['ID'],
            'diff': row['exhaust_pct'] - row['gap_to_win_pct'],
            'strategy': row['gap_to_win_pct'],
            'exhaust': row['exhaust_pct'],
            'region': 'NYC',
            'election_id': 'NYC_District2_2021'
        })
    
    district2_analysis_df = pd.DataFrame(analysis_data)
    print(f"District 2 analysis DataFrame: {len(district2_analysis_df)} rows")
    
    # Display the exhaust percentages
    print("\nExhaust percentages for NYC Council District 2 Primary (2021):")
    for _, row in district2_df.iterrows():
        if row['ID'] != 'A':  # Skip the winner
            print(f"Candidate {row['ID']} ({row['Candidate']}): {row['exhaust_pct']:.2f}%")
    
    return district2_analysis_df

def main():
    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 12})
    
    # Process election data
    nyc_df, alaska_df = process_election_data()
    
    # Process District 1 data directly
    district1_df = process_district1_data()
    
    # Process District 2 data directly
    district2_df = process_district2_data()
    
    # Combine District 1 and 2 data with other NYC data
    nyc_df = pd.concat([nyc_df, district1_df, district2_df])
    
    # Create basic visualizations
    create_letter_frequency_plot(nyc_df, alaska_df)
    create_diff_boxplots(nyc_df, alaska_df)
    create_scatter_plots(nyc_df, alaska_df)
    create_distribution_plots()
    
    # Calculate and visualize probabilities
    probability_df = calculate_probabilities(nyc_df, alaska_df)
    create_probability_plots(probability_df)
    
    print("\nAnalysis complete. Visualizations saved to 'figures' directory.")

if __name__ == "__main__":
    main() 