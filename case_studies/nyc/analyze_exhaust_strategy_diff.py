# Paper figures produced by this script:
#   - Fig 5 left:  nyc_exhaust_strategy_diff_boxplot.pdf   (Fig \ref{fig:exhaustion_nyc})
#   - Fig 6 left:  alaska_exhaust_strategy_diff_boxplot.pdf (Fig \ref{fig:exhaustion_alaska})
#
# Input data (paper's source of truth for single-winner exhaustion):
#   - results/tables/summary_table_nyc_final.xlsx   (exhaust_percents column)
#   - results/tables/summary_table_alska_lite.xlsx  (exhaust_percents column)
#
# Raw ballot data: case_studies/nyc/data/ and case_studies/alaska/data/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

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

def analyze_and_visualize():
    # Load NYC data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nyc_path = os.path.join(base_dir, "results/tables/summary_table_nyc_final.xlsx")
    nyc_df = pd.read_excel(nyc_path)
    nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
    
    # Load Alaska data
    alaska_path = os.path.join(base_dir, "results/tables/summary_table_alska_lite.xlsx")
    alaska_df = pd.read_excel(alaska_path)
    
    # Calculate differences
    nyc_diffs, nyc_election_count, nyc_elections_with_exhaust_gt_strategy = calculate_differences(nyc_df, "NYC")
    alaska_diffs, alaska_election_count, alaska_elections_with_exhaust_gt_strategy = calculate_differences(alaska_df, "Alaska")
    
    # Print election counts
    print(f"\nNYC Elections: {nyc_election_count}")
    print(f"NYC Elections with exhaust > strategy: {nyc_elections_with_exhaust_gt_strategy} ({nyc_elections_with_exhaust_gt_strategy/nyc_election_count*100:.1f}%)")
    
    print(f"\nAlaska Elections: {alaska_election_count}")
    print(f"Alaska Elections with exhaust > strategy: {alaska_elections_with_exhaust_gt_strategy} ({alaska_elections_with_exhaust_gt_strategy/alaska_election_count*100:.1f}%)")
    
    # Convert to DataFrame for easier analysis - NYC
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
    
    # Combine for overall analysis
    diff_df = pd.concat([nyc_df_analysis, alaska_df_analysis])
    
    # Calculate summary statistics - NYC
    nyc_summary = nyc_df_analysis.groupby('letter')['diff'].agg(['mean', 'median', 'std', 'count']).reset_index()
    nyc_summary = nyc_summary.sort_values('letter')
    
    # Calculate summary statistics - Alaska
    alaska_summary = alaska_df_analysis.groupby('letter')['diff'].agg(['mean', 'median', 'std', 'count']).reset_index()
    alaska_summary = alaska_summary.sort_values('letter')
    
    # Calculate summary statistics - Overall
    summary = diff_df.groupby('letter')['diff'].agg(['mean', 'median', 'std', 'count']).reset_index()
    summary = summary.sort_values('letter')
    
    # Calculate exhaust percentage statistics
    nyc_exhaust_summary = nyc_df_analysis.groupby('letter')['exhaust'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
    nyc_exhaust_summary = nyc_exhaust_summary.sort_values('letter')
    
    alaska_exhaust_summary = alaska_df_analysis.groupby('letter')['exhaust'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
    alaska_exhaust_summary = alaska_exhaust_summary.sort_values('letter')
    
    # Print summary statistics
    print("\nSummary statistics for exhaust - strategy differences by letter (NYC):")
    print(nyc_summary)
    
    print("\nSummary statistics for exhaust - strategy differences by letter (Alaska):")
    print(alaska_summary)
    
    print("\nSummary statistics for exhaust percentages by letter (NYC):")
    print(nyc_exhaust_summary)
    
    print("\nSummary statistics for exhaust percentages by letter (Alaska):")
    print(alaska_exhaust_summary)
    
    # Count instances where exhaust > strategy by letter
    nyc_exhaust_gt_strategy = nyc_df_analysis.groupby('letter').apply(lambda x: (x['exhaust'] > x['strategy']).sum()).reset_index()
    nyc_exhaust_gt_strategy.columns = ['letter', 'count_exhaust_gt_strategy']
    nyc_exhaust_gt_strategy['total'] = nyc_df_analysis.groupby('letter').size().values
    nyc_exhaust_gt_strategy['percentage'] = nyc_exhaust_gt_strategy['count_exhaust_gt_strategy'] / nyc_exhaust_gt_strategy['total'] * 100
    nyc_exhaust_gt_strategy = nyc_exhaust_gt_strategy.sort_values('letter')
    
    alaska_exhaust_gt_strategy = alaska_df_analysis.groupby('letter').apply(lambda x: (x['exhaust'] > x['strategy']).sum()).reset_index()
    alaska_exhaust_gt_strategy.columns = ['letter', 'count_exhaust_gt_strategy']
    alaska_exhaust_gt_strategy['total'] = alaska_df_analysis.groupby('letter').size().values
    alaska_exhaust_gt_strategy['percentage'] = alaska_exhaust_gt_strategy['count_exhaust_gt_strategy'] / alaska_exhaust_gt_strategy['total'] * 100
    alaska_exhaust_gt_strategy = alaska_exhaust_gt_strategy.sort_values('letter')
    
    print("\nCount of instances where exhaust > strategy by letter (NYC):")
    print(nyc_exhaust_gt_strategy)
    
    print("\nCount of instances where exhaust > strategy by letter (Alaska):")
    print(alaska_exhaust_gt_strategy)
    
    # Save to CSV
    nyc_df_analysis.to_csv("nyc_exhaust_strategy_differences.csv", index=False)
    alaska_df_analysis.to_csv("alaska_exhaust_strategy_differences.csv", index=False)
    nyc_summary.to_csv("nyc_exhaust_strategy_summary_by_letter.csv", index=False)
    alaska_summary.to_csv("alaska_exhaust_strategy_summary_by_letter.csv", index=False)
    nyc_exhaust_summary.to_csv("nyc_exhaust_summary_by_letter.csv", index=False)
    alaska_exhaust_summary.to_csv("alaska_exhaust_summary_by_letter.csv", index=False)
    nyc_exhaust_gt_strategy.to_csv("nyc_exhaust_gt_strategy_by_letter.csv", index=False)
    alaska_exhaust_gt_strategy.to_csv("alaska_exhaust_gt_strategy_by_letter.csv", index=False)
    
    # Visualizations - always sorted alphabetically
    sns.set_style("whitegrid")
    
    # 1. Boxplot of differences by letter - NYC (excluding A)
    plt.figure(figsize=(14, 8))
    nyc_df_no_a = nyc_df_analysis[nyc_df_analysis['letter'] != 'A'].copy()
    letters_sorted = sorted([l for l in nyc_df_no_a['letter'].unique() if l != 'A'])
    ax = sns.boxplot(x='letter', y='diff', data=nyc_df_no_a, order=letters_sorted, palette='Set3')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('NYC: Exhaust - Victory Gap Differences by Letter', fontsize=24)
    plt.xlabel('Candidate Letter', fontsize=22)
    plt.ylabel('Exhaust % - Victory Gap %', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Increase tick label font sizes
    ax.tick_params(labelsize=20)
    
    # Add count labels
    for i, letter in enumerate(letters_sorted):
        count = nyc_df_no_a[nyc_df_no_a['letter'] == letter].shape[0]
        ax.text(i, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('nyc_exhaust_strategy_diff_boxplot.pdf', dpi=300)
    plt.savefig('nyc_exhaust_strategy_diff_boxplot.png', dpi=300)
    
    # 2. Boxplot of differences by letter - Alaska (excluding A)
    plt.figure(figsize=(14, 8))
    alaska_df_no_a = alaska_df_analysis[alaska_df_analysis['letter'] != 'A'].copy()
    letters_sorted = sorted([l for l in alaska_df_no_a['letter'].unique() if l != 'A'])
    ax = sns.boxplot(x='letter', y='diff', data=alaska_df_no_a, order=letters_sorted, palette='Set3')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Alaska: Exhaust - Victory Gap Differences by Letter', fontsize=24)
    plt.xlabel('Candidate Letter', fontsize=22)
    plt.ylabel('Exhaust % - Victory Gap %', fontsize=22)
    ax.tick_params(labelsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, letter in enumerate(letters_sorted):
        count = alaska_df_no_a[alaska_df_no_a['letter'] == letter].shape[0]
        ax.text(i, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('alaska_exhaust_strategy_diff_boxplot.pdf', dpi=300)
    plt.savefig('alaska_exhaust_strategy_diff_boxplot.png', dpi=300)
    
    # 3. Boxplot of exhaust percentages by letter - NYC
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='letter', y='exhaust', data=nyc_df_analysis, order=letters_sorted, palette='Set3')
    plt.title('NYC: Distribution of Exhaust Percentages by Letter', fontsize=16)
    plt.xlabel('Letter (Candidate)', fontsize=14)
    plt.ylabel('Exhaust %', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, letter in enumerate(letters_sorted):
        count = nyc_df_analysis[nyc_df_analysis['letter'] == letter].shape[0]
        ax.text(i, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('nyc_exhaust_boxplot.png', dpi=300)
    
    # 4. Boxplot of exhaust percentages by letter - Alaska
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='letter', y='exhaust', data=alaska_df_analysis, order=letters_sorted, palette='Set3')
    plt.title('Alaska: Distribution of Exhaust Percentages by Letter', fontsize=16)
    plt.xlabel('Letter (Candidate)', fontsize=14)
    plt.ylabel('Exhaust %', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, letter in enumerate(letters_sorted):
        count = alaska_df_analysis[alaska_df_analysis['letter'] == letter].shape[0]
        ax.text(i, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('alaska_exhaust_boxplot.png', dpi=300)
    
    # 5. Scatter plot of exhaust vs strategy - NYC
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(x='strategy', y='exhaust', hue='letter', data=nyc_df_analysis, s=80, alpha=0.7)
    
    # Add identity line (where exhaust = strategy)
    min_val = min(nyc_df_analysis['strategy'].min(), nyc_df_analysis['exhaust'].min())
    max_val = max(nyc_df_analysis['strategy'].max(), nyc_df_analysis['exhaust'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.title('NYC: Exhaust % vs Strategy % by Letter', fontsize=16)
    plt.xlabel('Strategy %', fontsize=14)
    plt.ylabel('Exhaust %', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotations for points far from the identity line
    threshold = 30  # Points where the difference is more than 30%
    for _, row in nyc_df_analysis[abs(nyc_df_analysis['diff']) > threshold].iterrows():
        plt.annotate(f"{row['letter']}", 
                    (row['strategy'], row['exhaust']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig('nyc_exhaust_vs_strategy_scatter.png', dpi=300)
    
    # 6. Scatter plot of exhaust vs strategy - Alaska
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(x='strategy', y='exhaust', hue='letter', data=alaska_df_analysis, s=80, alpha=0.7)
    
    # Add identity line (where exhaust = strategy)
    min_val = min(alaska_df_analysis['strategy'].min(), alaska_df_analysis['exhaust'].min())
    max_val = max(alaska_df_analysis['strategy'].max(), alaska_df_analysis['exhaust'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.title('Alaska: Exhaust % vs Strategy % by Letter', fontsize=16)
    plt.xlabel('Strategy %', fontsize=14)
    plt.ylabel('Exhaust %', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotations for points far from the identity line
    threshold = 30  # Points where the difference is more than 30%
    for _, row in alaska_df_analysis[abs(alaska_df_analysis['diff']) > threshold].iterrows():
        plt.annotate(f"{row['letter']}", 
                    (row['strategy'], row['exhaust']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig('alaska_exhaust_vs_strategy_scatter.png', dpi=300)
    
    # 7. Bar chart of percentage of elections with exhaust > strategy by letter - NYC
    plt.figure(figsize=(12, 8))
    sns.barplot(x='letter', y='percentage', data=nyc_exhaust_gt_strategy, order=sorted(nyc_exhaust_gt_strategy['letter']))
    plt.title('NYC: Percentage of Elections with Exhaust > Strategy by Letter', fontsize=16)
    plt.xlabel('Letter (Candidate)', fontsize=14)
    plt.ylabel('Percentage of Elections', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, row in enumerate(nyc_exhaust_gt_strategy.itertuples()):
        plt.text(i, 5, f'{row.count_exhaust_gt_strategy}/{row.total}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('nyc_exhaust_gt_strategy_percentage.png', dpi=300)
    
    # 8. Bar chart of percentage of elections with exhaust > strategy by letter - Alaska
    plt.figure(figsize=(12, 8))
    sns.barplot(x='letter', y='percentage', data=alaska_exhaust_gt_strategy, order=sorted(alaska_exhaust_gt_strategy['letter']))
    plt.title('Alaska: Percentage of Elections with Exhaust > Strategy by Letter', fontsize=16)
    plt.xlabel('Letter (Candidate)', fontsize=14)
    plt.ylabel('Percentage of Elections', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, row in enumerate(alaska_exhaust_gt_strategy.itertuples()):
        plt.text(i, 5, f'{row.count_exhaust_gt_strategy}/{row.total}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('alaska_exhaust_gt_strategy_percentage.png', dpi=300)
    
    print("\nVisualizations saved as:")
    print("1. nyc_exhaust_strategy_diff_boxplot.png")
    print("2. alaska_exhaust_strategy_diff_boxplot.png")
    print("3. nyc_exhaust_boxplot.png")
    print("4. alaska_exhaust_boxplot.png")
    print("5. nyc_exhaust_vs_strategy_scatter.png")
    print("6. alaska_exhaust_vs_strategy_scatter.png")
    print("7. nyc_exhaust_gt_strategy_percentage.png")
    print("8. alaska_exhaust_gt_strategy_percentage.png")
    
    print("\nDetailed data saved to CSV files")

if __name__ == "__main__":
    analyze_and_visualize() 