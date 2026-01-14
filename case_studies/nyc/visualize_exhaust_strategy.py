import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os

def extract_strategy_dict(strategy_str):
    """Extract the strategy dictionary from the strategy string."""
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
    """Extract the exhaust dictionary from the exhaust string."""
    try:
        exhaust_dict = ast.literal_eval(exhaust_str)
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except (ValueError, SyntaxError, TypeError):
        return {}

def analyze_and_visualize():
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Load NYC data
    nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
    nyc_df = pd.read_excel(nyc_path)
    nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
    
    # Load Alaska data
    alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
    alaska_df = pd.read_excel(alaska_path)
    
    # Process data for visualization
    nyc_data = process_data(nyc_df, "NYC")
    alaska_data = process_data(alaska_df, "Alaska")
    
    # Combine data
    all_data = pd.concat([nyc_data, alaska_data])
    
    # Save processed data
    all_data.to_csv("exhaust_vs_strategy_analysis.csv", index=False)
    
    # Create visualizations
    create_scatter_plots(all_data)
    create_boxplots(all_data)
    create_bar_charts(all_data)
    create_ratio_charts(all_data)

def process_data(df, region):
    """Process dataframe to extract strategy and exhaust data."""
    data_list = []
    
    for idx, row in df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find candidates that appear in both dictionaries
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        for letter in common_candidates:
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            
            # Calculate metrics
            strategy_to_exhaust_ratio = (strategy_val / exhaust_val) * 100 if exhaust_val > 0 else 0
            voters_needed_pct = (exhaust_val - strategy_val) / exhaust_val * 100 if exhaust_val > 0 and exhaust_val > strategy_val else 0
            
            data_list.append({
                'region': region,
                'election_id': row.get('file_name', f"{region}_{idx}"),
                'letter': letter,
                'strategy': strategy_val,
                'exhaust': exhaust_val,
                'diff': exhaust_val - strategy_val,
                'exhaust_greater': exhaust_val > strategy_val,
                'strategy_to_exhaust_ratio': strategy_to_exhaust_ratio,
                'voters_needed_pct': voters_needed_pct
            })
    
    return pd.DataFrame(data_list)

def create_scatter_plots(df):
    """Create scatter plots comparing exhaust vs strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC scatter plot
    nyc_df = df[df['region'] == 'NYC']
    for letter in sorted(nyc_df['letter'].unique()):
        letter_df = nyc_df[nyc_df['letter'] == letter]
        axes[0].scatter(letter_df['exhaust'], letter_df['strategy'], 
                       label=f'Letter {letter}', alpha=0.7)
    
    # Add diagonal line where exhaust = strategy
    max_val = max(nyc_df['exhaust'].max(), nyc_df['strategy'].max()) * 1.1
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    axes[0].set_xlabel('Exhaust Percentage')
    axes[0].set_ylabel('Strategy Percentage')
    axes[0].set_title('NYC: Exhaust vs Strategy Percentage')
    axes[0].legend()
    
    # Alaska scatter plot
    alaska_df = df[df['region'] == 'Alaska']
    for letter in sorted(alaska_df['letter'].unique()):
        letter_df = alaska_df[alaska_df['letter'] == letter]
        axes[1].scatter(letter_df['exhaust'], letter_df['strategy'], 
                       label=f'Letter {letter}', alpha=0.7)
    
    # Add diagonal line where exhaust = strategy
    max_val = max(alaska_df['exhaust'].max(), alaska_df['strategy'].max()) * 1.1
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    axes[1].set_xlabel('Exhaust Percentage')
    axes[1].set_ylabel('Strategy Percentage')
    axes[1].set_title('Alaska: Exhaust vs Strategy Percentage')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("exhaust_vs_strategy_scatter.png", dpi=300)
    plt.close()

def create_boxplots(df):
    """Create boxplots showing distribution of differences."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC boxplot
    nyc_df = df[df['region'] == 'NYC']
    letter_diffs = []
    letter_labels = []
    
    for letter in sorted(nyc_df['letter'].unique()):
        letter_df = nyc_df[nyc_df['letter'] == letter]
        letter_diffs.append(letter_df['diff'].values)
        letter_labels.append(f'Letter {letter}')
    
    axes[0].boxplot(letter_diffs, labels=letter_labels)
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    axes[0].set_ylabel('Exhaust - Strategy (percentage points)')
    axes[0].set_title('NYC: Distribution of Exhaust - Strategy by Letter')
    
    # Alaska boxplot
    alaska_df = df[df['region'] == 'Alaska']
    letter_diffs = []
    letter_labels = []
    
    for letter in sorted(alaska_df['letter'].unique()):
        letter_df = alaska_df[alaska_df['letter'] == letter]
        letter_diffs.append(letter_df['diff'].values)
        letter_labels.append(f'Letter {letter}')
    
    axes[1].boxplot(letter_diffs, labels=letter_labels)
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    axes[1].set_ylabel('Exhaust - Strategy (percentage points)')
    axes[1].set_title('Alaska: Distribution of Exhaust - Strategy by Letter')
    
    plt.tight_layout()
    plt.savefig("exhaust_vs_strategy_boxplot.png", dpi=300)
    plt.close()

def create_bar_charts(df):
    """Create bar charts showing percentage of elections with exhaust > strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC bar chart
    nyc_df = df[df['region'] == 'NYC']
    nyc_counts = []
    nyc_percentages = []
    letter_labels = []
    
    for letter in sorted(nyc_df['letter'].unique()):
        letter_df = nyc_df[nyc_df['letter'] == letter]
        count = sum(letter_df['exhaust_greater'])
        total = len(letter_df)
        percentage = (count / total) * 100 if total > 0 else 0
        
        nyc_counts.append(count)
        nyc_percentages.append(percentage)
        letter_labels.append(f'Letter {letter}')
    
    axes[0].bar(letter_labels, nyc_percentages)
    axes[0].set_ylabel('Percentage of Elections')
    axes[0].set_title('NYC: Elections with Exhaust > Strategy by Letter')
    
    # Add count labels on bars
    for i, count in enumerate(nyc_counts):
        total = len(nyc_df[nyc_df['letter'] == sorted(nyc_df['letter'].unique())[i]])
        axes[0].text(i, nyc_percentages[i] + 2, f'{count}/{total}', 
                    ha='center', va='bottom')
    
    # Alaska bar chart
    alaska_df = df[df['region'] == 'Alaska']
    alaska_counts = []
    alaska_percentages = []
    letter_labels = []
    
    for letter in sorted(alaska_df['letter'].unique()):
        letter_df = alaska_df[alaska_df['letter'] == letter]
        count = sum(letter_df['exhaust_greater'])
        total = len(letter_df)
        percentage = (count / total) * 100 if total > 0 else 0
        
        alaska_counts.append(count)
        alaska_percentages.append(percentage)
        letter_labels.append(f'Letter {letter}')
    
    axes[1].bar(letter_labels, alaska_percentages)
    axes[1].set_ylabel('Percentage of Elections')
    axes[1].set_title('Alaska: Elections with Exhaust > Strategy by Letter')
    
    # Add count labels on bars
    for i, count in enumerate(alaska_counts):
        total = len(alaska_df[alaska_df['letter'] == sorted(alaska_df['letter'].unique())[i]])
        axes[1].text(i, alaska_percentages[i] + 2, f'{count}/{total}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("exhaust_vs_strategy_bar.png", dpi=300)
    plt.close()

def create_ratio_charts(df):
    """Create charts showing strategy to exhaust ratio."""
    # Filter for cases where exhaust > strategy
    df_filtered = df[df['exhaust_greater']]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # NYC ratio chart
    nyc_df = df_filtered[df_filtered['region'] == 'NYC']
    letter_ratios = []
    letter_labels = []
    
    for letter in sorted(nyc_df['letter'].unique()):
        letter_df = nyc_df[nyc_df['letter'] == letter]
        letter_ratios.append(letter_df['strategy_to_exhaust_ratio'].values)
        letter_labels.append(f'Letter {letter}')
    
    if letter_ratios:  # Only plot if there's data
        axes[0].boxplot(letter_ratios, labels=letter_labels)
        axes[0].set_ylabel('Strategy as % of Exhaust')
        axes[0].set_title('NYC: Strategy to Exhaust Ratio by Letter')
    
    # Alaska ratio chart
    alaska_df = df_filtered[df_filtered['region'] == 'Alaska']
    letter_ratios = []
    letter_labels = []
    
    for letter in sorted(alaska_df['letter'].unique()):
        letter_df = alaska_df[alaska_df['letter'] == letter]
        letter_ratios.append(letter_df['strategy_to_exhaust_ratio'].values)
        letter_labels.append(f'Letter {letter}')
    
    if letter_ratios:  # Only plot if there's data
        axes[1].boxplot(letter_ratios, labels=letter_labels)
        axes[1].set_ylabel('Strategy as % of Exhaust')
        axes[1].set_title('Alaska: Strategy to Exhaust Ratio by Letter')
    
    plt.tight_layout()
    plt.savefig("strategy_to_exhaust_ratio.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    analyze_and_visualize() 