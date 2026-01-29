"""
NYC 2025 Visualization Scripts

1. Violin plot with 3 bands comparing NYC 2021 vs NYC 2025 victory margins
2. Boxplot of exhaust - strategy differences by letter for NYC 2025

Filters out REP elections.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
from matplotlib.patches import Patch

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_margin_of_victory(strategies_str):
    """
    Extract the margin of victory (smallest non-zero gap) from strategies.
    """
    try:
        if isinstance(strategies_str, str):
            strategies = ast.literal_eval(strategies_str)
        else:
            strategies = strategies_str
        
        if not isinstance(strategies, dict):
            return None
        
        # Get all gaps, excluding the winner (gap=0)
        gaps = []
        for candidate, data in strategies.items():
            if isinstance(data, list) and len(data) > 0:
                gap = data[0]
                if gap > 0:
                    gaps.append(gap)
        
        # Margin of victory = smallest gap (runner-up's gap)
        return min(gaps) if gaps else None
    except:
        return None


def extract_strategy_dict(strategy_str):
    """Extract the strategy dictionary from the strategy string."""
    try:
        if isinstance(strategy_str, str):
            strategy_dict = ast.literal_eval(strategy_str)
        else:
            strategy_dict = strategy_str
        if not isinstance(strategy_dict, dict):
            return {}
        
        result = {}
        for candidate, data_list in strategy_dict.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                result[candidate] = data_list[0]
        return result
    except:
        return {}


def extract_exhaust_dict(exhaust_str):
    """Extract the exhaust dictionary from the exhaust string."""
    try:
        if isinstance(exhaust_str, str):
            exhaust_dict = ast.literal_eval(exhaust_str)
        else:
            exhaust_dict = exhaust_str
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except:
        return {}


def create_violin_plot_3bands():
    """
    Create violin plot comparing NYC 2021 vs NYC 2025 victory margins
    with 3 competitiveness bands.
    """
    print("Creating violin plot with 3 bands...")
    
    # Load data
    df_2021 = pd.read_excel(RESULTS_DIR / "summary_table_nyc_final.xlsx")
    df_2025 = pd.read_excel(RESULTS_DIR / "summary_table_nyc_2025_with_margins.xlsx")
    
    # Filter out REP elections
    df_2021 = df_2021[df_2021['file_name'].str.contains("DEM", na=False)].copy()
    df_2025 = df_2025[df_2025['file_name'].str.contains("DEM", na=False)].copy()
    
    print(f"  2021 DEM elections: {len(df_2021)}")
    print(f"  2025 DEM elections: {len(df_2025)}")
    
    # Extract margins of victory - 2021 from strategies, 2025 from margin column
    margins_2021 = df_2021['Strategies'].apply(extract_margin_of_victory).dropna()
    margins_2025 = df_2025['margin'].dropna()
    
    # Filter out 100% margins (uncontested)
    margins_2021 = margins_2021[margins_2021 < 100]
    margins_2025 = margins_2025[margins_2025 < 100]
    
    print(f"  2021 margins extracted: {len(margins_2021)}")
    print(f"  2025 margins extracted: {len(margins_2025)}")
    
    # 3 bands color scheme
    winner_color = (189/255, 223/255, 167/255)  # Winner (0%)
    categories_3bands = [
        ("Competitive",  0,  10, (223/255, 240/255, 216/255)),   # Light green
        ("Moderate",    10,  30, (253/255, 245/255, 206/255)),   # Light yellow  
        ("Distant",     30, 100, (248/255, 218/255, 205/255)),   # Light salmon
    ]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Background bands
    ax.axhline(0, color=winner_color, linewidth=6, alpha=0.5, zorder=0)
    for label, low, high, color in categories_3bands:
        ax.axhspan(low, high, color=color, alpha=0.5, zorder=0)
    
    # Violin + boxplot
    data_to_plot = [margins_2021.values, margins_2025.values]
    
    vp = ax.violinplot(
        data_to_plot,
        positions=[1, 2],
        widths=0.6,
        showmeans=False,
        showmedians=False,
    )
    
    # Style violin bodies
    for body in vp["bodies"]:
        body.set_facecolor("#7BAFD4")
        body.set_alpha(0.7)
    
    ax.boxplot(
        data_to_plot,
        positions=[1, 2],
        widths=0.2,
        showfliers=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=2),
    )
    
    # Labels & styling
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["2021", "2025"], fontsize=14)
    ax.set_ylabel("Margin of Victory (%)", fontsize=14)
    ax.set_title("Victory Margin Distribution: NYC Democratic Primaries", fontsize=16)
    ax.set_ylim(-1, max(margins_2021.max(), margins_2025.max()) * 1.1)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Add sample sizes
    ax.text(1, -3, f'n={len(margins_2021)}', ha='center', fontsize=11)
    ax.text(2, -3, f'n={len(margins_2025)}', ha='center', fontsize=11)
    
    # Legend
    legend_handles = [Patch(facecolor=winner_color, alpha=0.5, label="Winner (0%)")]
    legend_handles += [
        Patch(facecolor=clr, alpha=0.5, label=lbl) for lbl, _, _, clr in categories_3bands
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", 
              bbox_to_anchor=(1.02, 1), fontsize=11)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(OUTPUT_DIR / "violin_competitive_bands_3.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "violin_competitive_bands_3.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'violin_competitive_bands_3.pdf'}")
    
    plt.close()


def create_exhaust_strategy_boxplot_2025():
    """
    Create boxplot of exhaust - strategy differences by letter for NYC 2025.
    Excludes letter A (winner) and REP elections.
    """
    print("\nCreating exhaust-strategy boxplot for NYC 2025...")
    
    # Load data
    df = pd.read_excel(RESULTS_DIR / "summary_table_nyc_2025.xlsx")
    
    # Filter out REP elections
    df = df[df['file_name'].str.contains("DEM", na=False)].copy()
    print(f"  DEM elections: {len(df)}")
    
    # Calculate differences for each letter
    data_list = []
    
    for _, row in df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find common candidates
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        for letter in common_candidates:
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            diff = exhaust_val - strategy_val
            
            data_list.append({
                'letter': letter,
                'diff': diff,
                'strategy': strategy_val,
                'exhaust': exhaust_val,
                'election_id': row.get('file_name', '')
            })
    
    df_analysis = pd.DataFrame(data_list)
    
    # Exclude letter A (winner, gap=0)
    df_no_a = df_analysis[df_analysis['letter'] != 'A'].copy()
    
    print(f"  Data points (excluding A): {len(df_no_a)}")
    
    # Get sorted letters
    letters_sorted = sorted(df_no_a['letter'].unique())
    print(f"  Letters: {letters_sorted}")
    
    # Create plot
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    ax = sns.boxplot(x='letter', y='diff', data=df_no_a, order=letters_sorted, palette='Set3')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.title('NYC 2025: Exhaust - Victory Gap Differences by Letter', fontsize=24)
    plt.xlabel('Candidate Letter', fontsize=22)
    plt.ylabel('Exhaust % - Victory Gap %', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ax.tick_params(labelsize=20)
    
    # Add count labels
    for i, letter in enumerate(letters_sorted):
        count = df_no_a[df_no_a['letter'] == letter].shape[0]
        ax.text(i, ax.get_ylim()[0] + 0.5, f'n={count}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'nyc_2025_exhaust_strategy_diff_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'nyc_2025_exhaust_strategy_diff_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'nyc_2025_exhaust_strategy_diff_boxplot.pdf'}")
    
    # Print summary stats
    print("\n  Summary statistics (Exhaust - Strategy):")
    summary = df_no_a.groupby('letter')['diff'].agg(['mean', 'median', 'count']).round(2)
    print(summary)
    
    plt.close()


def create_models_by_ratio_2025():
    """
    Create a plot showing probability models vs strategy/exhaust ratio for NYC 2025.
    Simplified version - shows strategy vs exhaust relationship.
    """
    print("\nCreating models by ratio plot for NYC 2025...")
    
    # Load data
    df = pd.read_excel(RESULTS_DIR / "summary_table_nyc_2025_with_margins.xlsx")
    df_dem = df[df['file_name'].str.contains("DEM", na=False)].copy()
    
    # Extract strategy and exhaust data for each candidate
    data_list = []
    
    for _, row in df_dem.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        for letter in common_candidates:
            if letter == 'A':  # Skip winner
                continue
            
            strategy_val = strategy_dict.get(letter, 0)
            exhaust_val = exhaust_dict.get(letter, 0)
            
            if strategy_val > 0 and exhaust_val > 0:
                ratio = strategy_val / exhaust_val
                
                # Calculate required preference percentage
                required_net_advantage = (strategy_val / exhaust_val) * 100
                required_pref_pct = (1 + required_net_advantage / 100) / 2 * 100
                
                # Simple probability model (based on required preference)
                # If required > 50%, probability decreases
                if required_pref_pct <= 50:
                    prob = 0.5 + (50 - required_pref_pct) / 100
                else:
                    prob = max(0, 0.5 - (required_pref_pct - 50) / 50)
                
                data_list.append({
                    'letter': letter,
                    'strategy': strategy_val,
                    'exhaust': exhaust_val,
                    'ratio': ratio,
                    'required_pref_pct': required_pref_pct,
                    'probability': prob,
                    'election': row['file_name']
                })
    
    if not data_list:
        print("  No data for models_by_ratio plot")
        return
    
    df_analysis = pd.DataFrame(data_list)
    df_sorted = df_analysis.sort_values('ratio')
    
    print(f"  Data points: {len(df_analysis)}")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot probability vs ratio
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_sorted['letter'].unique())))
    letter_colors = {letter: colors[i] for i, letter in enumerate(sorted(df_sorted['letter'].unique()))}
    
    for letter in sorted(df_sorted['letter'].unique()):
        letter_data = df_sorted[df_sorted['letter'] == letter]
        plt.scatter(letter_data['ratio'], letter_data['probability'], 
                   label=f'Candidate {letter}', 
                   color=letter_colors[letter], 
                   s=100, alpha=0.7, marker='o')
    
    # Add reference line at 0.5
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Add trend line
    from scipy import stats
    if len(df_sorted) > 2:
        slope, intercept, r, p, se = stats.linregress(df_sorted['ratio'], df_sorted['probability'])
        x_line = np.linspace(df_sorted['ratio'].min(), df_sorted['ratio'].max(), 100)
        plt.plot(x_line, slope * x_line + intercept, 'r-', alpha=0.5, 
                label=f'Trend (R²={r**2:.2f})')
    
    plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
    plt.ylabel('Estimated Probability of Alternate Winner', fontsize=14)
    plt.title('NYC 2025: Probability of Outcome Change vs Strategy-Exhaust Ratio', fontsize=16)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'nyc_2025_models_by_ratio.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'nyc_2025_models_by_ratio.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'nyc_2025_models_by_ratio.pdf'}")
    
    plt.close()


if __name__ == "__main__":
    create_violin_plot_3bands()
    create_exhaust_strategy_boxplot_2025()
    create_models_by_ratio_2025()
    print("\nDone!")
