import pandas as pd
import re
import ast  # Needed for safe literal evaluation

def parse_main_and_synergy(data_list):
    """Returns (main_val, synergy_dict) from something like [31.875, {'A': 31.753, 'FA': 0.122}]."""
    if not isinstance(data_list, list) or len(data_list) == 0:
        return (0.0, {})
    main_val = data_list[0]
    synergy_dict = data_list[1] if len(data_list) > 1 and isinstance(data_list[1], dict) else {}
    return (main_val, synergy_dict)

def format_candidate_with_synergy(candidate, data_list):
    """
    E.g. candidate='A', data_list=[31.875, {'A':31.753,'FA':0.122}] 
    => 'A: 31.9\\% (FA: 0.122\\%)'
    """
    main_val, synergy_dict = parse_main_and_synergy(data_list)
    synergy_parts = []
    for k, v in synergy_dict.items():
        # Only show synergy for keys that are not this candidate
        if k != candidate:
            synergy_parts.append(f"{k}: {v:.3f}\\%")
    synergy_str = f" ({', '.join(synergy_parts)})" if synergy_parts else ""
    return f"{candidate}: {round(main_val,1)}\\%{synergy_str}"

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

def compare_strategy_with_exhaust_by_candidate(strategy_str, exhaust_str):
    """
    Compare the strategy percentage with the exhaust percentage for each candidate.
    Returns a dictionary mapping candidates to whether their exhaust percentage is higher than strategy.
    Only includes candidates that appear in both strategy and exhaust dictionaries.
    """
    strategy_dict = extract_strategy_dict(strategy_str)
    exhaust_dict = extract_exhaust_dict(exhaust_str)
    
    # Find candidates that appear in both dictionaries
    common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
    
    # Compare for each candidate
    result = {}
    for candidate in common_candidates:
        strategy_val = strategy_dict[candidate]
        exhaust_val = exhaust_dict[candidate]
        result[candidate] = {
            'strategy': strategy_val,
            'exhaust': exhaust_val,
            'exhaust_greater': exhaust_val > strategy_val
        }
    
    return result

def format_comparison_results(comparison_dict):
    """
    Format the comparison results into a readable string.
    """
    if not comparison_dict:
        return "No comparison data available"
    
    # Sort candidates alphabetically
    sorted_candidates = sorted(comparison_dict.keys())
    
    parts = []
    for candidate in sorted_candidates:
        data = comparison_dict[candidate]
        strategy_val = data['strategy']
        exhaust_val = data['exhaust']
        comparison_symbol = "<" if data['exhaust_greater'] else "≥"
        parts.append(f"{candidate}: Strategy {strategy_val:.1f}% {comparison_symbol} Exhaust {exhaust_val:.1f}%")
    
    return ", ".join(parts)

def count_exhaust_greater_than_strategy(comparison_dict):
    """
    Count how many candidates have exhaust percentage greater than strategy percentage.
    """
    count = sum(1 for data in comparison_dict.values() if data['exhaust_greater'])
    total = len(comparison_dict)
    return count, total

def readable_strategies(strategy_str):
    """
    Convert the 'Strategies' dict into a single line with ascending order of main candidate's percentage.
    Example:
      {
        'A': [31.875, {'A': 31.753, 'FA': 0.122}],
        'B': [25.919, {'B':20.765, 'D':5.153}]
      }
    => 'B: 25.9\\% (D: 5.153\\%), A: 31.9\\% (FA: 0.122\\%)'
    """
    try:
        strategy_dict = ast.literal_eval(strategy_str)
        if not isinstance(strategy_dict, dict) or not strategy_dict:
            return "None"
        
        # Sort by the main candidate percentage *ascending*
        sorted_candidates = sorted(
            strategy_dict.items(),
            key=lambda item: item[1][0],  # item[1][0] is the main percentage
            reverse=False
        )

        parts = []
        for candidate, data_list in sorted_candidates:
            candidate_str = format_candidate_with_synergy(candidate, data_list)
            parts.append(candidate_str)

        return ", ".join(parts)

    except (ValueError, SyntaxError, TypeError):
        return "None"

def format_district_name(file_name):
    if "CouncilMember" in file_name:
        district_num = file_name.split("CouncilMember")[1].split("CouncilDistrict")[0]
        return f"Council District {district_num}"
    elif "BoroughPresident" in file_name:
        borough = file_name.split("BoroughPresident")[1].split(".")[0]
        return f"{borough} President"
    elif "MayorCitywide" in file_name:
        return "Mayor Citywide"
    elif "ComptrollerCitywide" in file_name:
        return "Comptroller"
    elif "PublicAdvocateCitywide" in file_name:
        return "Public Advocate"
    else:
        return file_name

def extract_sort_key(name):
    if "Council District" in name:
        match = re.search(r'(\d+)(?:st|nd|rd|th)', name)
        if match:
            return (1, int(match.group(1)))
    elif "Borough President" in name:
        return (2, name)
    elif "Mayor Citywide" in name:
        return (3, name)
    elif "Comptroller" in name:
        return (4, name)
    elif "Public Advocate" in name:
        return (5, name)
    else:
        return (6, name)

def process_and_generate_latex(file_path):
    # Load the spreadsheet
    df = pd.read_excel(file_path)
    # Filter for DEM elections
    df_dem = df[df['file_name'].str.contains("DEM", na=False)].copy()

    # Format district and synergy
    df_dem['District Name'] = df_dem['file_name'].apply(format_district_name)
    df_dem['Candidate Win Margin'] = df_dem['Strategies'].apply(readable_strategies)
    
    # Compare strategy with exhaust for each candidate
    df_dem['Comparison Results'] = df_dem.apply(
        lambda row: compare_strategy_with_exhaust_by_candidate(row['Strategies'], row['exhaust_percents']), 
        axis=1
    )
    
    # Format comparison results
    df_dem['Strategy vs Exhaust'] = df_dem['Comparison Results'].apply(format_comparison_results)
    
    # Count candidates with exhaust > strategy
    df_dem['Exhaust > Strategy Count'] = df_dem['Comparison Results'].apply(
        lambda results: count_exhaust_greater_than_strategy(results)
    )
    
    # Calculate percentage of candidates with exhaust > strategy
    df_dem['Exhaust > Strategy Percentage'] = df_dem['Exhaust > Strategy Count'].apply(
        lambda counts: f"{counts[0]}/{counts[1]} ({counts[0]/counts[1]*100:.1f}%)" if counts[1] > 0 else "N/A"
    )

    # Build summary
    df_summary = df_dem[['District Name', 'total_votes', 'num_candidates', 'Candidate Win Margin', 
                         'Strategy vs Exhaust', 'Exhaust > Strategy Percentage', 'budget_percent']].copy()
    df_summary.rename(columns={
        'total_votes': 'Votes',
        'num_candidates': 'Candidates',
        'budget_percent': 'Allowance'
    }, inplace=True)
    df_summary['Allowance'] = df_summary['Allowance'].apply(lambda x: f"{float(x):.1f}")

    # Sort by district name
    df_summary['sort_key'] = df_summary['District Name'].apply(extract_sort_key)
    df_summary_sorted = df_summary.sort_values('sort_key').drop(columns='sort_key')

    # Convert to LaTeX
    latex_table = df_summary_sorted.to_latex(
        index=False, 
        caption="NYC Democratic Election Results Summary", 
        label="tab:nyc_elections",
        escape=False
    )

    # Save to file
    with open("nyc_election_summary.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX file saved as nyc_election_summary.tex")
    
    # Print summary of candidates where exhaust percent is higher than strategy percent
    total_candidates_with_exhaust_greater = sum(count[0] for count in df_dem['Exhaust > Strategy Count'])
    total_candidates = sum(count[1] for count in df_dem['Exhaust > Strategy Count'])
    print(f"\nCandidates where exhaust percent > strategy percent: {total_candidates_with_exhaust_greater}/{total_candidates} ({total_candidates_with_exhaust_greater/total_candidates*100:.1f}%)")
    
    # Count by letter (candidate)
    letter_counts = {
        'letter': [],
        'exhaust_greater_count': [],
        'total_count': [],
        'percentage': []
    }
    
    # Collect all comparison results
    all_comparisons = []
    for results in df_dem['Comparison Results']:
        all_comparisons.append(results)
    
    # Get unique letters across all districts
    all_letters = set()
    for results in all_comparisons:
        all_letters.update(results.keys())
    
    # Count for each letter
    for letter in sorted(all_letters):
        exhaust_greater_count = sum(1 for results in all_comparisons if letter in results and results[letter]['exhaust_greater'])
        total_count = sum(1 for results in all_comparisons if letter in results)
        percentage = (exhaust_greater_count / total_count * 100) if total_count > 0 else 0
        
        letter_counts['letter'].append(letter)
        letter_counts['exhaust_greater_count'].append(exhaust_greater_count)
        letter_counts['total_count'].append(total_count)
        letter_counts['percentage'].append(percentage)
    
    # Create a DataFrame for letter counts
    df_letter_counts = pd.DataFrame(letter_counts)
    
    # Print letter counts
    print("\nExhaust > Strategy counts by letter (candidate):")
    for _, row in df_letter_counts.iterrows():
        print(f"Letter {row['letter']}: {row['exhaust_greater_count']}/{row['total_count']} ({row['percentage']:.1f}%)")
    
    # Save letter counts to CSV
    df_letter_counts.to_csv("exhaust_vs_strategy_by_letter.csv", index=False)
    print("\nLetter counts saved to exhaust_vs_strategy_by_letter.csv")
    
    # Save the detailed comparison to a CSV file for further analysis
    df_summary[['District Name', 'Strategy vs Exhaust', 'Exhaust > Strategy Percentage']].to_csv("exhaust_vs_strategy_comparison.csv", index=False)
    print("Detailed comparison saved to exhaust_vs_strategy_comparison.csv")


# Example Usage:
file_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
process_and_generate_latex(file_path)
