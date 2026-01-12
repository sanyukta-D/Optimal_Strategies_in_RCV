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

def analyze_elections_excluding_a():
    # Load NYC data
    nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
    nyc_df = pd.read_excel(nyc_path)
    nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
    
    # Load Alaska data
    alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
    alaska_df = pd.read_excel(alaska_path)
    
    # Analyze NYC elections
    nyc_elections = []
    nyc_total_elections = 0
    
    for idx, row in nyc_df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find candidates that appear in both dictionaries
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        # Skip if no common candidates
        if not common_candidates:
            continue
            
        nyc_total_elections += 1
        
        # Check if any non-A candidate has exhaust > strategy
        has_exhaust_gt_strategy_excluding_a = False
        candidates_with_exhaust_gt_strategy = []
        
        for letter in common_candidates:
            if letter == 'A':
                continue  # Skip letter A
                
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            
            if exhaust_val > strategy_val:
                has_exhaust_gt_strategy_excluding_a = True
                candidates_with_exhaust_gt_strategy.append({
                    'letter': letter,
                    'exhaust': exhaust_val,
                    'strategy': strategy_val,
                    'diff': exhaust_val - strategy_val
                })
        
        if has_exhaust_gt_strategy_excluding_a:
            nyc_elections.append({
                'election_id': row.get('file_name', f"NYC_{idx}"),
                'candidates': candidates_with_exhaust_gt_strategy
            })
    
    # Analyze Alaska elections
    alaska_elections = []
    alaska_total_elections = 0
    
    for idx, row in alaska_df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find candidates that appear in both dictionaries
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        # Skip if no common candidates
        if not common_candidates:
            continue
            
        alaska_total_elections += 1
        
        # Check if any non-A candidate has exhaust > strategy
        has_exhaust_gt_strategy_excluding_a = False
        candidates_with_exhaust_gt_strategy = []
        
        for letter in common_candidates:
            if letter == 'A':
                continue  # Skip letter A
                
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            
            if exhaust_val > strategy_val:
                has_exhaust_gt_strategy_excluding_a = True
                candidates_with_exhaust_gt_strategy.append({
                    'letter': letter,
                    'exhaust': exhaust_val,
                    'strategy': strategy_val,
                    'diff': exhaust_val - strategy_val
                })
        
        if has_exhaust_gt_strategy_excluding_a:
            alaska_elections.append({
                'election_id': row.get('file_name', f"Alaska_{idx}"),
                'candidates': candidates_with_exhaust_gt_strategy
            })
    
    # Print results
    print(f"NYC Elections with exhaust > strategy (excluding A): {len(nyc_elections)}/{nyc_total_elections} ({len(nyc_elections)/nyc_total_elections*100:.1f}%)")
    print(f"Alaska Elections with exhaust > strategy (excluding A): {len(alaska_elections)}/{alaska_total_elections} ({len(alaska_elections)/alaska_total_elections*100:.1f}%)")
    
    # Count by letter
    nyc_letter_counts = {}
    for election in nyc_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in nyc_letter_counts:
                nyc_letter_counts[letter] = 0
            nyc_letter_counts[letter] += 1
    
    alaska_letter_counts = {}
    for election in alaska_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in alaska_letter_counts:
                alaska_letter_counts[letter] = 0
            alaska_letter_counts[letter] += 1
    
    print("\nNYC - Count of elections where each letter has exhaust > strategy:")
    for letter in sorted(nyc_letter_counts.keys()):
        print(f"Letter {letter}: {nyc_letter_counts[letter]} elections")
    
    print("\nAlaska - Count of elections where each letter has exhaust > strategy:")
    for letter in sorted(alaska_letter_counts.keys()):
        print(f"Letter {letter}: {alaska_letter_counts[letter]} elections")
    
    # Calculate average difference for each letter
    nyc_letter_diffs = {}
    for election in nyc_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in nyc_letter_diffs:
                nyc_letter_diffs[letter] = []
            nyc_letter_diffs[letter].append(candidate['diff'])
    
    alaska_letter_diffs = {}
    for election in alaska_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in alaska_letter_diffs:
                alaska_letter_diffs[letter] = []
            alaska_letter_diffs[letter].append(candidate['diff'])
    
    print("\nNYC - Average difference (exhaust - strategy) for each letter where exhaust > strategy:")
    for letter in sorted(nyc_letter_diffs.keys()):
        avg_diff = sum(nyc_letter_diffs[letter]) / len(nyc_letter_diffs[letter])
        print(f"Letter {letter}: {avg_diff:.2f}% (n={len(nyc_letter_diffs[letter])})")
    
    print("\nAlaska - Average difference (exhaust - strategy) for each letter where exhaust > strategy:")
    for letter in sorted(alaska_letter_diffs.keys()):
        avg_diff = sum(alaska_letter_diffs[letter]) / len(alaska_letter_diffs[letter])
        print(f"Letter {letter}: {avg_diff:.2f}% (n={len(alaska_letter_diffs[letter])})")
    
    # List elections with the largest differences
    print("\nNYC - Top 5 elections with largest exhaust > strategy differences (excluding A):")
    all_nyc_candidates = []
    for election in nyc_elections:
        for candidate in election['candidates']:
            all_nyc_candidates.append({
                'election_id': election['election_id'],
                'letter': candidate['letter'],
                'diff': candidate['diff'],
                'exhaust': candidate['exhaust'],
                'strategy': candidate['strategy']
            })
    
    top_nyc = sorted(all_nyc_candidates, key=lambda x: x['diff'], reverse=True)[:5]
    for i, candidate in enumerate(top_nyc):
        print(f"{i+1}. Election: {candidate['election_id']}, Letter: {candidate['letter']}, Diff: {candidate['diff']:.2f}%, Exhaust: {candidate['exhaust']:.2f}%, Strategy: {candidate['strategy']:.2f}%")
    
    print("\nAlaska - Top 5 elections with largest exhaust > strategy differences (excluding A):")
    all_alaska_candidates = []
    for election in alaska_elections:
        for candidate in election['candidates']:
            all_alaska_candidates.append({
                'election_id': election['election_id'],
                'letter': candidate['letter'],
                'diff': candidate['diff'],
                'exhaust': candidate['exhaust'],
                'strategy': candidate['strategy']
            })
    
    top_alaska = sorted(all_alaska_candidates, key=lambda x: x['diff'], reverse=True)[:5]
    for i, candidate in enumerate(top_alaska):
        print(f"{i+1}. Election: {candidate['election_id']}, Letter: {candidate['letter']}, Diff: {candidate['diff']:.2f}%, Exhaust: {candidate['exhaust']:.2f}%, Strategy: {candidate['strategy']:.2f}%")

if __name__ == "__main__":
    analyze_elections_excluding_a() 