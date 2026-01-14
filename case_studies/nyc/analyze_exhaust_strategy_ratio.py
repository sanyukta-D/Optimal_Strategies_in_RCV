import pandas as pd
import numpy as np
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

def analyze_exhaust_strategy_ratio():
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
                
                # Calculate what percentage of exhausted ballots would need to be strategic
                strategy_to_exhaust_ratio = (strategy_val / exhaust_val) * 100 if exhaust_val > 0 else 0
                
                # Calculate how many voters would need to complete ballots to match strategy
                voters_needed_pct = (exhaust_val - strategy_val) / exhaust_val * 100 if exhaust_val > 0 else 0
                
                candidates_with_exhaust_gt_strategy.append({
                    'letter': letter,
                    'exhaust': exhaust_val,
                    'strategy': strategy_val,
                    'diff': exhaust_val - strategy_val,
                    'strategy_to_exhaust_ratio': strategy_to_exhaust_ratio,
                    'voters_needed_pct': voters_needed_pct
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
                
                # Calculate what percentage of exhausted ballots would need to be strategic
                strategy_to_exhaust_ratio = (strategy_val / exhaust_val) * 100 if exhaust_val > 0 else 0
                
                # Calculate how many voters would need to complete ballots to match strategy
                voters_needed_pct = (exhaust_val - strategy_val) / exhaust_val * 100 if exhaust_val > 0 else 0
                
                candidates_with_exhaust_gt_strategy.append({
                    'letter': letter,
                    'exhaust': exhaust_val,
                    'strategy': strategy_val,
                    'diff': exhaust_val - strategy_val,
                    'strategy_to_exhaust_ratio': strategy_to_exhaust_ratio,
                    'voters_needed_pct': voters_needed_pct
                })
        
        if has_exhaust_gt_strategy_excluding_a:
            alaska_elections.append({
                'election_id': row.get('file_name', f"Alaska_{idx}"),
                'candidates': candidates_with_exhaust_gt_strategy
            })
    
    # Print results
    print(f"NYC Elections with exhaust > strategy (excluding A): {len(nyc_elections)}/{nyc_total_elections} ({len(nyc_elections)/nyc_total_elections*100:.1f}%)")
    print(f"Alaska Elections with exhaust > strategy (excluding A): {len(alaska_elections)}/{alaska_total_elections} ({len(alaska_elections)/alaska_total_elections*100:.1f}%)")
    
    # Calculate average ratios by letter
    nyc_letter_ratios = {}
    for election in nyc_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in nyc_letter_ratios:
                nyc_letter_ratios[letter] = {
                    'strategy_to_exhaust': [],
                    'voters_needed': []
                }
            nyc_letter_ratios[letter]['strategy_to_exhaust'].append(candidate['strategy_to_exhaust_ratio'])
            nyc_letter_ratios[letter]['voters_needed'].append(candidate['voters_needed_pct'])
    
    alaska_letter_ratios = {}
    for election in alaska_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in alaska_letter_ratios:
                alaska_letter_ratios[letter] = {
                    'strategy_to_exhaust': [],
                    'voters_needed': []
                }
            alaska_letter_ratios[letter]['strategy_to_exhaust'].append(candidate['strategy_to_exhaust_ratio'])
            alaska_letter_ratios[letter]['voters_needed'].append(candidate['voters_needed_pct'])
    
    print("\nNYC - For candidates with exhaust > strategy:")
    for letter in sorted(nyc_letter_ratios.keys()):
        strategy_to_exhaust = np.mean(nyc_letter_ratios[letter]['strategy_to_exhaust'])
        voters_needed = np.mean(nyc_letter_ratios[letter]['voters_needed'])
        count = len(nyc_letter_ratios[letter]['strategy_to_exhaust'])
        print(f"Letter {letter} (n={count}):")
        print(f"  - Strategy is {strategy_to_exhaust:.1f}% of exhaust on average")
        print(f"  - {voters_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")
    
    print("\nAlaska - For candidates with exhaust > strategy:")
    for letter in sorted(alaska_letter_ratios.keys()):
        strategy_to_exhaust = np.mean(alaska_letter_ratios[letter]['strategy_to_exhaust'])
        voters_needed = np.mean(alaska_letter_ratios[letter]['voters_needed'])
        count = len(alaska_letter_ratios[letter]['strategy_to_exhaust'])
        print(f"Letter {letter} (n={count}):")
        print(f"  - Strategy is {strategy_to_exhaust:.1f}% of exhaust on average")
        print(f"  - {voters_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")
    
    # Find elections with lowest strategy to exhaust ratio (where strategic voting is least effective)
    print("\nNYC - Top 5 elections with lowest strategy to exhaust ratio (where strategic voting is least effective):")
    all_nyc_candidates = []
    for election in nyc_elections:
        for candidate in election['candidates']:
            all_nyc_candidates.append({
                'election_id': election['election_id'],
                'letter': candidate['letter'],
                'exhaust': candidate['exhaust'],
                'strategy': candidate['strategy'],
                'strategy_to_exhaust_ratio': candidate['strategy_to_exhaust_ratio'],
                'voters_needed_pct': candidate['voters_needed_pct']
            })
    
    lowest_nyc = sorted(all_nyc_candidates, key=lambda x: x['strategy_to_exhaust_ratio'])[:5]
    for i, candidate in enumerate(lowest_nyc):
        print(f"{i+1}. Election: {candidate['election_id']}, Letter: {candidate['letter']}")
        print(f"   Strategy: {candidate['strategy']:.2f}%, Exhaust: {candidate['exhaust']:.2f}%")
        print(f"   Strategy is only {candidate['strategy_to_exhaust_ratio']:.2f}% of exhaust")
        print(f"   {candidate['voters_needed_pct']:.1f}% of exhausted ballot voters would need to complete ballots")
    
    print("\nAlaska - Top 5 elections with lowest strategy to exhaust ratio (where strategic voting is least effective):")
    all_alaska_candidates = []
    for election in alaska_elections:
        for candidate in election['candidates']:
            all_alaska_candidates.append({
                'election_id': election['election_id'],
                'letter': candidate['letter'],
                'exhaust': candidate['exhaust'],
                'strategy': candidate['strategy'],
                'strategy_to_exhaust_ratio': candidate['strategy_to_exhaust_ratio'],
                'voters_needed_pct': candidate['voters_needed_pct']
            })
    
    lowest_alaska = sorted(all_alaska_candidates, key=lambda x: x['strategy_to_exhaust_ratio'])[:5]
    for i, candidate in enumerate(lowest_alaska):
        print(f"{i+1}. Election: {candidate['election_id']}, Letter: {candidate['letter']}")
        print(f"   Strategy: {candidate['strategy']:.2f}%, Exhaust: {candidate['exhaust']:.2f}%")
        print(f"   Strategy is only {candidate['strategy_to_exhaust_ratio']:.2f}% of exhaust")
        print(f"   {candidate['voters_needed_pct']:.1f}% of exhausted ballot voters would need to complete ballots")
    
    # Summary statistics
    nyc_overall_strategy = sum(c['strategy'] for e in nyc_elections for c in e['candidates'])
    nyc_overall_exhaust = sum(c['exhaust'] for e in nyc_elections for c in e['candidates'])
    nyc_overall_ratio = (nyc_overall_strategy / nyc_overall_exhaust) * 100 if nyc_overall_exhaust > 0 else 0
    nyc_overall_needed = ((nyc_overall_exhaust - nyc_overall_strategy) / nyc_overall_exhaust) * 100 if nyc_overall_exhaust > 0 else 0
    
    alaska_overall_strategy = sum(c['strategy'] for e in alaska_elections for c in e['candidates'])
    alaska_overall_exhaust = sum(c['exhaust'] for e in alaska_elections for c in e['candidates'])
    alaska_overall_ratio = (alaska_overall_strategy / alaska_overall_exhaust) * 100 if alaska_overall_exhaust > 0 else 0
    alaska_overall_needed = ((alaska_overall_exhaust - alaska_overall_strategy) / alaska_overall_exhaust) * 100 if alaska_overall_exhaust > 0 else 0
    
    print("\nOverall summary:")
    print(f"NYC: Strategy is {nyc_overall_ratio:.1f}% of exhaust on average")
    print(f"NYC: {nyc_overall_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")
    print(f"Alaska: Strategy is {alaska_overall_ratio:.1f}% of exhaust on average")
    print(f"Alaska: {alaska_overall_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")

if __name__ == "__main__":
    analyze_exhaust_strategy_ratio() 