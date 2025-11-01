import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from portland_strategy_data import create_portland_summary_dataframe, create_portland_strategy_data
from STVandIRV_results import STV_optimal_result_simple
from Case_Studies.Portland_City_Council_Data_and_Analysis.load_district_data import district_data
import case_study_helpers
from scipy import stats

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

def analyze_portland_exhaust_strategy_ratio():
    """
    Analyze exhaust vs strategy ratios for Portland multi-winner elections.
    Adapted from the single-winner analysis approach.
    """
    # Load Portland data
    portland_df = create_portland_summary_dataframe()
    strategy_data = create_portland_strategy_data()
    
    print("Portland Multi-Winner Exhaust vs Strategy Analysis")
    print("=" * 60)
    print(f"Analyzing {len(portland_df)} districts with strategy data")
    
    # Analyze Portland elections
    portland_elections = []
    portland_total_elections = 0
    
    for idx, row in portland_df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Find candidates that appear in both dictionaries
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        # Skip if no common candidates
        if not common_candidates:
            continue
            
        portland_total_elections += 1
        district = row['district']
        
        # Check if any candidate has exhaust > strategy (including winners)
        # Unlike single-winner analysis, we include all candidates since multi-winner dynamics are different
        has_exhaust_gt_strategy = False
        candidates_with_exhaust_gt_strategy = []
        
        for letter in common_candidates:
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            
            if exhaust_val > strategy_val:
                has_exhaust_gt_strategy = True
                
                # Calculate what percentage of exhausted ballots would need to be strategic
                strategy_to_exhaust_ratio = (strategy_val / exhaust_val) * 100 if exhaust_val > 0 else 0
                
                # Calculate how many voters would need to complete ballots to match strategy
                voters_needed_pct = (exhaust_val - strategy_val) / exhaust_val * 100 if exhaust_val > 0 else 0
                
                # Get candidate name
                candidates_mapping = strategy_data[district]['candidates_mapping']
                reverse_mapping = {v: k for k, v in candidates_mapping.items()}
                candidate_name = reverse_mapping.get(letter, f"Unknown ({letter})")
                
                candidates_with_exhaust_gt_strategy.append({
                    'letter': letter,
                    'candidate_name': candidate_name,
                    'exhaust': exhaust_val,
                    'strategy': strategy_val,
                    'diff': exhaust_val - strategy_val,
                    'strategy_to_exhaust_ratio': strategy_to_exhaust_ratio,
                    'voters_needed_pct': voters_needed_pct
                })
        
        if has_exhaust_gt_strategy:
            portland_elections.append({
                'election_id': row['file_name'],
                'district': district,
                'candidates': candidates_with_exhaust_gt_strategy
            })
    
    # Print results
    print(f"Portland Districts with exhaust > strategy: {len(portland_elections)}/{portland_total_elections} ({len(portland_elections)/portland_total_elections*100:.1f}%)")
    
    # Calculate average ratios by letter
    portland_letter_ratios = {}
    for election in portland_elections:
        for candidate in election['candidates']:
            letter = candidate['letter']
            if letter not in portland_letter_ratios:
                portland_letter_ratios[letter] = {
                    'strategy_to_exhaust': [],
                    'voters_needed': [],
                    'candidate_names': []
                }
            portland_letter_ratios[letter]['strategy_to_exhaust'].append(candidate['strategy_to_exhaust_ratio'])
            portland_letter_ratios[letter]['voters_needed'].append(candidate['voters_needed_pct'])
            portland_letter_ratios[letter]['candidate_names'].append(candidate['candidate_name'])
    
    print("\nPortland - For candidates with exhaust > strategy:")
    for letter in sorted(portland_letter_ratios.keys()):
        strategy_to_exhaust = np.mean(portland_letter_ratios[letter]['strategy_to_exhaust'])
        voters_needed = np.mean(portland_letter_ratios[letter]['voters_needed'])
        count = len(portland_letter_ratios[letter]['strategy_to_exhaust'])
        candidate_names = list(set(portland_letter_ratios[letter]['candidate_names']))
        
        print(f"Letter {letter} - {', '.join(candidate_names)} (n={count}):")
        print(f"  - Strategy is {strategy_to_exhaust:.1f}% of exhaust on average")
        print(f"  - {voters_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")
    
    # Find elections with lowest strategy to exhaust ratio (where strategic voting is least effective)
    print("\nPortland - Elections with lowest strategy to exhaust ratio (where strategic voting is least effective):")
    all_portland_candidates = []
    for election in portland_elections:
        for candidate in election['candidates']:
            all_portland_candidates.append({
                'election_id': election['election_id'],
                'district': election['district'],
                'letter': candidate['letter'],
                'candidate_name': candidate['candidate_name'],
                'exhaust': candidate['exhaust'],
                'strategy': candidate['strategy'],
                'strategy_to_exhaust_ratio': candidate['strategy_to_exhaust_ratio'],
                'voters_needed_pct': candidate['voters_needed_pct']
            })
    
    lowest_portland = sorted(all_portland_candidates, key=lambda x: x['strategy_to_exhaust_ratio'])
    for i, candidate in enumerate(lowest_portland):
        print(f"{i+1}. District {candidate['district']}, {candidate['candidate_name']} ({candidate['letter']})")
        print(f"   Strategy: {candidate['strategy']:.2f}%, Exhaust: {candidate['exhaust']:.2f}%")
        print(f"   Strategy is only {candidate['strategy_to_exhaust_ratio']:.2f}% of exhaust")
        print(f"   {candidate['voters_needed_pct']:.1f}% of exhausted ballot voters would need to complete ballots")
        print()
    
    # Summary statistics
    portland_overall_strategy = sum(c['strategy'] for e in portland_elections for c in e['candidates'])
    portland_overall_exhaust = sum(c['exhaust'] for e in portland_elections for c in e['candidates'])
    portland_overall_ratio = (portland_overall_strategy / portland_overall_exhaust) * 100 if portland_overall_exhaust > 0 else 0
    portland_overall_needed = ((portland_overall_exhaust - portland_overall_strategy) / portland_overall_exhaust) * 100 if portland_overall_exhaust > 0 else 0
    
    print("Overall summary:")
    print(f"Portland: Strategy is {portland_overall_ratio:.1f}% of exhaust on average")
    print(f"Portland: {portland_overall_needed:.1f}% of exhausted ballot voters would need to complete ballots to match strategy")
    
    return portland_elections, portland_letter_ratios

def create_portland_detailed_analysis():
    """
    Create a detailed analysis showing strategy vs exhaust for each district.
    """
    strategy_data = create_portland_strategy_data()
    portland_df = create_portland_summary_dataframe()
    
    print("\nDetailed District Analysis")
    print("=" * 80)
    
    for idx, row in portland_df.iterrows():
        district = row['district']
        print(f"\nDistrict {district} Analysis:")
        print("-" * 40)
        
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        # Get candidate mapping for names
        candidates_mapping = strategy_data[district]['candidates_mapping']
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        # Find all candidates with either strategy or exhaust data
        all_candidates = set(strategy_dict.keys()) | set(exhaust_dict.keys())
        
        print(f"{'Candidate':<25} {'Letter':<6} {'Strategy %':<12} {'Exhaust %':<12} {'Difference':<12} {'Status'}")
        print("-" * 80)
        
        for letter in sorted(all_candidates):
            candidate_name = reverse_mapping.get(letter, f"Unknown ({letter})")
            strategy_val = strategy_dict.get(letter, 0.0)
            exhaust_val = exhaust_dict.get(letter, 0.0)
            diff = exhaust_val - strategy_val
            
            if strategy_val == 0 and exhaust_val > 0:
                status = "Exhaust only"
            elif strategy_val > 0 and exhaust_val == 0:
                status = "Strategy only"
            elif exhaust_val > strategy_val:
                status = "Exhaust > Strategy"
            elif strategy_val > exhaust_val:
                status = "Strategy > Exhaust"
            else:
                status = "Equal"
            
            print(f"{candidate_name:<25} {letter:<6} {strategy_val:<12.2f} {exhaust_val:<12.2f} {diff:<12.2f} {status}")
        
        print(f"\nTotal ballots in district: {row['total_ballots']:,}")

def create_visualization():
    """
    Create visualizations comparing Portland exhaust vs strategy data.
    """
    portland_df = create_portland_summary_dataframe()
    
    # Set plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Prepare data for visualization
    visualization_data = []
    
    for idx, row in portland_df.iterrows():
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
        
        for letter in common_candidates:
            strategy_val = strategy_dict[letter]
            exhaust_val = exhaust_dict[letter]
            
            visualization_data.append({
                'district': row['district'],
                'letter': letter,
                'strategy': strategy_val,
                'exhaust': exhaust_val,
                'diff': exhaust_val - strategy_val,
                'exhaust_greater': exhaust_val > strategy_val
            })
    
    viz_df = pd.DataFrame(visualization_data)
    
    # Create scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Scatter plot by district
    districts = sorted(viz_df['district'].unique())
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, district in enumerate(districts):
        district_df = viz_df[viz_df['district'] == district]
        axes[0].scatter(district_df['exhaust'], district_df['strategy'], 
                       label=f'District {district}', alpha=0.7, color=colors[i % len(colors)])
    
    # Add diagonal line where exhaust = strategy
    max_val = max(viz_df['exhaust'].max(), viz_df['strategy'].max()) * 1.1
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Exhaust = Strategy')
    axes[0].set_xlabel('Exhaust Percentage')
    axes[0].set_ylabel('Strategy Percentage')
    axes[0].set_title('Portland: Exhaust vs Strategy by District')
    axes[0].legend()
    
    # Bar chart showing exhaust > strategy counts
    district_counts = []
    district_totals = []
    
    for district in districts:
        district_df = viz_df[viz_df['district'] == district]
        count = sum(district_df['exhaust_greater'])
        total = len(district_df)
        district_counts.append(count)
        district_totals.append(total)
    
    bars = axes[1].bar([f'District {d}' for d in districts], 
                      [c/t*100 if t > 0 else 0 for c, t in zip(district_counts, district_totals)])
    axes[1].set_ylabel('Percentage of Candidates')
    axes[1].set_title('Portland: Candidates with Exhaust > Strategy by District')
    
    # Add count labels on bars
    for i, (count, total) in enumerate(zip(district_counts, district_totals)):
        axes[1].text(i, count/total*100 + 2, f'{count}/{total}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("portland_exhaust_vs_strategy_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Run comprehensive Portland exhaust vs strategy analysis.
    """
    print("Portland Multi-Winner STV: Exhaust vs Strategy Analysis")
    print("=" * 70)
    
    # Run main analysis
    elections, letter_ratios = analyze_portland_exhaust_strategy_ratio()
    
    # Create detailed analysis
    create_portland_detailed_analysis()
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    create_visualization()
    
    print("\nAnalysis complete! Check 'portland_exhaust_vs_strategy_analysis.png' for visualizations.")


import pandas as pd
from portland_strategy_data import create_portland_summary_dataframe, create_portland_strategy_data
from portland_exhaust_strategy_analysis import extract_strategy_dict, extract_exhaust_dict

def create_portland_summary():
    """
    Create a comprehensive summary of Portland multi-winner exhaust vs strategy analysis.
    Focus only on non-winning candidates with actual strategy data.
    """
    print("Portland Multi-Winner STV: Exhaust vs Strategy Summary")
    print("=" * 65)
    print("(Analysis excludes winners A, B, C and candidates without strategy data)")
    
    # Load data
    portland_df = create_portland_summary_dataframe()
    strategy_data = create_portland_strategy_data()
    
    print("\n1. KEY FINDINGS:")
    print("-" * 40)
    print("• 100% of districts (3/3) show non-winning candidates with exhaust > strategy")
    print("• Only analyzing candidates with provided strategy data (others assumed strategy > exhaust)")
    print("• Winners (A, B, C) excluded from analysis as they achieved their goal")
    
    print("\n2. DISTRICT-BY-DISTRICT BREAKDOWN:")
    print("-" * 40)
    
    total_candidates_with_strategies = 0
    total_candidates_with_exhaust_gt_strategy = 0
    
    for idx, row in portland_df.iterrows():
        district = row['district']
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        
        candidates_mapping = strategy_data[district]['candidates_mapping']
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        print(f"\nDistrict {district}:")
        print(f"  Total ballots: {row['total_ballots']:,}")
        
        # Count candidates with strategies (only non-winners with data)
        candidates_with_strategies = len(strategy_dict)
        total_candidates_with_strategies += candidates_with_strategies
        
        # Count candidates with exhaust > strategy
        exhaust_gt_strategy = 0
        for letter in strategy_dict:
            if letter in exhaust_dict and exhaust_dict[letter] > strategy_dict[letter]:
                exhaust_gt_strategy += 1
        
        total_candidates_with_exhaust_gt_strategy += exhaust_gt_strategy
        
        print(f"  Non-winning candidates with strategy data: {candidates_with_strategies}")
        print(f"  Candidates with exhaust > strategy: {exhaust_gt_strategy}/{candidates_with_strategies} ({exhaust_gt_strategy/candidates_with_strategies*100:.1f}%)")
        
        # Show specific candidates with strategies
        print("  Strategic candidates analyzed:")
        for letter, percentage in strategy_dict.items():
            candidate_name = reverse_mapping[letter]
            exhaust_pct = exhaust_dict.get(letter, 0.0)
            ratio = (percentage / exhaust_pct * 100) if exhaust_pct > 0 else 0
            print(f"    {candidate_name} ({letter}): {percentage}% strategy vs {exhaust_pct:.2f}% exhaust (ratio: {ratio:.1f}%)")
        
        # Show winners excluded
        winners = []
        for letter in ['A', 'B', 'C']:
            if letter in reverse_mapping:
                winners.append(f"{reverse_mapping[letter]} ({letter})")
        if winners:
            print(f"  Winners excluded from analysis: {', '.join(winners)}")
    
    print(f"\n3. AGGREGATE STATISTICS:")
    print("-" * 40)
    print(f"Total non-winning candidates with strategy data: {total_candidates_with_strategies}")
    print(f"Total with exhaust > strategy: {total_candidates_with_exhaust_gt_strategy}/{total_candidates_with_strategies} ({total_candidates_with_exhaust_gt_strategy/total_candidates_with_strategies*100:.1f}%)")
    
    print("\n4. COMPARISON WITH SINGLE-WINNER PATTERNS:")
    print("-" * 40)
    print("Multi-winner STV vs Single-winner RCV differences:")
    print("• Winner exclusion: Unlike single-winner, we exclude winners A, B, C")
    print("• Focus on losing candidates who could benefit from strategic voting")
    print("• Strategic effectiveness varies widely among non-winners")
    print("• Multi-round nature creates different exhaustion patterns than single-winner")
    
    print("\n5. MOST STRATEGIC OPPORTUNITIES (Lowest Strategy/Exhaust Ratios):")
    print("-" * 40)
    
    # Collect all candidates with strategy data (non-winners only)
    all_candidates = []
    for idx, row in portland_df.iterrows():
        district = row['district']
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        candidates_mapping = strategy_data[district]['candidates_mapping']
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        for letter, strategy_pct in strategy_dict.items():
            if letter in exhaust_dict:
                exhaust_pct = exhaust_dict[letter]
                if exhaust_pct > strategy_pct:
                    ratio = (strategy_pct / exhaust_pct * 100) if exhaust_pct > 0 else 0
                    candidate_name = reverse_mapping[letter]
                    all_candidates.append({
                        'district': district,
                        'name': candidate_name,
                        'letter': letter,
                        'strategy': strategy_pct,
                        'exhaust': exhaust_pct,
                        'ratio': ratio,
                        'voters_needed': (exhaust_pct - strategy_pct) / exhaust_pct * 100
                    })
    
    # Sort by ratio (ascending - lowest ratios first)
    sorted_candidates = sorted(all_candidates, key=lambda x: x['ratio'])
    
    for i, candidate in enumerate(sorted_candidates):
        print(f"{i+1}. District {candidate['district']}: {candidate['name']} ({candidate['letter']})")
        print(f"   Strategy: {candidate['strategy']:.2f}% | Exhaust: {candidate['exhaust']:.2f}%")
        print(f"   Strategy is only {candidate['ratio']:.1f}% of exhaust")
        print(f"   {candidate['voters_needed']:.1f}% of exhausted voters would need to complete ballots")
        print()

def create_comparison_table():
    """
    Create a comparison table showing the data in a clean format.
    Only includes non-winning candidates with strategy data.
    """
    portland_df = create_portland_summary_dataframe()
    strategy_data = create_portland_strategy_data()
    
    print("\nPORTLAND EXHAUST vs STRATEGY COMPARISON TABLE")
    print("(Non-winning candidates with strategy data only)")
    print("=" * 85)
    print(f"{'District':<8} {'Candidate':<20} {'Letter':<6} {'Strategy %':<12} {'Exhaust %':<12} {'Ratio %':<10} {'Gap %':<10}")
    print("-" * 85)
    
    for idx, row in portland_df.iterrows():
        district = row['district']
        strategy_dict = extract_strategy_dict(row['Strategies'])
        exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
        candidates_mapping = strategy_data[district]['candidates_mapping']
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        for letter, strategy_pct in strategy_dict.items():
            if letter in exhaust_dict:
                exhaust_pct = exhaust_dict[letter]
                ratio = (strategy_pct / exhaust_pct * 100) if exhaust_pct > 0 else 0
                gap = exhaust_pct - strategy_pct
                candidate_name = reverse_mapping[letter][:18]  # Truncate long names
                
                print(f"{district:<8} {candidate_name:<20} {letter:<6} {strategy_pct:<12.2f} {exhaust_pct:<12.2f} {ratio:<10.1f} {gap:<10.2f}")
    
    print("=" * 85)
    print("\nNotes:")
    print("- Analysis excludes winners A, B, C (they achieved their goal)")
    print("- Candidates without strategy data assumed to have strategy > exhaust")
    print("- Ratio %: What percentage of exhaust is covered by strategy (strategy/exhaust * 100)")
    print("- Gap %: Difference between exhaust and strategy percentages")
    print("- Lower ratios indicate greater strategic voting opportunities")

def get_active_candidates_at_elimination(district_number, k=3):
    """
    Get the set of active candidates at the time each candidate gets eliminated.
    
    Args:
        district_number (int): District number (1-4)
        k (int): Number of winners (default: 3)
        
    Returns:
        dict: Mapping from candidate letter to set of active candidates when they were eliminated
    """
    # Get district data
    df = district_data[district_number]['df']
    candidates_mapping = district_data[district_number]['candidates_mapping']
    
    # Convert ballot data to the format needed for STV analysis
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
    candidates = list(candidates_mapping.values())
    
    # Calculate quota (Droop quota)
    total_votes = sum(ballot_counts.values())
    quota = (total_votes // (k + 1)) + 1
    
    # Get the social choice order
    event_log, result_dict, round_history = STV_optimal_result_simple(candidates, ballot_counts, k, quota)
    
    # Track active candidates at each elimination
    active_at_elimination = {}
    candidates_remaining = set(candidates)
    
    print(f"\nDistrict {district_number} - Social Choice Order Analysis:")
    print("=" * 60)
    print(f"Total candidates: {len(candidates)}")
    print(f"Winners needed (k): {k}")
    print(f"Quota: {quota}")
    
    # Process each round based on the event log
    for round_num, (candidate, is_winner) in enumerate(event_log, 1):
        # Record active candidates at the time of this candidate's processing
        active_at_elimination[candidate] = candidates_remaining.copy()
        
        # Get candidate name for display
        candidate_name = None
        for name, code in candidates_mapping.items():
            if code == candidate:
                candidate_name = name
                break
        
        if is_winner:
            print(f"Round {round_num}: {candidate} ({candidate_name}) WINS")
            print(f"  Active candidates when {candidate} won: {sorted(candidates_remaining)}")
        else:
            print(f"Round {round_num}: {candidate} ({candidate_name}) ELIMINATED")
            print(f"  Active candidates when {candidate} was eliminated: {sorted(candidates_remaining)}")
        
        # Remove candidate from remaining set
        candidates_remaining.remove(candidate)
        
        print(f"  Remaining after round {round_num}: {sorted(candidates_remaining)}")
        print()
    
    return active_at_elimination

def analyze_candidates_with_exhaust_gt_strategy(district_number, k=3):
    """
    Analyze candidates with exhaust > strategy and show their active candidates at elimination.
    
    Args:
        district_number (int): District number (1-4)
        k (int): Number of winners (default: 3)
    """
    # Get strategy and exhaust data
    strategy_data = create_portland_strategy_data()
    portland_df = create_portland_summary_dataframe()
    
    # Filter for the specific district
    district_row = portland_df[portland_df['district'] == district_number]
    if district_row.empty:
        print(f"No data found for District {district_number}")
        return
    
    district_row = district_row.iloc[0]
    strategy_dict = extract_strategy_dict(district_row['Strategies'])
    exhaust_dict = extract_exhaust_dict(district_row['exhaust_percents'])
    
    # Get active candidates at elimination
    active_at_elimination = get_active_candidates_at_elimination(district_number, k)
    
    # Get candidate mapping for names
    candidates_mapping = strategy_data[district_number]['candidates_mapping']
    reverse_mapping = {v: k for k, v in candidates_mapping.items()}
    
    print(f"\nDistrict {district_number} - Candidates with Exhaust > Strategy:")
    print("=" * 80)
    
    candidates_with_exhaust_gt_strategy = []
    
    # Find candidates that appear in both dictionaries and have exhaust > strategy
    common_candidates = set(strategy_dict.keys()) & set(exhaust_dict.keys())
    
    for letter in sorted(common_candidates):
        strategy_val = strategy_dict[letter]
        exhaust_val = exhaust_dict[letter]
        
        if exhaust_val > strategy_val:
            candidate_name = reverse_mapping.get(letter, f"Unknown ({letter})")
            active_candidates = active_at_elimination.get(letter, set())
            
            candidates_with_exhaust_gt_strategy.append({
                'letter': letter,
                'candidate_name': candidate_name,
                'strategy': strategy_val,
                'exhaust': exhaust_val,
                'diff': exhaust_val - strategy_val,
                'active_candidates': active_candidates
            })
            
            print(f"\nCandidate {letter} ({candidate_name}):")
            print(f"  Strategy: {strategy_val:.2f}%")
            print(f"  Exhaust: {exhaust_val:.2f}%")
            print(f"  Difference: {exhaust_val - strategy_val:.2f}%")
            print(f"  Active candidates at elimination: {sorted(active_candidates)}")
            
            # Show candidate names for active candidates
            active_names = []
            for active_letter in sorted(active_candidates):
                active_name = reverse_mapping.get(active_letter, f"Unknown ({active_letter})")
                active_names.append(f"{active_letter} ({active_name})")
            print(f"  Active candidates (with names): {', '.join(active_names)}")
    
    if not candidates_with_exhaust_gt_strategy:
        print("No candidates found with exhaust > strategy in this district.")
    
    return candidates_with_exhaust_gt_strategy

def test_active_candidates_analysis():
    """
    Test the active candidates analysis for all districts with strategy data.
    """
    print("Portland Multi-Winner STV: Active Candidates at Elimination Analysis")
    print("=" * 80)
    
    # Test for districts with strategy data
    districts_with_data = [1, 2, 4]  # Districts that have strategy data
    
    for district in districts_with_data:
        print(f"\n{'='*80}")
        print(f"ANALYZING DISTRICT {district}")
        print(f"{'='*80}")
        
        try:
            candidates_with_exhaust_gt_strategy = analyze_candidates_with_exhaust_gt_strategy(district)
            
            if candidates_with_exhaust_gt_strategy:
                print(f"\nSUMMARY for District {district}:")
                print("-" * 40)
                for candidate in candidates_with_exhaust_gt_strategy:
                    print(f"• {candidate['letter']} ({candidate['candidate_name']}): "
                          f"{len(candidate['active_candidates'])} active candidates at elimination")
            else:
                print(f"\nNo candidates with exhaust > strategy found in District {district}")
                
        except Exception as e:
            print(f"Error analyzing District {district}: {e}")
            import traceback
            traceback.print_exc()

def calculate_required_preference_percentage(strategy_pct, exhaust_pct):
    """
    Calculate the required preference percentage for a candidate to win,
    following the same methodology as single-winner RCV analysis.
    
    Args:
        strategy_pct (float): Strategy percentage (gap to win)
        exhaust_pct (float): Exhaust percentage
        
    Returns:
        tuple: (required_net_advantage, required_preference_pct)
    """
    if exhaust_pct <= 0:
        return 0, 50  # No exhausted ballots means no opportunity
    
    # Calculate required net advantage: how much advantage candidate needs among exhausted ballots
    required_net_advantage = (strategy_pct / exhaust_pct) * 100
    
    # Calculate required preference percentage: what % of exhausted ballots must prefer this candidate
    # Formula: if candidate needs X% net advantage, they need (50 + X/2)% preference
    required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
    
    return required_net_advantage, required_preference_pct

def analyze_required_preferences_for_candidates(district_number, k=3):
    """
    Analyze required preference percentages for candidates with exhaust > strategy
    and show which candidates they compete against.
    
    Args:
        district_number (int): District number (1-4)
        k (int): Number of winners (default: 3)
    """
    print(f"\nDistrict {district_number} - Required Preference Analysis:")
    print("=" * 80)
    
    # Get candidates with exhaust > strategy and their active candidates
    candidates_with_exhaust_gt_strategy = analyze_candidates_with_exhaust_gt_strategy(district_number, k)
    
    if not candidates_with_exhaust_gt_strategy:
        print("No candidates found with exhaust > strategy in this district.")
        return []
    
    # Get strategy and exhaust data
    strategy_data = create_portland_strategy_data()
    portland_df = create_portland_summary_dataframe()
    district_row = portland_df[portland_df['district'] == district_number].iloc[0]
    strategy_dict = extract_strategy_dict(district_row['Strategies'])
    exhaust_dict = extract_exhaust_dict(district_row['exhaust_percents'])
    
    # Get candidate mapping for names
    candidates_mapping = strategy_data[district_number]['candidates_mapping']
    reverse_mapping = {v: k for k, v in candidates_mapping.items()}
    
    results = []
    
    print(f"{'Candidate':<25} {'Letter':<6} {'Strategy %':<12} {'Exhaust %':<12} {'Net Adv %':<12} {'Req Pref %':<12} {'Active Candidates'}")
    print("-" * 120)
    
    for candidate_info in candidates_with_exhaust_gt_strategy:
        letter = candidate_info['letter']
        candidate_name = candidate_info['candidate_name']
        strategy_pct = candidate_info['strategy']
        exhaust_pct = candidate_info['exhaust']
        active_candidates = candidate_info['active_candidates']
        
        # Calculate required preference percentage
        required_net_advantage, required_preference_pct = calculate_required_preference_percentage(
            strategy_pct, exhaust_pct
        )
        
        # Format active candidates with names
        active_names = []
        for active_letter in sorted(active_candidates):
            active_name = reverse_mapping.get(active_letter, f"Unknown ({active_letter})")
            active_names.append(f"{active_letter}({active_name[:8]})")
        active_candidates_str = ", ".join(active_names)
        
        print(f"{candidate_name[:23]:<25} {letter:<6} {strategy_pct:<12.2f} {exhaust_pct:<12.2f} {required_net_advantage:<12.1f} {required_preference_pct:<12.1f} {active_candidates_str}")
        
        results.append({
            'district': district_number,
            'letter': letter,
            'candidate_name': candidate_name,
            'strategy_pct': strategy_pct,
            'exhaust_pct': exhaust_pct,
            'required_net_advantage': required_net_advantage,
            'required_preference_pct': required_preference_pct,
            'active_candidates': active_candidates,
            'active_candidates_count': len(active_candidates),
            'competes_against': [reverse_mapping.get(c, c) for c in active_candidates if c != letter]
        })
    
    print("-" * 120)
    print(f"\nSummary for District {district_number}:")
    print(f"• {len(results)} candidates have exhaust > strategy")
    print(f"• Required preference ranges from {min(r['required_preference_pct'] for r in results):.1f}% to {max(r['required_preference_pct'] for r in results):.1f}%")
    print(f"• Active candidates range from {min(r['active_candidates_count'] for r in results)} to {max(r['active_candidates_count'] for r in results)}")
    
    return results

def test_required_preference_analysis():
    """
    Test the required preference analysis for all districts with strategy data.
    """
    print("Portland Multi-Winner STV: Required Preference Analysis")
    print("=" * 80)
    print("This analysis shows what percentage of exhausted ballots each candidate")
    print("needs to prefer them over other active candidates to win the election.")
    print("=" * 80)
    
    # Test for districts with strategy data
    districts_with_data = [1, 2, 4]  # Districts that have strategy data
    all_results = []
    
    for district in districts_with_data:
        print(f"\n{'='*80}")
        print(f"ANALYZING DISTRICT {district}")
        print(f"{'='*80}")
        
        try:
            results = analyze_required_preferences_for_candidates(district)
            all_results.extend(results)
                
        except Exception as e:
            print(f"Error analyzing District {district}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    if all_results:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS ALL DISTRICTS")
        print(f"{'='*80}")
        
        print(f"Total candidates with exhaust > strategy: {len(all_results)}")
        
        # Create summary table (without probability columns since they're not calculated here)
        print(f"\n{'District':<8} {'Candidate':<20} {'Letter':<6} {'Req Pref %':<12} {'Strategy %':<12} {'Exhaust %':<12} {'Active':<8}")
        print("-" * 100)
        
        for result in sorted(all_results, key=lambda x: x['required_preference_pct']):
            print(f"{result['district']:<8} {result['candidate_name'][:18]:<20} {result['letter']:<6} "
                  f"{result['required_preference_pct']:<12.1f} {result['strategy_pct']:<12.2f} "
                  f"{result['exhaust_pct']:<12.2f} {result['active_candidates_count']:<8}")
        
        print("-" * 100)
        print(f"\nSummary Statistics:")
        print(f"• Required preference ranges from {min(r['required_preference_pct'] for r in all_results):.1f}% to {max(r['required_preference_pct'] for r in all_results):.1f}%")
        print(f"• Active candidates range from {min(r['active_candidates_count'] for r in all_results)} to {max(r['active_candidates_count'] for r in all_results)}")
        
        # Show most challenging cases (highest required preference)
        sorted_by_req_pref = sorted(all_results, key=lambda x: x['required_preference_pct'], reverse=True)
        print(f"\nMost Challenging Cases (highest required preference):")
        for i, result in enumerate(sorted_by_req_pref[:3]):
            print(f"{i+1}. District {result['district']}: {result['candidate_name']} ({result['letter']})")
            print(f"   Required preference: {result['required_preference_pct']:.1f}%")
            print(f"   Strategy: {result['strategy_pct']:.2f}%, Exhaust: {result['exhaust_pct']:.2f}%")
            print(f"   Competes against {result['active_candidates_count']-1} other candidates")
        
        print(f"\nNote: For full probability analysis including Beta models and bootstrap simulations,")
        print(f"use the comprehensive_probability_analysis() function.")
    
    return all_results

def extract_exhausted_ballots_for_candidate(district_number, candidate_letter, k=3):
    """
    Extract exhausted ballots and complete ballot data for a specific candidate's analysis.
    
    Args:
        district_number (int): District number (1-4)
        candidate_letter (str): Letter of the candidate to analyze
        k (int): Number of winners (default: 3)
        
    Returns:
        dict: Contains ballot_counts, exhausted_ballots, active_candidates, candidates_mapping
    """
    # Get district data
    df = district_data[district_number]['df']
    candidates_mapping = district_data[district_number]['candidates_mapping']
    
    # Convert ballot data to the format needed for STV analysis
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
    candidates = list(candidates_mapping.values())
    
    # Calculate quota (Droop quota)
    total_votes = sum(ballot_counts.values())
    quota = (total_votes // (k + 1)) + 1
    
    # Get the social choice order and active candidates at elimination
    active_at_elimination = get_active_candidates_at_elimination(district_number, k)
    active_candidates = active_at_elimination.get(candidate_letter, set())
    
    # Extract exhausted ballots (ballots that don't rank any active candidate)
    exhausted_ballots = {}
    for ballot, count in ballot_counts.items():
        if ballot and not any(c in active_candidates for c in ballot):
            exhausted_ballots[ballot] = count
    
    return {
        'ballot_counts': ballot_counts,
        'exhausted_ballots': exhausted_ballots,
        'active_candidates': active_candidates,
        'candidates_mapping': candidates_mapping,
        'total_votes': total_votes,
        'quota': quota
    }

def analyze_preference_patterns(ballot_counts, exhausted_ballots, candidate_letter, active_candidates):
    """
    Analyze preference patterns for candidate vs all other active candidates,
    categorized by first preference (similar to single-winner RCV analysis).
    
    Args:
        ballot_counts (dict): All ballot counts
        exhausted_ballots (dict): Exhausted ballot counts  
        candidate_letter (str): The candidate we're analyzing
        active_candidates (set): Set of all active candidates at elimination
        
    Returns:
        dict: Preference analysis results
    """
    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Find non-exhausted ballots that rank at least one active candidate
    # and categorize by first preference
    complete_by_first_pref = {}
    
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
            
        # Check if ballot ranks at least one active candidate
        ballot_active_candidates = [c for c in ballot if c in active_candidates]
        
        if ballot_active_candidates:  # Ballot ranks at least one active candidate
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {
                    'candidate_first': 0,  # candidate_letter ranked highest among active
                    'others_first': 0,     # other active candidates ranked highest
                    'total': 0
                }
            
            complete_by_first_pref[first_pref]['total'] += count
            
            # Determine if candidate_letter or others are ranked highest among active candidates
            if candidate_letter in ballot_active_candidates:
                # Find position of candidate_letter among active candidates in this ballot
                candidate_pos = ballot.index(candidate_letter)
                other_active_positions = [ballot.index(c) for c in ballot_active_candidates if c != candidate_letter]
                
                if not other_active_positions or candidate_pos < min(other_active_positions):
                    # candidate_letter is ranked highest among active candidates
                    complete_by_first_pref[first_pref]['candidate_first'] += count
                else:
                    # Some other active candidate is ranked higher
                    complete_by_first_pref[first_pref]['others_first'] += count
            else:
                # candidate_letter not ranked, so others win by default
                complete_by_first_pref[first_pref]['others_first'] += count
    
    return {
        'exhausted_by_first_pref': exh_by_first_pref,
        'complete_by_first_pref': complete_by_first_pref,
        'total_exhausted': sum(exhausted_ballots.values())
    }

def beta_probability_multi_winner(required_preference_pct, gap_to_win_pct):
    """
    Calculate probability using Beta distribution for multi-winner STV.
    Uses same parameters as single-winner analysis: α + β = 100 (percentage scale).
    Maps: a = parameter for "others" (like B in single-winner), b = parameter for "candidate" (like A in single-winner)
    """
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))
    
    # Convert from percentage to proportion for Beta CDF
    required_proportion = required_preference_pct / 100
    
    # Use same beta parameters as single-winner analysis (percentage scale)
    base_param = 50.0
    max_shift = min(gap_to_win_pct * 0.5, 40)
    a = max(base_param - max_shift, 10.0)  # Parameter for others (like B/challenger in single-winner)
    b = max(base_param + max_shift, 10.0)  # Parameter for candidate (like A/leader in single-winner)
    
    # Calculate probability
    probability = 1 - stats.beta.cdf(required_proportion, a, b)
    
    print(f"[Gap-Based Beta] a={a:.2f} (others), b={b:.2f} (candidate), α+β={a+b:.2f}, " +
          f"required: {required_preference_pct:.2f}%, probability: {probability:.4f}")
    
    return probability

def category_bootstrap_multi_winner(preference_analysis, candidate_letter, active_candidates, 
                                  required_preference_pct, gap_to_win_pct, n_bootstrap=1000):
    """
    Category-based bootstrap for multi-winner STV, following single-winner logic exactly.
    Properly handles ballot counts as weights in sampling.
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']
    
    if total_exhausted == 0:
        return 0.5, (0.5, 0.5)
    
    # Calculate votes needed for candidate to win (net advantage needed)
    votes_needed = int((required_preference_pct - 50) * total_exhausted / 100)
    
    print(f"\n[Category Bootstrap] Total exhausted: {total_exhausted}, " +
          f"Required preference: {required_preference_pct:.2f}%, Net votes needed: {votes_needed}")
    
    # Print category distributions for debugging
    print("Exhausted ballots by first preference:")
    for first_pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {first_pref}: {count} ({100*count/total_exhausted:.2f}%)")
    
    print("Complete ballots preference patterns by first preference:")
    for first_pref, data in sorted(complete_by_first_pref.items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] > 0:
            candidate_first_pct = 100 * data['candidate_first'] / data['total']
            print(f"  {first_pref}: Total={data['total']}, {candidate_letter} first={data['candidate_first']} ({candidate_first_pct:.2f}%), " +
                  f"Others first={data['others_first']} ({100-candidate_first_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    candidate_win_counts = 0
    
    for i in range(n_bootstrap):
        net_votes_for_candidate = 0
        candidate_first_completions = 0
        others_first_completions = 0
        
        # Process each category of exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Use category-specific data
                category_data = complete_by_first_pref[first_pref]
                prob_candidate_first = category_data['candidate_first'] / category_data['total']
                
                # Sample candidate vs others preferences for this category (using ballot count as weight)
                candidate_completions = np.random.binomial(count, prob_candidate_first)
                others_completions = count - candidate_completions
                
                candidate_first_completions += candidate_completions
                others_first_completions += others_completions
                net_votes_for_candidate += (candidate_completions - others_completions)
            else:
                # Use overall distribution if no category data
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob = total_candidate_first / total_complete
                    
                    candidate_completions = np.random.binomial(count, overall_prob)
                    others_completions = count - candidate_completions
                    
                    candidate_first_completions += candidate_completions
                    others_first_completions += others_completions
                    net_votes_for_candidate += (candidate_completions - others_completions)
        
        # Check if candidate wins
        candidate_wins = net_votes_for_candidate >= votes_needed
        bootstrap_results.append(candidate_wins)
        if candidate_wins:
            candidate_win_counts += 1
        
        # Debug first few iterations
        if i < 3:
            total_completions = candidate_first_completions + others_first_completions
            candidate_pct = 100 * candidate_first_completions / total_completions if total_completions > 0 else 0
            print(f"  Iteration {i}: {candidate_letter} first={candidate_first_completions}, Others first={others_first_completions}, " +
                  f"{candidate_letter}%={candidate_pct:.2f}%, Net={net_votes_for_candidate}, Wins={candidate_wins}")
    
    # Calculate probability and confidence interval
    win_probability = candidate_win_counts / n_bootstrap
    se = np.sqrt((win_probability * (1 - win_probability)) / n_bootstrap)
    ci_lower = max(0, win_probability - 1.96 * se)
    ci_upper = min(1, win_probability + 1.96 * se)
    
    print(f"[Category Bootstrap] Wins: {candidate_win_counts}/{n_bootstrap}, " +
          f"probability: {win_probability:.4f} ({ci_lower:.4f}, {ci_upper:.4f})")
    
    return win_probability, (ci_lower, ci_upper)

def similarity_beta_multi_winner(preference_analysis, candidate_letter, active_candidates, 
                                required_preference_pct, gap_to_win_pct):
    """
    Similarity Beta model for multi-winner STV.
    Uses observed preference patterns to set Beta parameters directly (α + β = 100).
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']
    
    if total_exhausted == 0:
        return 0.5
    
    # Calculate expected completions based on observed patterns
    total_expected_candidate = 0
    total_expected_others = 0
    
    for first_pref, count in exh_by_first_pref.items():
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            category_data = complete_by_first_pref[first_pref]
            prob_candidate_first = category_data['candidate_first'] / category_data['total']
            
            expected_candidate = count * prob_candidate_first
            expected_others = count * (1 - prob_candidate_first)
            
            total_expected_candidate += expected_candidate
            total_expected_others += expected_others
        else:
            # Use overall distribution if no category data
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())
            
            if total_complete > 0:
                overall_prob = total_candidate_first / total_complete
                expected_candidate = count * overall_prob
                expected_others = count * (1 - overall_prob)
                
                total_expected_candidate += expected_candidate
                total_expected_others += expected_others
    
    # Calculate preference percentages from expected completions (0-100 scale)
    total_completions = total_expected_candidate + total_expected_others
    if total_completions > 0:
        candidate_pct = 100 * total_expected_candidate / total_completions
        others_pct = 100 * total_expected_others / total_completions
        
        # Use these percentages directly as Beta parameters (following single-winner direct_posterior_beta)
        alpha = candidate_pct  # Parameter for candidate (like B>A in single-winner)
        beta = others_pct      # Parameter for others (like A>B in single-winner)
        
        # Calculate probability (convert required_preference_pct to proportion for Beta CDF)
        required_proportion = required_preference_pct / 100
        probability = 1 - stats.beta.cdf(required_proportion, alpha, beta)
        
        print(f"[Similarity Beta] Expected {candidate_letter}: {total_expected_candidate:.2f} ({candidate_pct:.2f}%), " +
              f"Others: {total_expected_others:.2f} ({others_pct:.2f}%), " +
              f"α={alpha:.2f}, β={beta:.2f}, α+β={alpha+beta:.2f}, probability: {probability:.4f}")
    else:
        probability = 0.5
        print(f"[Similarity Beta] No completion data available, using neutral probability: {probability:.4f}")
    
    return probability

def prior_posterior_beta_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                    required_preference_pct, gap_to_win_pct):
    """
    Prior-Posterior Beta model for multi-winner STV.
    Combines prior beliefs (gap-based) with observed evidence, following single-winner logic exactly.
    """
    exh_by_first_pref = preference_analysis['exhausted_by_first_pref']
    complete_by_first_pref = preference_analysis['complete_by_first_pref']
    total_exhausted = preference_analysis['total_exhausted']
    
    if total_exhausted == 0:
        return 0.5
    
    # Prior parameters based on gap (percentage scale, α + β = 100)
    base_param = 50.0
    max_shift = min(gap_to_win_pct * 0.5, 40)
    a_prior = max(base_param - max_shift, 10.0)  # Parameter for others (like B in single-winner)
    b_prior = max(base_param + max_shift, 10.0)  # Parameter for candidate (like A in single-winner)
    
    # Calculate observed evidence (expected completions)
    total_expected_candidate = 0
    total_expected_others = 0
    
    for first_pref, count in exh_by_first_pref.items():
        if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
            category_data = complete_by_first_pref[first_pref]
            prob_candidate_first = category_data['candidate_first'] / category_data['total']
            
            expected_candidate = count * prob_candidate_first
            expected_others = count * (1 - prob_candidate_first)
            
            total_expected_candidate += expected_candidate
            total_expected_others += expected_others
        else:
            # Use overall distribution if no category data
            total_complete = sum(data['total'] for data in complete_by_first_pref.values())
            total_candidate_first = sum(data['candidate_first'] for data in complete_by_first_pref.values())
            
            if total_complete > 0:
                overall_prob = total_candidate_first / total_complete
                expected_candidate = count * overall_prob
                expected_others = count * (1 - overall_prob)
                
                total_expected_candidate += expected_candidate
                total_expected_others += expected_others
    
    # Convert observed evidence to percentages (0-100 scale)
    total_expected = total_expected_candidate + total_expected_others
    if total_expected > 0:
        candidate_over_others_pct = 100 * total_expected_candidate / total_expected  # Like B>A in single-winner
        others_over_candidate_pct = 100 * total_expected_others / total_expected     # Like A>B in single-winner
        
        # Weighted combination following single-winner prior_posterior_beta logic exactly:
        # a_prior (others parameter) combines with candidate_over_others_pct (observed candidate>others evidence)
        # b_prior (candidate parameter) combines with others_over_candidate_pct (observed others>candidate evidence)
        weight_prior = 1.0  # Weight for prior
        weight_data = 1.0   # Weight for observed data
        
        # Correct Bayesian updating (following single-winner exactly)
        a_post = (weight_prior * a_prior + weight_data * candidate_over_others_pct) / (weight_prior + weight_data)
        b_post = (weight_prior * b_prior + weight_data * others_over_candidate_pct) / (weight_prior + weight_data)
    else:
        # If no data, just use the prior
        a_post = a_prior
        b_post = b_prior
        candidate_over_others_pct = 0
        others_over_candidate_pct = 0
    
    # Calculate probability (convert required_preference_pct to proportion for Beta CDF)
    required_proportion = required_preference_pct / 100
    probability = 1 - stats.beta.cdf(required_proportion, a_post, b_post)
    
    print(f"[Prior-Posterior Beta] Prior: a={a_prior:.2f}, b={b_prior:.2f} (α+β={a_prior+b_prior:.2f}), " +
          f"Observed: {candidate_letter}>{candidate_over_others_pct:.2f}%, Others>{others_over_candidate_pct:.2f}%, " +
          f"Posterior: a={a_post:.2f}, b={b_post:.2f} (α+β={a_post+b_post:.2f}), " +
          f"probability: {probability:.4f}")
    
    return probability

def unconditional_bootstrap_multi_winner(ballot_counts, exhausted_ballots, candidate_letter, active_candidates,
                                        required_preference_pct, gap_to_win_pct, n_bootstrap=1000, max_rankings=6):
    """
    Unconditional Bootstrap model for multi-winner STV.
    Samples from ALL ballots that rank any active candidate, following single-winner logic exactly.
    Properly handles ballot counts as weights.
    """
    total_exhausted = sum(exhausted_ballots.values())
    if total_exhausted == 0:
        return 0.5, (0.5, 0.5)
    
    # Calculate votes needed for candidate to win
    votes_needed = int((required_preference_pct - 50) * total_exhausted / 100)
    
    # Find all ballots that rank any active candidate (weighted by count)
    relevant_ballots = []
    candidate_first_total = 0
    others_first_total = 0
    total_with_active = 0
    
    for ballot, count in ballot_counts.items():
        if ballot and any(c in active_candidates for c in ballot):
            # Add this ballot 'count' times to the sampling pool
            relevant_ballots.extend([ballot] * count)
            total_with_active += count
            
            # Determine if candidate or others are ranked highest among active candidates
            ballot_active_candidates = [c for c in ballot if c in active_candidates]
            
            if candidate_letter in ballot_active_candidates:
                # Find position of candidate among active candidates
                candidate_pos = ballot.index(candidate_letter)
                other_active_positions = [ballot.index(c) for c in ballot_active_candidates if c != candidate_letter]
                
                if not other_active_positions or candidate_pos < min(other_active_positions):
                    # Candidate ranked highest among active candidates
                    candidate_first_total += count
                else:
                    # Others ranked higher
                    others_first_total += count
            else:
                # Candidate not ranked, others win
                others_first_total += count
    
    if not relevant_ballots:
        return 0.5, (0.5, 0.5)
    
    # Calculate overall probability that candidate is preferred
    overall_prob_candidate_first = candidate_first_total / (candidate_first_total + others_first_total) if (candidate_first_total + others_first_total) > 0 else 0.5
    
    print(f"\n[Unconditional Bootstrap] Total exhausted: {total_exhausted}, " +
          f"Relevant ballots for sampling: {len(relevant_ballots)}, " +
          f"Overall {candidate_letter} first probability: {overall_prob_candidate_first:.4f}, " +
          f"Required preference: {required_preference_pct:.2f}%, Net votes needed: {votes_needed}")
    
    # Run bootstrap iterations
    candidate_win_counts = 0
    
    for i in range(n_bootstrap):
        # Sample using overall distribution (following single-winner unconditional bootstrap exactly)
        candidate_completions = np.random.binomial(total_exhausted, overall_prob_candidate_first)
        others_completions = total_exhausted - candidate_completions
        
        # Calculate net votes for candidate
        net_votes_for_candidate = candidate_completions - others_completions
        
        # Check if candidate wins
        if net_votes_for_candidate >= votes_needed:
            candidate_win_counts += 1
        
        # Debug first few iterations
        if i < 3:
            candidate_pct = 100 * candidate_completions / total_exhausted if total_exhausted > 0 else 0
            print(f"  Iteration {i}: {candidate_letter}={candidate_completions}, Others={others_completions}, " +
                  f"{candidate_letter}%={candidate_pct:.2f}%, Net={net_votes_for_candidate}, " +
                  f"Wins={net_votes_for_candidate >= votes_needed}")
    
    # Calculate probability and confidence interval
    win_probability = candidate_win_counts / n_bootstrap
    se = np.sqrt((win_probability * (1 - win_probability)) / n_bootstrap)
    ci_lower = max(0, win_probability - 1.96 * se)
    ci_upper = min(1, win_probability + 1.96 * se)
    
    print(f"[Unconditional Bootstrap] Wins: {candidate_win_counts}/{n_bootstrap}, " +
          f"probability: {win_probability:.4f} ({ci_lower:.4f}, {ci_upper:.4f})")
    
    return win_probability, (ci_lower, ci_upper)

def calculate_multi_winner_probabilities(district_number, candidate_letter, k=3, n_bootstrap=1000):
    """
    Calculate probabilities for a candidate to win using multiple models
    adapted for multi-winner STV.
    
    Args:
        district_number (int): District number
        candidate_letter (str): Candidate letter to analyze
        k (int): Number of winners
        n_bootstrap (int): Bootstrap iterations
        
    Returns:
        dict: Results from all probability models
    """
    print(f"\n{'='*80}")
    print(f"PROBABILITY ANALYSIS: District {district_number}, Candidate {candidate_letter}")
    print(f"{'='*80}")
    
    # Get required preference percentage
    results = analyze_required_preferences_for_candidates(district_number, k)
    candidate_result = next((r for r in results if r['letter'] == candidate_letter), None)
    
    if not candidate_result:
        print(f"Candidate {candidate_letter} not found or doesn't have exhaust > strategy")
        return None
    
    required_preference_pct = candidate_result['required_preference_pct']
    strategy_pct = candidate_result['strategy_pct']
    exhaust_pct = candidate_result['exhaust_pct']
    active_candidates = candidate_result['active_candidates']
    
    print(f"Required preference: {required_preference_pct:.2f}%")
    print(f"Strategy: {strategy_pct:.2f}%, Exhaust: {exhaust_pct:.2f}%")
    print(f"Active candidates: {sorted(active_candidates)}")
    
    # Extract ballot data
    ballot_data = extract_exhausted_ballots_for_candidate(district_number, candidate_letter, k)
    
    # Analyze preference patterns
    preference_analysis = analyze_preference_patterns(
        ballot_data['ballot_counts'],
        ballot_data['exhausted_ballots'], 
        candidate_letter,
        active_candidates
    )
    
    # Calculate probabilities using different models
    results = {
        'district': district_number,
        'candidate_letter': candidate_letter,
        'candidate_name': candidate_result['candidate_name'],
        'required_preference_pct': required_preference_pct,
        'strategy_pct': strategy_pct,
        'exhaust_pct': exhaust_pct,
        'active_candidates_count': len(active_candidates),
        'total_exhausted_ballots': preference_analysis['total_exhausted']
    }
    
    # 1. Beta Model
    print(f"\n--- BETA MODEL ---")
    beta_prob = beta_probability_multi_winner(required_preference_pct, strategy_pct)
    results['beta_probability'] = beta_prob
    print(f"Beta probability: {beta_prob:.4f}")
    
    # 2. Category Bootstrap
    print(f"\n--- CATEGORY BOOTSTRAP ---")
    bootstrap_prob, bootstrap_ci = category_bootstrap_multi_winner(
        preference_analysis, candidate_letter, active_candidates,
        required_preference_pct, strategy_pct, n_bootstrap
    )
    results['bootstrap_probability'] = bootstrap_prob
    results['bootstrap_ci_lower'] = bootstrap_ci[0]
    results['bootstrap_ci_upper'] = bootstrap_ci[1]
    
    # 3. Similarity Beta
    print(f"\n--- SIMILARITY BETA ---")
    similarity_prob = similarity_beta_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                                    required_preference_pct, strategy_pct)
    results['similarity_probability'] = similarity_prob
    print(f"Similarity Beta probability: {similarity_prob:.4f}")
    
    # 4. Prior-Posterior Beta
    print(f"\n--- PRIOR-POSTERIOR BETA ---")
    prior_posterior_prob = prior_posterior_beta_multi_winner(preference_analysis, candidate_letter, active_candidates,
                                                            required_preference_pct, strategy_pct)
    results['prior_posterior_probability'] = prior_posterior_prob
    print(f"Prior-Posterior Beta probability: {prior_posterior_prob:.4f}")
    
    # 5. Unconditional Bootstrap
    print(f"\n--- UNCONDITIONAL BOOTSTRAP ---")
    unconditional_prob, unconditional_ci = unconditional_bootstrap_multi_winner(
        ballot_data['ballot_counts'],
        ballot_data['exhausted_ballots'],
        candidate_letter,
        active_candidates,
        required_preference_pct,
        strategy_pct,
        n_bootstrap,
        max_rankings=6
    )
    results['unconditional_probability'] = unconditional_prob
    results['unconditional_ci_lower'] = unconditional_ci[0]
    results['unconditional_ci_upper'] = unconditional_ci[1]
    
    return results

def comprehensive_probability_analysis(districts_to_analyze=None, n_bootstrap=1000):
    """
    Run comprehensive probability analysis for all candidates with exhaust > strategy
    across specified districts.
    
    Args:
        districts_to_analyze (list): Districts to analyze (default: [1, 2, 4])
        n_bootstrap (int): Bootstrap iterations
        
    Returns:
        list: Results for all analyzed candidates
    """
    if districts_to_analyze is None:
        districts_to_analyze = [1, 2, 4]  # Districts with strategy data
    
    print("COMPREHENSIVE MULTI-WINNER STV PROBABILITY ANALYSIS")
    print("=" * 80)
    print("Analyzing candidates with exhaust > strategy using:")
    print("• Beta probability model (theoretical)")
    print("• Category-based bootstrap (empirical)")
    print("• Similarity Beta (observed patterns)")
    print("• Prior-Posterior Beta (gap-based + observed patterns)")
    print("• Unconditional Bootstrap (random completions)")
    print("=" * 80)
    
    all_results = []
    
    for district in districts_to_analyze:
        print(f"\n{'='*80}")
        print(f"DISTRICT {district} ANALYSIS")
        print(f"{'='*80}")
        
        try:
            # Get candidates with exhaust > strategy
            candidates_analysis = analyze_required_preferences_for_candidates(district)
            
            for candidate_info in candidates_analysis:
                candidate_letter = candidate_info['letter']
                
                # Calculate probabilities for this candidate
                prob_results = calculate_multi_winner_probabilities(
                    district, candidate_letter, k=3, n_bootstrap=n_bootstrap
                )
                
                if prob_results:
                    all_results.append(prob_results)
                    
        except Exception as e:
            print(f"Error analyzing District {district}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary analysis
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL CANDIDATES")
        print(f"{'='*80}")
        
        print(f"Total candidates analyzed: {len(all_results)}")
        
        # Create summary table
        print(f"\n{'District':<8} {'Candidate':<20} {'Letter':<6} {'Req Pref %':<12} {'Gap Beta':<10} {'Category':<10} {'Similar':<10} {'Prior-Post':<10} {'Uncondit':<10} {'Active':<8}")
        print("-" * 120)
        
        for result in sorted(all_results, key=lambda x: x['required_preference_pct']):
            print(f"{result['district']:<8} {result['candidate_name'][:18]:<20} {result['letter']:<6} "
                  f"{result['required_preference_pct']:<12.1f} {result['beta_probability']:<10.4f} "
                  f"{result['bootstrap_probability']:<10.4f} {result['similarity_probability']:<10.4f} "
                  f"{result['prior_posterior_probability']:<10.4f} {result['unconditional_probability']:<10.4f} "
                  f"{result['active_candidates_count']:<8}")
        
        # Analysis by probability ranges (using category bootstrap as reference)
        low_prob = [r for r in all_results if r['bootstrap_probability'] < 0.3]
        med_prob = [r for r in all_results if 0.3 <= r['bootstrap_probability'] < 0.7]
        high_prob = [r for r in all_results if r['bootstrap_probability'] >= 0.7]
        
        print(f"\nProbability Distribution (Category Bootstrap):")
        print(f"• Low probability (< 30%): {len(low_prob)} candidates")
        print(f"• Medium probability (30-70%): {len(med_prob)} candidates")
        print(f"• High probability (≥ 70%): {len(high_prob)} candidates")
        
        # Most promising opportunities (using highest average across all models)
        for result in all_results:
            avg_prob = np.mean([
                result['beta_probability'],
                result['bootstrap_probability'], 
                result['similarity_probability'],
                result['prior_posterior_probability'],
                result['unconditional_probability']
            ])
            result['average_probability'] = avg_prob
        
        sorted_by_avg_prob = sorted(all_results, key=lambda x: x['average_probability'], reverse=True)
        print(f"\nMost Promising Opportunities (by average probability):")
        for i, result in enumerate(sorted_by_avg_prob[:3]):
            print(f"{i+1}. District {result['district']}: {result['candidate_name']} ({result['letter']})")
            print(f"   Average probability: {result['average_probability']:.1%}")
            print(f"   Gap Beta: {result['beta_probability']:.1%}, Category: {result['bootstrap_probability']:.1%}, " +
                  f"Similarity: {result['similarity_probability']:.1%}")
            print(f"   Prior-Posterior: {result['prior_posterior_probability']:.1%}, Unconditional: {result['unconditional_probability']:.1%}")
            print(f"   Required preference: {result['required_preference_pct']:.1f}%")
            print(f"   Competes against {result['active_candidates_count']-1} other candidates")
    
    return all_results

if __name__ == "__main__":
    # Run the comprehensive probability analysis
    comprehensive_probability_analysis(districts_to_analyze=[1, 2, 4], n_bootstrap=500)


