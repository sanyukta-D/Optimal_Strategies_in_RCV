import pandas as pd
import os
from rcv_strategies.core.stv_irv import STV_ballot_exhaust, STV_optimal_result_simple
from case_studies.portland.load_district_data import district_data
from rcv_strategies.utils import case_study_helpers

def print_exhaustion_analysis(district_number, k=3):
    """
    Print detailed analysis of ballot exhaustion for a district.
    """
    print(f"\nAnalyzing District {district_number} STV Election")
    print("=" * 50)
    
    # Get district data first to print summary
    df = district_data[district_number]['df']
    candidates_mapping = district_data[district_number]['candidates_mapping']
    candidates = list(candidates_mapping.values())
    
    print(f"\nSummary:")
    print(f"Total candidates: {len(candidates)}")
    print(f"Number of winners (k): {k}")
    print(f"Total ballots: {len(df)}")
    
    # Convert ballot data using the proper helper function
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
    
    # Calculate quota (Droop quota)
    total_votes = sum(ballot_counts.values())
    quota = (total_votes // (k + 1)) + 1
    
    # Get the social choice order first
    event_log, result_dict, round_history = STV_optimal_result_simple(candidates, ballot_counts, k, quota)
    
    # Initialize tracking variables
    candidates_remaining = set(candidates)
    used_ballots = {}  # Dictionary to track used ballots and their counts
    exhausted_ballots = {}  # Dictionary to track exhausted ballots and their counts
    round_data = []
    current_ballots = ballot_counts.copy()
    
    # Process each round based on the event log
    for round_num, (candidate, is_winner) in enumerate(event_log, 1):
        if is_winner:
            # This is a winning round
            candidates_remaining.remove(candidate)
            
            # Move ballots ranking this winner first to used_ballots
            # This includes both original ballots and transferred ballots
            for ballot, count in current_ballots.items():
                if ballot.startswith(candidate):
                    used_ballots[ballot] = used_ballots.get(ballot, 0) + count
        else:
            # This is an elimination round
            candidates_remaining.remove(candidate)
        
        # Count exhausted ballots (excluding used ballots)
        new_exhausted = {}
        for ballot, count in current_ballots.items():
            if ballot not in used_ballots:  # Only count ballots that haven't been used
                # Check if ballot has any remaining active candidates
                if not any(c in candidates_remaining for c in ballot):
                    new_exhausted[ballot] = count
        
        # Update exhausted ballots
        for ballot, count in new_exhausted.items():
            exhausted_ballots[ballot] = exhausted_ballots.get(ballot, 0) + count
        
        # Get full name of candidate
        candidate_name = None
        for name, code in candidates_mapping.items():
            if code == candidate:
                candidate_name = f"{candidate} ({name})"
                break
        
        # Calculate percentages
        used_percentage = sum(used_ballots.values()) / total_votes * 100
        exhausted_percentage = sum(exhausted_ballots.values()) / total_votes * 100
        
        # Store round data
        round_data.append({
            'Round': round_num,
            'Exhausted %': exhausted_percentage,
            'Candidate': candidate_name,
            'Type': 'Winner' if is_winner else 'Eliminated',
            'Used Ballots %': used_percentage
        })
        
        # Update current_ballots for next round
        if round_num < len(round_history):
            current_ballots = round_history[round_num][0]
    
    # Print round-by-round analysis
    print("\nRound-by-round analysis:")
    print("=" * 80)
    # for data in round_data:
    #     print(f"\nRound {data['Round']}:")
    #     print(f"Exhausted ballots: {data['Exhausted %']:.2f}%")
    #     print(f"Used ballots: {data['Used Ballots %']:.2f}%")
    #     print(f"{data['Type']}: {data['Candidate']}")
    
    # Print total exhaustion
    print(f"\nTotal ballots exhausted: {sum(exhausted_ballots.values())/total_votes*100:.2f}%")
    print(f"Total ballots used: {sum(used_ballots.values())/total_votes*100:.2f}%")
    
    # Print table summary
    print("\nTable Summary:")
    print("=" * 120)
    print(f"{'Round':<6} {'Exhausted %':<12} {'Used %':<12} {'Type':<10} {'Candidate':<100}")
    print("-" * 120)
    for data in round_data:
        print(f"{data['Round']:<6} {data['Exhausted %']:<12.2f} {data['Used Ballots %']:<12.2f} {data['Type']:<10} {data['Candidate']:<100}")
    print("=" * 120)

def create_district_summary():
    """
    Create a summary table for all districts showing key statistics.
    """
    summary_data = []
    
    for district in range(1, 5):
        # Get district data
        df = district_data[district]['df']
        candidates_mapping = district_data[district]['candidates_mapping']
        candidates = list(candidates_mapping.values())
        
        # Create reverse mapping for candidate names
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        # Convert ballot data using the proper helper function
        ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
        
        # Calculate quota (Droop quota)
        total_votes = sum(ballot_counts.values())
        quota = (total_votes // (3 + 1)) + 1  # k=3 for all districts
        
        # Calculate exhaustion using the correct STV_ballot_exhaust function - it returns 3 values
        exhausted_ballots_list, exhausted_ballots_dict, winners = STV_ballot_exhaust(
            cands=candidates,
            ballot_counts=ballot_counts,
            k=3,
            Q=quota
        )
        
        # Calculate statistics
        total_exhausted = sum(exhausted_ballots_list)
        max_exhaustion = max(exhausted_ballots_list) if exhausted_ballots_list else 0
        max_exhaustion_round = exhausted_ballots_list.index(max_exhaustion) + 1 if exhausted_ballots_list else 0
        
        # Get names of winners
        winner_names = [reverse_mapping[c] for c in winners]
        
        # Calculate used ballots (ballots that ranked any winner first)
        used_ballots_count = 0
        for ballot, count in ballot_counts.items():
            if ballot and any(ballot.startswith(winner) for winner in winners):
                used_ballots_count += count
        
        summary_data.append({
            'District': district,
            'Candidates': len(candidates),
            'Total Ballots': total_votes,
            'Used Ballots': used_ballots_count,
            'Total Exhaustion %': (total_exhausted/total_votes)*100,
            'Number of Rounds': len(exhausted_ballots_list),
            'Max Exhaustion %': (max_exhaustion/total_votes)*100,
            'Max Exhaustion Round': max_exhaustion_round,
            'Winners': ', '.join(winner_names)
        })
    
    # Create DataFrame and format
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(2)
    
    # Print formatted table
    print("\nPortland City Council Districts Summary")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("=" * 120)
    
    return summary_df

def main():
    """Analyze exhaustion for all four districts."""
    # Print summary table first
    create_district_summary()
    
    # Then print detailed analysis for each district
    for district in range(1, 5):
        print_exhaustion_analysis(district)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 