import os
import pandas as pd
from Case_Studies.NewYork_City_RCV.convert_data import process_single_file, create_candidate_mapping
from STVandIRV_results import STV_optimal_result_simple
import case_study_helpers
import utils
from string import ascii_uppercase

def debug_alaska_at_budget(budget_percent, rigorous_check=True):
    """
    Debug Alaska election at specific budget percentage
    """
    alaska_folder = "Case_Studies/Alaska_RCV/alaska_files"
    presidential_file = "Alaska_11052024_President.csv"
    file_path = os.path.join(alaska_folder, presidential_file)
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING Alaska Presidential at {budget_percent}% budget")
    print(f"Rigorous check: {rigorous_check}")
    print(f"{'='*60}")
    
    try:
        df, processed_file = process_single_file(file_path)
        k = 1
        
        # Process the election data
        candidates_mapping = create_candidate_mapping(processed_file)
        ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
        candidates = list(candidates_mapping.values())
        
        print(f"Total candidates: {len(candidates)}")
        print(f"Total votes: {sum(ballot_counts.values())}")
        
        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
        results, subresults = utils.return_main_sub(rt)
        
        # Create final mapping 
        reverse_mapping = {code: name for name, code in candidates_mapping.items()}
        ordered_candidate_names = [reverse_mapping[code] for code in results]
        final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
        ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
        candidates = list(final_mapping.values())
        
        print(f"Reordered candidates: {candidates}")
        print(f"Winning order: {results}")
        
        # Use standard parameters
        elim_cands = []
        keep_at_least = 9
        
        # Test using the detailed function with printing
        print(f"\nCalling case_study_helpers.process_ballot_counts_post_elim with:")
        print(f"  k = {k}")
        print(f"  candidates = {candidates}")
        print(f"  elim_cands = {elim_cands}")
        print(f"  budget_percent = {budget_percent}")
        print(f"  keep_at_least = {keep_at_least}")
        print(f"  rigorous_check = {rigorous_check}")
        
        # Use the version with printing to see what's happening
        case_study_helpers.process_ballot_counts_post_elim(
            ballot_counts,
            k,
            candidates,
            elim_cands,
            check_strats=False,
            budget_percent=budget_percent,
            check_removal_here=True,
            keep_at_least=keep_at_least,
            rigorous_check=rigorous_check
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Test specific budget percentages"""
    
    print("Testing Alaska Presidential Election at specific budget percentages...")
    
    # Test the boundary cases
    debug_alaska_at_budget(40.8, rigorous_check=True)
    debug_alaska_at_budget(40.8, rigorous_check=False)
    
    debug_alaska_at_budget(40.0, rigorous_check=True)
    debug_alaska_at_budget(40.0, rigorous_check=False)
    
    debug_alaska_at_budget(41.0, rigorous_check=True)
    debug_alaska_at_budget(41.0, rigorous_check=False)

if __name__ == "__main__":
    main() 