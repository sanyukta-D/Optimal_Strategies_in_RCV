"""
NYC 2025 Large Elections Analysis Script

Uses auto-search (divide and conquer) approach to find tractable thresholds
for large elections where 40% budget doesn't work.

Targets: Mayor and D41
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from string import ascii_uppercase

from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils import helpers as utils


def analyze_with_auto_threshold(full_path, input_file, k=1, max_budget=40):
    """
    Analyze a single election with auto-threshold search.
    """
    print(f"\nProcessing: {input_file}")
    print("=" * 60)
    
    # Step 1: Clean and process the file
    df, processed_file = process_single_file(full_path)
    
    # Step 2: Create initial candidate mapping and ballot counts
    candidates_mapping = create_candidate_mapping(processed_file)
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
    candidates = list(candidates_mapping.values())
    
    total_votes = sum(ballot_counts.values())
    num_candidates = len(candidates)
    
    print(f"Candidates: {num_candidates}, Total votes: {total_votes}")
    
    # Step 3: Run STV to determine elimination order
    Q = round(total_votes / (k + 1) + 1, 3)
    if k == 1:
        Q = Q * (k + 1)
    
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, subresults = utils.return_main_sub(rt)
    
    # Step 4: Remap candidates by elimination order
    reverse_mapping = {code: name for name, code in candidates_mapping.items()}
    ordered_candidate_names = [reverse_mapping[code] for code in results]
    final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
    
    ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
    candidates = list(final_mapping.values())
    
    print(f"Winner: {ordered_candidate_names[0]}")
    print(f"Elimination order: {' -> '.join(ordered_candidate_names[:5])}...")
    
    # Step 5: Compute exhaustion
    exhausted_list, exhausted_dict = IRV_ballot_exhaust(candidates, ballot_counts)
    exhausted_pct = {key: round(value / total_votes * 100, 3) 
                   for key, value in exhausted_dict.items()}
    
    # Step 6: Auto-search for tractable threshold
    print(f"\nSearching for tractable threshold (starting from {max_budget}%)...")
    
    def try_budget(test_budget):
        """Helper to test if a budget works."""
        result = case_study_helpers.process_ballot_counts_post_elim_no_print(
            ballot_counts=ballot_counts,
            k=k,
            candidates=candidates,
            elim_cands=[],
            check_strats=True,
            budget_percent=test_budget,
            check_removal_here=True,
            keep_at_least=min(9, num_candidates),
            rigorous_check=True
        )
        strategies = result.get("Strategies", {})
        if strategies:
            return result
        return None
    
    # Phase 1: Coarse search
    coarse_budgets = [b for b in [30, 25, 20, 15, 10, 7.5, 5] if b < max_budget]
    fine_budgets = [4, 3, 2.5, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.2]
    all_budgets = coarse_budgets + fine_budgets
    
    working_budget = None
    working_result = None
    
    for test_budget in all_budgets:
        print(f"  Trying {test_budget}%...", end=" ")
        result = try_budget(test_budget)
        if result:
            print(f"SUCCESS! ({len(result.get('Strategies', {}))} strategies)")
            working_budget = test_budget
            working_result = result
            break
        else:
            print("too large")
    
    if working_budget is None:
        print("  Could not find tractable threshold!")
        return None
    
    # Phase 2: Binary search to find highest working budget
    print(f"\nRefining threshold (binary search between {working_budget}% and higher)...")
    
    try:
        idx = all_budgets.index(working_budget)
        upper = all_budgets[idx - 1] if idx > 0 else max_budget
    except ValueError:
        upper = max_budget
    lower = working_budget
    
    precision = 0.1 if working_budget < 5 else 0.5
    
    while upper - lower > precision:
        mid = round((upper + lower) / 2, 2)
        print(f"  Trying {mid}%...", end=" ")
        result = try_budget(mid)
        if result:
            print(f"works")
            lower = mid
            working_budget = mid
            working_result = result
        else:
            print("too large")
            upper = mid
    
    print(f"\n*** Found optimal threshold: {working_budget}% ***")
    print(f"Strategies computed: {len(working_result.get('Strategies', {}))}")
    
    # Add metadata
    working_result["file_name"] = input_file
    working_result["exhaust_percents"] = exhausted_pct
    working_result["budget_percent"] = working_budget  # Actual threshold used
    working_result["candidate_mapping"] = str(final_mapping)
    working_result["original_budget_requested"] = max_budget
    
    return working_result


def main():
    """Analyze the two large NYC 2025 elections."""
    
    project_root = Path(__file__).parent.parent
    nyc_2025_folder = project_root / "case_studies" / "nyc" / "nyc 2025 files"
    results_dir = project_root / "results" / "tables"
    
    # The two large elections that need auto-threshold
    large_elections = [
        "NewYorkCity_20250624_DEMMayorCitywide.csv",
        "NewYorkCity_20250624_DEMCityCouncilD41.csv"
    ]
    
    results_list = []
    
    for input_file in large_elections:
        full_path = nyc_2025_folder / input_file
        
        if not full_path.exists():
            print(f"File not found: {input_file}")
            continue
        
        result = analyze_with_auto_threshold(full_path, input_file, k=1, max_budget=40)
        
        if result:
            results_list.append(result)
    
    # Load existing results and update
    existing_path = results_dir / "summary_table_nyc_2025.xlsx"
    
    if existing_path.exists() and results_list:
        print("\n" + "=" * 60)
        print("Updating existing results file...")
        
        df_existing = pd.read_excel(existing_path)
        df_new = pd.DataFrame(results_list)
        
        # Remove old entries for these files
        large_election_names = [r["file_name"] for r in results_list]
        df_existing = df_existing[~df_existing["file_name"].isin(large_election_names)]
        
        # Add new entries
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Sort by file name for consistency
        df_combined = df_combined.sort_values("file_name").reset_index(drop=True)
        
        # Save
        df_combined.to_excel(existing_path, index=False)
        print(f"Updated: {existing_path}")
        print(f"Total elections in file: {len(df_combined)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("LARGE ELECTIONS SUMMARY")
    print("=" * 60)
    for result in results_list:
        print(f"\n{result['file_name']}:")
        print(f"  Threshold used: {result['budget_percent']}%")
        print(f"  Candidates: {result['num_candidates']}")
        print(f"  Strategies: {len(result.get('Strategies', {}))}")


if __name__ == "__main__":
    main()
