"""
NYC 2025 RCV Elections Analysis Script

Analyzes all NYC 2025 election files and produces a summary Excel table
with victory gaps, strategies, and exhaustion data.

Based on the same methodology used for NYC 2021 analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
from string import ascii_uppercase

from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils import helpers as utils


def analyze_nyc_2025():
    """Analyze all NYC 2025 election files."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    nyc_2025_folder = project_root / "case_studies" / "nyc" / "nyc 2025 files"
    results_dir = project_root / "results" / "tables"
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis parameters
    k = 1  # Single winner elections
    budget_percent = 40  # Start high, auto-reduces for big elections
    
    results_list = []
    file_count = 0
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(nyc_2025_folder) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    print(f"Found {total_files} CSV files to analyze")
    print("=" * 60)
    
    for input_file in csv_files:
        file_count += 1
        full_path = nyc_2025_folder / input_file
        
        try:
            print(f"\n[{file_count}/{total_files}] Processing: {input_file}")
            
            # Step 1: Clean and process the file
            df, processed_file = process_single_file(full_path)
            
            # Step 2: Create initial candidate mapping and ballot counts
            candidates_mapping = create_candidate_mapping(processed_file)
            ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
            candidates = list(candidates_mapping.values())
            
            total_votes = sum(ballot_counts.values())
            num_candidates = len(candidates)
            
            print(f"   Candidates: {num_candidates}, Total votes: {total_votes}")
            
            # Skip if too few votes or candidates
            if total_votes < 100 or num_candidates < 2:
                print(f"   Skipping: too few votes ({total_votes}) or candidates ({num_candidates})")
                continue
            
            # Step 3: Run STV to determine elimination order
            Q = round(total_votes / (k + 1) + 1, 3)
            if k == 1:
                Q = Q * (k + 1)  # For IRV, quota is majority
            
            rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
            results, subresults = utils.return_main_sub(rt)
            
            # Step 4: Remap candidates by elimination order (A=winner, B=runner-up, etc.)
            reverse_mapping = {code: name for name, code in candidates_mapping.items()}
            ordered_candidate_names = [reverse_mapping[code] for code in results]
            final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
            
            # Rebuild ballot counts with the new mapping
            ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
            candidates = list(final_mapping.values())
            
            # Step 5: Compute ballot exhaustion
            exhausted_list, exhausted_dict = IRV_ballot_exhaust(candidates, ballot_counts)
            exhausted_pct = {key: round(value / total_votes * 100, 3) 
                           for key, value in exhausted_dict.items()}
            
            # Step 6: Run full analysis with strategy computation
            file_result = case_study_helpers.process_ballot_counts_post_elim_no_print(
                ballot_counts=ballot_counts,
                k=k,
                candidates=candidates,
                elim_cands=[],
                check_strats=True,
                budget_percent=budget_percent,
                check_removal_here=(num_candidates > 9),
                keep_at_least=min(9, num_candidates),
                rigorous_check=True
            )
            
            # Add metadata
            file_result["file_name"] = input_file
            file_result["exhaust_percents"] = exhausted_pct
            file_result["budget_percent"] = budget_percent
            file_result["candidate_mapping"] = str(final_mapping)
            
            results_list.append(file_result)
            
            print(f"   Winner: {ordered_candidate_names[0]}")
            print(f"   Strategies computed: {len(file_result.get('Strategies', {}))}")
            
        except Exception as e:
            print(f"   ERROR processing {input_file}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Processed {len(results_list)} elections successfully")
    
    # Create DataFrame and save
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Reorder columns for clarity
        column_order = [
            "file_name", "total_votes", "num_candidates", "quota",
            "overall_winning_order", "Strategies", "exhaust_percents",
            "candidates_removed", "candidates_retained",
            "initial_zeros", "def_losing", "budget_percent", "candidate_mapping"
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df_results.columns]
        df_results = df_results[available_columns]
        
        # Save to Excel
        output_path = results_dir / "summary_table_nyc_2025.xlsx"
        df_results.to_excel(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total elections analyzed: {len(df_results)}")
        print(f"DEM elections: {len(df_results[df_results['file_name'].str.contains('DEM', na=False)])}")
        print(f"REP elections: {len(df_results[df_results['file_name'].str.contains('REP', na=False)])}")
        
        return df_results
    else:
        print("No results to save!")
        return None


if __name__ == "__main__":
    analyze_nyc_2025()
