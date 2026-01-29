"""
NYC 2025 Fast Margins Analysis

Computes margin of victory directly from IRV simulation (final round vote difference).
Much faster than full strategy computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
from string import ascii_uppercase

from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils import helpers as utils


def compute_irv_margin(ballot_counts, candidates, k=1):
    """
    Compute the margin of victory from IRV elimination.
    Returns the vote difference needed for runner-up to win.
    """
    total_votes = sum(ballot_counts.values())
    Q = total_votes / 2 + 1  # Majority threshold
    
    # Run full IRV simulation
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, _ = utils.return_main_sub(rt)
    
    if len(results) < 2:
        return None, results
    
    # Get final round ballot state (after all eliminations except winner vs runner-up)
    # The collection stores ballot states at each round
    if len(collection) >= 2:
        # Get the ballot state when only 2 candidates remain
        final_round_idx = len(candidates) - 2
        if final_round_idx < len(collection):
            final_ballots = collection[final_round_idx][0]
            final_votes = utils.get_new_dict(final_ballots)
            
            winner = results[0]
            runner_up = results[1]
            
            winner_votes = final_votes.get(winner, 0)
            runner_up_votes = final_votes.get(runner_up, 0)
            
            # Margin is votes runner-up needs to overtake winner
            margin_votes = winner_votes - runner_up_votes + 1
            margin_pct = margin_votes / total_votes * 100
            
            return margin_pct, results
    
    # Fallback: use first-choice difference
    first_choice = utils.get_new_dict(ballot_counts)
    winner = results[0]
    runner_up = results[1]
    
    margin_votes = first_choice.get(winner, 0) - first_choice.get(runner_up, 0) + 1
    margin_pct = margin_votes / total_votes * 100
    
    return margin_pct, results


def analyze_fast():
    """Analyze all NYC 2025 DEM elections with fast margin computation."""
    
    project_root = Path(__file__).parent.parent
    nyc_2025_folder = project_root / "case_studies" / "nyc" / "nyc 2025 files"
    results_dir = project_root / "results" / "tables"
    
    k = 1
    results_list = []
    
    csv_files = [f for f in os.listdir(nyc_2025_folder) if f.endswith('.csv') and 'DEM' in f]
    total_files = len(csv_files)
    
    print(f"Analyzing {total_files} DEM elections (fast margin computation)")
    print("=" * 60)
    
    for idx, input_file in enumerate(csv_files, 1):
        full_path = nyc_2025_folder / input_file
        
        try:
            df, processed_file = process_single_file(full_path)
            candidates_mapping = create_candidate_mapping(processed_file)
            ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
            candidates = list(candidates_mapping.values())
            
            total_votes = sum(ballot_counts.values())
            num_candidates = len(candidates)
            
            if total_votes < 100 or num_candidates < 2:
                continue
            
            # Compute margin directly
            margin_pct, results = compute_irv_margin(ballot_counts, candidates, k)
            
            # Remap for display
            reverse_mapping = {code: name for name, code in candidates_mapping.items()}
            ordered_names = [reverse_mapping[code] for code in results]
            final_mapping = {name: ascii_uppercase[i] for i, name in enumerate(ordered_names)}
            
            # Recompute ballot counts with new mapping for exhaustion
            ballot_counts_remapped = case_study_helpers.get_ballot_counts_df(final_mapping, df)
            candidates_remapped = list(final_mapping.values())
            
            # Compute exhaustion
            exhausted_list, exhausted_dict = IRV_ballot_exhaust(candidates_remapped, ballot_counts_remapped)
            exhausted_pct = {key: round(value / total_votes * 100, 3) 
                           for key, value in exhausted_dict.items()}
            
            print(f"[{idx}/{total_files}] {input_file[:45]:45} | Margin: {margin_pct:.2f}%")
            
            # Create strategy-like structure for compatibility
            strats_pct = {'A': [0.0, []]}  # Winner
            if margin_pct is not None:
                strats_pct['B'] = [margin_pct, {'B': margin_pct}]  # Runner-up
            
            results_list.append({
                "file_name": input_file,
                "total_votes": total_votes,
                "num_candidates": num_candidates,
                "overall_winning_order": candidates_remapped,
                "Strategies": strats_pct,
                "exhaust_percents": exhausted_pct,
                "margin_of_victory": margin_pct
            })
            
        except Exception as e:
            print(f"[{idx}/{total_files}] {input_file[:45]:45} | ERROR: {e}")
            continue
    
    # Save
    df_results = pd.DataFrame(results_list)
    output_path = results_dir / "summary_table_nyc_2025_margins.xlsx"
    df_results.to_excel(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    margins = df_results['margin_of_victory'].dropna()
    print(f"\nMargins computed: {len(margins)}/{len(df_results)}")
    print(f"Mean margin: {margins.mean():.2f}%")
    print(f"Median margin: {margins.median():.2f}%")
    print(f"Min margin: {margins.min():.2f}%")
    print(f"Max margin: {margins.max():.2f}%")
    
    return df_results


if __name__ == "__main__":
    analyze_fast()
