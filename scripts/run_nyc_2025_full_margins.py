"""
NYC 2025 Full Margins Analysis

Re-runs analysis with 100% budget to ensure ALL victory margins are computed,
even for blowout elections.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
from string import ascii_uppercase

from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.core.strategy import reach_any_winners_campaign
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils import helpers as utils


def analyze_with_full_margins():
    """Analyze all NYC 2025 DEM elections with 100% budget to get ALL margins."""
    
    project_root = Path(__file__).parent.parent
    nyc_2025_folder = project_root / "case_studies" / "nyc" / "nyc 2025 files"
    results_dir = project_root / "results" / "tables"
    
    k = 1
    budget_percent = 100  # Full budget to ensure all margins computed
    
    results_list = []
    
    csv_files = [f for f in os.listdir(nyc_2025_folder) if f.endswith('.csv') and 'DEM' in f]
    total_files = len(csv_files)
    
    print(f"Analyzing {total_files} DEM elections with 100% budget")
    print("=" * 60)
    
    for idx, input_file in enumerate(csv_files, 1):
        full_path = nyc_2025_folder / input_file
        
        try:
            print(f"\n[{idx}/{total_files}] {input_file}")
            
            df, processed_file = process_single_file(full_path)
            candidates_mapping = create_candidate_mapping(processed_file)
            ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
            candidates = list(candidates_mapping.values())
            
            total_votes = sum(ballot_counts.values())
            num_candidates = len(candidates)
            
            if total_votes < 100 or num_candidates < 2:
                continue
            
            # Run STV
            Q = total_votes / 2 + 1  # Simple majority
            rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
            results, _ = utils.return_main_sub(rt)
            
            # Remap
            reverse_mapping = {code: name for name, code in candidates_mapping.items()}
            ordered_names = [reverse_mapping[code] for code in results]
            final_mapping = {name: ascii_uppercase[i] for i, name in enumerate(ordered_names)}
            
            ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
            candidates = list(final_mapping.values())
            
            # Compute exhaustion
            exhausted_list, exhausted_dict = IRV_ballot_exhaust(candidates, ballot_counts)
            exhausted_pct = {key: round(value / total_votes * 100, 3) 
                           for key, value in exhausted_dict.items()}
            
            # Compute strategies with FULL budget (100%)
            budget = total_votes  # 100% budget
            strats = reach_any_winners_campaign(candidates, k, Q, ballot_counts, budget)
            
            # Convert to percentage
            strats_pct = {}
            for cand, data in strats.items():
                if isinstance(data, list) and len(data) >= 2:
                    gap_pct = round(data[0] / total_votes * 100, 3)
                    strats_pct[cand] = [gap_pct, data[1]]
                else:
                    strats_pct[cand] = data
            
            # Get margin (smallest non-zero gap)
            gaps = [v[0] for v in strats_pct.values() if isinstance(v, list) and v[0] > 0]
            margin = min(gaps) if gaps else None
            
            print(f"   Candidates: {num_candidates}, Margin: {margin}")
            
            results_list.append({
                "file_name": input_file,
                "total_votes": total_votes,
                "num_candidates": num_candidates,
                "quota": Q,
                "overall_winning_order": results,
                "Strategies": strats_pct,
                "exhaust_percents": exhausted_pct,
                "budget_percent": budget_percent,
                "margin_of_victory": margin
            })
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    # Save
    df_results = pd.DataFrame(results_list)
    output_path = results_dir / "summary_table_nyc_2025_full_margins.xlsx"
    df_results.to_excel(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    margins = df_results['margin_of_victory'].dropna()
    print(f"\nMargins computed: {len(margins)}/{len(df_results)}")
    print(f"Mean margin: {margins.mean():.2f}%")
    print(f"Median margin: {margins.median():.2f}%")
    
    return df_results


if __name__ == "__main__":
    analyze_with_full_margins()
