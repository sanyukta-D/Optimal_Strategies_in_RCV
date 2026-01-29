"""
Analyze Tractable Thresholds for Large RCV Elections
=====================================================

This script analyzes the comprehensive RCV Dataverse dataset to identify
budget thresholds at which large elections (>8 candidates) can be reduced 
to tractable instances for strategy computation.

Dataset Overview (dataverse_files):
- 607 ranked-choice voting elections
- 22 years of coverage (2004-2025)
- 29 U.S. jurisdictions including:
    * New York City (127 elections)
    * Alaska statewide (115 elections)
    * San Francisco (82 elections)
    * Minneapolis (80 elections)
    * Oakland (73 elections)
    * And 24 additional cities/jurisdictions
- Over 46.5 million ballots cast
- 2-37 candidates per election (avg: 5.3)

This script specifically focuses on the 39 elections with >8 candidates,
finding the minimum budget percentage at which each can be reduced to
a tractable instance (≤8 candidates) using candidate removal algorithms.

Output: results/tables/large_elections_thresholds.xlsx
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
from string import ascii_uppercase, ascii_lowercase

from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.core.candidate_removal import remove_irrelevent
from rcv_strategies.utils.helpers import get_new_dict, return_main_sub

# Path to dataverse files - comprehensive RCV election dataset
# Contains 607 elections across 29 jurisdictions (2004-2025)
DATAVERSE_DIR = Path(__file__).parent.parent / "dataverse_files (1)"


def load_and_prepare_election(csv_path):
    """
    Load an RCV election CSV and prepare ballot counts for analysis.
    
    Handles various CSV formats from different jurisdictions in the dataverse,
    including NYC, San Francisco, Minneapolis, Oakland, Alaska, etc.
    
    Args:
        csv_path: Path to election CSV file
        
    Returns:
        Tuple of (ballot_counts, candidates, total_votes, error_message)
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Find ranking columns
    rank_cols = [c for c in df.columns if c.lower().startswith('rank')]
    if not rank_cols:
        rank_cols = [c for c in df.columns if c.startswith('Choice_')]
    
    if not rank_cols:
        return None, None, None, "No rank columns"
    
    # Sort columns
    try:
        rank_cols = sorted(rank_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))
    except:
        pass
    
    # Get unique candidates
    all_candidates = set()
    exclude = {'skipped', 'overvote', 'undervote', 'writein', 'write-in', ''}
    
    for col in rank_cols:
        for val in df[col].dropna().unique():
            val_str = str(val).strip()
            if val_str.lower() not in exclude and val_str:
                all_candidates.add(val_str)
    
    if len(all_candidates) < 2:
        return None, None, None, "Too few candidates"
    
    # Create mapping
    sorted_candidates = sorted(all_candidates)
    letters = ascii_uppercase + ascii_lowercase
    mapping = {name: letters[i] for i, name in enumerate(sorted_candidates) if i < len(letters)}
    
    # Build ballot counts
    ballot_counts = {}
    for _, row in df.iterrows():
        ballot = []
        for col in rank_cols:
            val = str(row[col]).strip() if pd.notna(row[col]) else ''
            if val.lower() not in exclude and val in mapping:
                ballot.append(mapping[val])
        
        if ballot:
            ballot_str = ''.join(ballot)
            ballot_counts[ballot_str] = ballot_counts.get(ballot_str, 0) + 1
    
    candidates = list(mapping.values())
    total_votes = sum(ballot_counts.values())
    
    return ballot_counts, candidates, total_votes, None


def find_tractable_threshold(csv_path, target_candidates=8, k=1):
    """
    Find the minimum budget threshold for tractable strategy computation.
    
    Uses a coarse-to-fine search to identify the budget percentage at which
    a large election can be reduced to target_candidates or fewer using
    the candidate removal algorithm. Does NOT compute full strategies.
    
    Args:
        csv_path: Path to election CSV file
        target_candidates: Maximum candidates for tractability (default: 8)
        k: Number of winners (default: 1 for IRV)
        
    Returns:
        Tuple of (threshold_percent, original_candidates, status_message)
    """
    ballot_counts, candidates, total_votes, error = load_and_prepare_election(csv_path)
    
    if error:
        return None, len(candidates) if candidates else 0, error
    
    num_candidates = len(candidates)
    if num_candidates <= target_candidates:
        return 100.0, num_candidates, "Already tractable"
    
    # Run STV to get elimination order
    Q = total_votes / 2 + 1
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, _ = return_main_sub(rt)
    
    # Binary search for threshold
    def can_reduce_at_budget(budget_pct):
        budget = budget_pct * total_votes / 100
        candidates_reduced, group, success = remove_irrelevent(
            ballot_counts, rt, results[:target_candidates], budget, ''.join(results), rigorous_check=True
        )
        return success and len(candidates_reduced) <= target_candidates
    
    # Coarse search first
    test_budgets = [50, 40, 30, 25, 20, 15, 10, 7.5, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1]
    
    working_budget = None
    for budget in test_budgets:
        if can_reduce_at_budget(budget):
            working_budget = budget
            break
    
    if working_budget is None:
        return None, num_candidates, "Cannot reduce"
    
    # Binary search to find highest working budget
    # Find the failed budget above working_budget
    idx = test_budgets.index(working_budget)
    upper = test_budgets[idx - 1] if idx > 0 else 100
    lower = working_budget
    
    precision = 0.1 if working_budget < 5 else 0.5
    
    while upper - lower > precision:
        mid = round((upper + lower) / 2, 2)
        if can_reduce_at_budget(mid):
            lower = mid
            working_budget = mid
        else:
            upper = mid
    
    return working_budget, num_candidates, "OK"


def find_all_thresholds():
    """
    Analyze all large elections (>8 candidates) from the dataverse dataset.
    
    From the 607 elections in the dataverse, 39 have more than 8 candidates.
    This function computes tractable thresholds for each and saves results
    to an Excel file for further analysis.
    
    The elections span multiple jurisdictions and years:
    - Minneapolis Mayor elections (2009-2021) with up to 35 candidates
    - San Francisco Board of Supervisors and Mayor races
    - NYC Democratic primaries (2021, 2025)
    - Oakland Mayor elections
    - And more across 22 years of RCV history
    """
    
    # Large elections (>8 candidates) identified from scanning all 607 files
    LARGE_ELECTION_FILES = [
        "Minneapolis_20131105_Mayor.csv",  # 35
        "SanFrancisco_20111108_Mayor.csv",  # 23
        "SanFrancisco_20041102_BoardofSupervisorsDistrict5.csv",  # 22
        "PortlandOR_20241105_Mayor.csv",  # 21
        "SanFrancisco_20101102_BoardofSupervisorsDistrict10.csv",  # 21
        "SanFrancisco_20071106_Mayor.csv",  # 18
        "Minneapolis_20171107_Mayor.csv",  # 18
        "Minneapolis_20211102_Mayor.csv",  # 18
        "Oakland_20141104_Mayor.csv",  # 16
        "NewYorkCity_20210622_DEM_CityCouncilD26.csv",  # 15
        "SanFrancisco_20241105_Mayor.csv",  # 15
        "SanFrancisco_20101102_BoardofSupervisorsDistrict6.csv",  # 14
        "SanFrancisco_20041102_BoardofSupervisorsDistrict7.csv",  # 13
        "NewYorkCity_20210622_DEM_CityCouncilD9.csv",  # 13
        "NewYorkCity_20210622_DEM_Mayor.csv",  # 13
        "NewYorkCity_20210622_DEM_BoroughPresidentKings.csv",  # 12
        "NewYorkCity_20210622_DEM_CityCouncilD27.csv",  # 12
        "NewYorkCity_20210622_DEM_CityCouncilD7.csv",  # 12
        "SanFrancisco_20151103_Mayor.csv",  # 12
        "Minneapolis_20200811_CityCouncilWard6_Special.csv",  # 12
        "Minneapolis_20091106_Mayor.csv",  # 11
        "NewYorkCity_20210622_DEM_CityCouncilD40.csv",  # 11
        "NewYorkCity_20250624_DEMMayorCitywide.csv",  # 11
        "NewYorkCity_20250624_DEMCityCouncilD41.csv",  # 9
        "LasCruces_20191105_Mayor.csv",  # 10
        "NewYorkCity_20210622_DEM_Comptroller.csv",  # 10
        "Oakland_20101102_Mayor.csv",  # 10
        "Oakland_20181106_Mayor.csv",  # 10
        "Oakland_20221108_Mayor.csv",  # 10
        "Oakland_20241105_CityCouncil_AtLarge.csv",  # 10
        "Oakland_20250415_Mayor.csv",  # 10
        "SanFrancisco_20161108_BoardofSupervisorsDistrict1.csv",  # 10
        "NewYorkCity_20210622_DEM_CityCouncilD1.csv",  # 9
        "NewYorkCity_20210622_DEM_CityCouncilD29.csv",  # 9
        "NewYorkCity_20210622_DEM_CityCouncilD49.csv",  # 9
        "SanFrancisco_20081104_BoardofSupervisorsDistrict1.csv",  # 9
        "SanFrancisco_20081104_BoardofSupervisorsDistrict11.csv",  # 9
        "SanFrancisco_20081104_BoardofSupervisorsDistrict3.csv",  # 9
        "SanFrancisco_20121106_BoardofSupervisorsDistrict7.csv",  # 9
        "SanFrancisco_20180605_Mayor.csv",  # 9
    ]
    
    large_elections = []
    for fname in LARGE_ELECTION_FILES:
        csv_path = DATAVERSE_DIR / fname
        if csv_path.exists():
            ballot_counts, candidates, total_votes, error = load_and_prepare_election(csv_path)
            if candidates:
                large_elections.append((csv_path, len(candidates), total_votes))
    
    large_elections.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {len(large_elections)} elections with > 8 candidates")
    print("=" * 90)
    print(f"{'Election':<55} | {'Cands':>5} | {'Threshold':>10} | Status")
    print("=" * 90)
    
    results = []
    
    for csv_path, num_cands, total_votes in large_elections:
        threshold, final_cands, status = find_tractable_threshold(csv_path, target_candidates=8)
        
        if threshold is not None:
            threshold_str = f"{threshold:.1f}%"
        else:
            threshold_str = "N/A"
        
        print(f"{csv_path.name[:55]:<55} | {num_cands:>5} | {threshold_str:>10} | {status}")
        
        results.append({
            'file': csv_path.name,
            'candidates': num_cands,
            'votes': total_votes,
            'threshold': threshold,
            'status': status
        })
    
    print("=" * 90)
    
    # Summary stats
    valid_thresholds = [r['threshold'] for r in results if r['threshold'] is not None]
    if valid_thresholds:
        print(f"\nSummary:")
        print(f"  Elections reduced: {len(valid_thresholds)}/{len(results)}")
        print(f"  Mean threshold: {sum(valid_thresholds)/len(valid_thresholds):.2f}%")
        print(f"  Min threshold: {min(valid_thresholds):.2f}%")
        print(f"  Max threshold: {max(valid_thresholds):.2f}%")
    
    # Save to Excel
    df = pd.DataFrame(results)
    df['reduced_to'] = 8  # Target reduction
    df = df.rename(columns={
        'file': 'Election',
        'candidates': 'Original_Candidates',
        'votes': 'Total_Votes',
        'threshold': 'Threshold_Percent',
        'status': 'Status',
        'reduced_to': 'Reduced_To'
    })
    
    # Reorder columns
    df = df[['Election', 'Original_Candidates', 'Reduced_To', 'Total_Votes', 'Threshold_Percent', 'Status']]
    df = df.sort_values('Threshold_Percent', ascending=True)
    
    output_path = Path(__file__).parent.parent / "results" / "tables" / "large_elections_thresholds.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = find_all_thresholds()
