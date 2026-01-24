#!/usr/bin/env python3
"""
Test script for large elections (>8 candidates) using divide-and-conquer
and some small elections for confirmation.
"""

import sys
import pandas as pd
from pathlib import Path
from string import ascii_uppercase

sys.path.insert(0, str(Path(__file__).parent))

from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.utils.helpers import return_main_sub
from rcv_strategies.utils.case_study_helpers import (
    get_ballot_counts_df,
    process_ballot_counts_post_elim_no_print
)

# Expected values from paper tables
LARGE_ELECTIONS = {
    # NYC elections with >8 candidates
    "case_studies/nyc/data/NewYorkCity_06222021_DEMCouncilMember9thCouncilDistrict.csv": {
        "name": "NYC Council District 9",
        "candidates": 13,
        "expected": {"A": 0.0, "B": 0.3},
        "paper_threshold": 9.5
    },
    "case_studies/nyc/data/NewYorkCity_06222021_DEMCouncilMember7thCouncilDistrict.csv": {
        "name": "NYC Council District 7",
        "candidates": 12,
        "expected": {"A": 0.0},
        "paper_threshold": 18.0
    },
    "case_studies/nyc/data/NewYorkCity_06222021_DEMCouncilMember40thCouncilDistrict.csv": {
        "name": "NYC Council District 40",
        "candidates": 11,
        "expected": {"A": 0.0, "C": 13.4},
        "paper_threshold": 14.0
    },
    "case_studies/nyc/data/NewYorkCity_06222021_DEMBoroughPresidentKings.csv": {
        "name": "NYC Kings Borough President",
        "candidates": 12,
        "expected": {"A": 0.0, "B": 6.6, "C": 12.1},
        "paper_threshold": 13.5
    },
}

SMALL_ELECTIONS = {
    # Small elections for confirmation
    "case_studies/nyc/data/NewYorkCity_06222021_DEMCouncilMember2ndCouncilDistrict.csv": {
        "name": "NYC Council District 2",
        "candidates": 2,
        "expected": {"A": 0.0, "B": 32.0}
    },
    "case_studies/nyc/data/NewYorkCity_06222021_DEMCouncilMember17thCouncilDistrict.csv": {
        "name": "NYC Council District 17",
        "candidates": 2,
        "expected": {"A": 0.0, "B": 13.8}
    },
    "case_studies/alaska/data/Alaska_08162022_HouseofRepresentativesSpecial.csv": {
        "name": "Alaska House Special",
        "candidates": 4,
        "expected": {"A": 0.0, "B": 2.7, "C": 2.7, "D": 53.1}
    },
    "case_studies/alaska/data/Alaska_11052024_StateHouseD6.csv": {
        "name": "Alaska State House D6",
        "candidates": 4,
        "expected": {"A": 0.0, "B": 4.3, "C": 34.0, "D": 54.6}
    },
}


def load_and_prepare_data(csv_path):
    """Load CSV and prepare ballot counts."""
    df = pd.read_csv(csv_path)

    # Detect and convert format
    if any(col.startswith('rank') for col in df.columns):
        rename_map = {col: f'Choice_{col[4:]}' for col in df.columns
                      if col.startswith('rank') and col[4:].isdigit()}
        df = df.rename(columns=rename_map)

    # Get candidates
    choice_cols = [col for col in df.columns if col.startswith('Choice_')]
    all_candidates = set()
    for col in choice_cols:
        all_candidates.update(df[col].dropna().unique())

    exclude = {'', 'skipped', 'overvote', 'undervote', 'writein', 'exhausted', 'nan'}
    candidates = sorted([c for c in all_candidates
                        if str(c).strip().lower() not in exclude
                        and pd.notna(c) and str(c).strip() != ''], key=str)

    return df, candidates


def run_analysis_with_divide_conquer(csv_path, user_budget=40.0):
    """Run analysis with divide-and-conquer for large elections."""
    df, candidates = load_and_prepare_data(csv_path)

    if len(candidates) < 2:
        return None, None, "Less than 2 candidates"

    # Initial mapping and first STV run for social choice order
    initial_mapping = {name: ascii_uppercase[i] for i, name in enumerate(candidates)}
    initial_ballot_counts = get_ballot_counts_df(initial_mapping, df)
    total_votes = sum(initial_ballot_counts.values())

    k = 1
    Q = round(total_votes / (k + 1) + 1, 3)
    if k == 1:
        Q = Q * (k + 1)

    initial_candidates_list = list(initial_mapping.values())
    rt_initial, _, _ = STV_optimal_result_simple(initial_candidates_list, initial_ballot_counts, k, Q)
    initial_results, _ = return_main_sub(rt_initial)

    # Remap: Winner→A, Runner-up→B, etc.
    initial_reverse = {v: k for k, v in initial_mapping.items()}
    ordered_names = [initial_reverse[code] for code in initial_results]
    final_mapping = {name: ascii_uppercase[i] for i, name in enumerate(ordered_names)}
    reverse_mapping = {v: k for k, v in final_mapping.items()}

    ballot_counts = get_ballot_counts_df(final_mapping, df)
    candidates_list = list(final_mapping.values())

    # Try user budget first
    analysis_result = process_ballot_counts_post_elim_no_print(
        ballot_counts=ballot_counts, k=k, candidates=candidates_list,
        elim_cands=[], check_strats=True, budget_percent=user_budget,
        check_removal_here=(len(candidates_list) > 9),
        keep_at_least=8, rigorous_check=True
    )

    strategies = analysis_result.get("Strategies", {})
    computed_threshold = user_budget

    # Divide and conquer if needed
    if not strategies and len(candidates_list) > 8:
        def try_budget(test_budget):
            result = process_ballot_counts_post_elim_no_print(
                ballot_counts=ballot_counts, k=k, candidates=candidates_list,
                elim_cands=[], check_strats=True, budget_percent=test_budget,
                check_removal_here=True, keep_at_least=8, rigorous_check=True
            )
            return result if result.get("Strategies", {}) else None

        # Phase 1: Coarse then fine search
        coarse_budgets = [b for b in [30, 25, 20, 15, 10, 7.5, 5] if b < user_budget]
        fine_budgets = [4, 3, 2.5, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.2]
        all_budgets = coarse_budgets + fine_budgets

        working_budget = None
        working_result = None

        for test_budget in all_budgets:
            if test_budget >= user_budget:
                continue
            result = try_budget(test_budget)
            if result:
                working_budget = test_budget
                working_result = result
                break

        # Phase 2: Binary search
        if working_budget is not None:
            try:
                idx = all_budgets.index(working_budget)
                upper = all_budgets[idx - 1] if idx > 0 else user_budget
            except ValueError:
                upper = user_budget
            lower = working_budget
            precision = 0.1 if working_budget < 5 else 0.5

            while upper - lower > precision:
                mid = round((upper + lower) / 2, 2)
                result = try_budget(mid)
                if result:
                    lower = mid
                    working_budget = mid
                    working_result = result
                else:
                    upper = mid

            strategies = working_result.get("Strategies", {})
            computed_threshold = working_budget
            analysis_result = working_result

    return strategies, computed_threshold, reverse_mapping


def test_election(csv_path, info, tolerance=0.2):
    """Test a single election."""
    name = info["name"]
    expected = info["expected"]
    paper_threshold = info.get("paper_threshold", None)

    print(f"\n{'='*60}")
    print(f"📊 {name} ({info['candidates']} candidates)")
    print(f"{'='*60}")

    try:
        strategies, threshold, reverse_mapping = run_analysis_with_divide_conquer(csv_path)

        if strategies is None:
            print(f"   ❌ Error: {reverse_mapping}")
            return False

        print(f"   Computed threshold: {threshold:.1f}%", end="")
        if paper_threshold:
            print(f" (paper: {paper_threshold}%)")
        else:
            print()

        # Compare strategies
        all_match = True
        for code, expected_gap in expected.items():
            if code in strategies:
                strat = strategies[code]
                computed_gap = strat[0] if isinstance(strat, (list, tuple)) else None
            else:
                computed_gap = None

            if computed_gap is None:
                print(f"   ❌ {code}: expected {expected_gap}%, got None")
                all_match = False
            else:
                diff = abs(computed_gap - expected_gap)
                if diff <= tolerance:
                    print(f"   ✅ {code}: {computed_gap:.2f}% (expected {expected_gap}%)")
                else:
                    print(f"   ❌ {code}: {computed_gap:.2f}% (expected {expected_gap}%, diff: {diff:.2f}%)")
                    all_match = False

        # Show extra strategies found
        for code, strat in strategies.items():
            if code not in expected:
                gap = strat[0] if isinstance(strat, (list, tuple)) else strat
                print(f"   ℹ️  {code}: {gap:.2f}% (extra, not in expected)")

        return all_match

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("TESTING LARGE ELECTIONS (Divide-and-Conquer)")
    print("="*60)

    large_passed = 0
    large_failed = 0

    for csv_path, info in LARGE_ELECTIONS.items():
        if test_election(csv_path, info):
            large_passed += 1
        else:
            large_failed += 1

    print(f"\n{'='*60}")
    print("TESTING SMALL ELECTIONS (Confirmation)")
    print("="*60)

    small_passed = 0
    small_failed = 0

    for csv_path, info in SMALL_ELECTIONS.items():
        if test_election(csv_path, info):
            small_passed += 1
        else:
            small_failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Large elections: {large_passed}/{large_passed + large_failed} passed")
    print(f"Small elections: {small_passed}/{small_passed + small_failed} passed")
    print(f"Total: {large_passed + small_passed}/{large_passed + large_failed + small_passed + small_failed} passed")

    if large_failed == 0 and small_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {large_failed + small_failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
