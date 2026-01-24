#!/usr/bin/env python3
"""
Test script to verify webapp produces correct victory gap values
by comparing against expected results from the paper's appendix tables.
"""

import sys
import pandas as pd
from pathlib import Path
from string import ascii_uppercase
import re

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.utils.helpers import return_main_sub
from rcv_strategies.utils.case_study_helpers import (
    get_ballot_counts_df,
    process_ballot_counts_post_elim_no_print
)

# ============================================================================
# EXPECTED VALUES FROM PAPER TABLES
# ============================================================================

# NYC Elections - extracted from nyc_election_summary.tex
NYC_EXPECTED = {
    "NewYorkCity_06222021_DEMCouncilMember1stCouncilDistrict.csv": {"A": 0.0, "B": 17.1, "C": 20.1, "D": 37.5, "E": 28.9},
    "NewYorkCity_06222021_DEMCouncilMember2ndCouncilDistrict.csv": {"A": 0.0, "B": 32.0},
    "NewYorkCity_06222021_DEMCouncilMember3rdCouncilDistrict.csv": {"A": 0.0, "B": 35.8, "C": 34.3},
    "NewYorkCity_06222021_DEMCouncilMember10thCouncilDistrict.csv": {"A": 0.0, "B": 17.2, "C": 26.4},
    "NewYorkCity_06222021_DEMCouncilMember11thCouncilDistrict.csv": {"A": 0.0, "B": 20.5, "C": 27.1},
    "NewYorkCity_06222021_DEMCouncilMember12thCouncilDistrict.csv": {"A": 0.0, "B": 17.0, "C": 24.3},
    "NewYorkCity_06222021_DEMCouncilMember13thCouncilDistrict.csv": {"A": 0.0, "B": 32.9},
    "NewYorkCity_06222021_DEMCouncilMember14thCouncilDistrict.csv": {"A": 0.0, "B": 20.5, "C": 21.4, "D": 38.3, "E": 39.1},
    "NewYorkCity_06222021_DEMCouncilMember15thCouncilDistrict.csv": {"A": 0.0, "B": 24.6, "C": 36.1, "D": 33.0},
    "NewYorkCity_06222021_DEMCouncilMember16thCouncilDistrict.csv": {"A": 0.0, "B": 28.0, "C": 29.7},
    "NewYorkCity_06222021_DEMCouncilMember17thCouncilDistrict.csv": {"A": 0.0, "B": 13.8},
    "NewYorkCity_06222021_DEMCouncilMember18thCouncilDistrict.csv": {"A": 0.0, "B": 3.9, "C": 26.8, "D": 27.4, "E": 35.6},
    "NewYorkCity_06222021_DEMCouncilMember19thCouncilDistrict.csv": {"A": 0.0, "B": 8.4, "C": 21.8, "D": 39.0},
    "NewYorkCity_06222021_DEMCouncilMember20thCouncilDistrict.csv": {"A": 0.0, "B": 7.5, "C": 16.4, "D": 21.0, "E": 29.2, "F": 24.0, "G": 28.4},
    "NewYorkCity_06222021_DEMCouncilMember21stCouncilDistrict.csv": {"A": 0.0, "B": 39.6, "C": 39.6},
    "NewYorkCity_06222021_DEMCouncilMember22ndCouncilDistrict.csv": {"A": 0.0, "B": 22.6},
    "NewYorkCity_06222021_DEMCouncilMember23rdCouncilDistrict.csv": {"A": 0.0, "B": 7.1, "C": 22.9, "D": 20.3, "E": 29.2, "G": 38.3},
    "NewYorkCity_06222021_DEMCouncilMember24thCouncilDistrict.csv": {"A": 0.0, "B": 30.5},
    "NewYorkCity_06222021_DEMCouncilMember25thCouncilDistrict.csv": {"A": 0.0, "B": 5.4, "C": 11.8, "D": 24.0, "E": 24.5, "F": 29.2, "G": 33.6},
    "NewYorkCity_06222021_DEMCouncilMember28thCouncilDistrict.csv": {"A": 0.0, "B": 39.6, "C": 34.2},
    "NewYorkCity_06222021_DEMCouncilMember29thCouncilDistrict.csv": {"A": 0.0, "B": 14.1, "C": 16.4, "D": 24.2, "E": 25.4, "F": 27.0, "G": 29.7, "H": 38.9},
    "NewYorkCity_06222021_DEMCouncilMember30thCouncilDistrict.csv": {"A": 0.0, "B": 6.4},
    "NewYorkCity_06222021_DEMCouncilMember31stCouncilDistrict.csv": {"A": 0.0, "B": 48.4, "C": 55.6},
    "NewYorkCity_06222021_DEMCouncilMember32ndCouncilDistrict.csv": {"A": 0.0, "B": 4.2, "C": 33.2, "D": 38.2, "E": 36.8},
    "NewYorkCity_06222021_DEMCouncilMember33rdCouncilDistrict.csv": {"A": 0.0, "B": 24.7},
    "NewYorkCity_06222021_DEMCouncilMember34thCouncilDistrict.csv": {"A": 0.0, "B": 73.9, "C": 75.4, "D": 81.4},
    "NewYorkCity_06222021_DEMCouncilMember35thCouncilDistrict.csv": {"A": 0.0, "B": 6.9, "C": 35.6},
    "NewYorkCity_06222021_DEMCouncilMember36thCouncilDistrict.csv": {"A": 0.0, "B": 11.8, "C": 7.4, "D": 23.1},
    "NewYorkCity_06222021_DEMCouncilMember37thCouncilDistrict.csv": {"A": 0.0, "B": 26.4},
    "NewYorkCity_06222021_DEMCouncilMember38thCouncilDistrict.csv": {"A": 0.0, "B": 26.2, "C": 34.9, "D": 36.9, "E": 36.4},
    "NewYorkCity_06222021_DEMCouncilMember39thCouncilDistrict.csv": {"A": 0.0, "B": 10.9, "C": 19.4, "D": 29.7, "E": 39.8},
    "NewYorkCity_06222021_DEMCouncilMember41stCouncilDistrict.csv": {"A": 0.0, "B": 4.8},
    "NewYorkCity_06222021_DEMCouncilMember42ndCouncilDistrict.csv": {"A": 0.0, "B": 7.1},
    "NewYorkCity_06222021_DEMCouncilMember47thCouncilDistrict.csv": {"A": 0.0, "B": 10.3, "C": 36.9},
    "NewYorkCity_06222021_DEMCouncilMember48thCouncilDistrict.csv": {"A": 0.0, "B": 12.2, "C": 23.4, "D": 25.9},
    "NewYorkCity_06222021_DEMBoroughPresidentBronx.csv": {"A": 0.0, "B": 6.3, "C": 22.2, "D": 30.5},
    "NewYorkCity_06222021_DEMBoroughPresidentQueens.csv": {"A": 0.0, "B": 0.5, "C": 30.8},
    "NewYorkCity_06222021_DEMBoroughPresidentRichmond.csv": {"A": 0.0, "B": 26.7, "C": 36.7},
}

# Alaska Elections - extracted from alaska_election_summary.tex
ALASKA_EXPECTED = {
    "Alaska_08162022_HouseofRepresentativesSpecial.csv": {"A": 0.0, "B": 2.7, "C": 2.7, "D": 53.1},
    "Alaska_11052024_StateHouseD1.csv": {"A": 0.0, "B": 20.6, "C": 23.8, "D": 64.7},
    "Alaska_11052024_StateHouseD2.csv": {"A": 0.0, "B": 94.5},
    "Alaska_11052024_StateHouseD3.csv": {"A": 0.0, "B": 92.0},
    "Alaska_11052024_StateHouseD4.csv": {"A": 0.0, "B": 93.0},
    "Alaska_11052024_StateHouseD5.csv": {"A": 0.0, "B": 55.2, "C": 82.2},
    "Alaska_11052024_StateHouseD6.csv": {"A": 0.0, "B": 4.3, "C": 34.0, "D": 54.6},
    "Alaska_11052024_StateHouseD7.csv": {"A": 0.0, "B": 18.8, "C": 68.1},
    "Alaska_11052024_StateHouseD8.csv": {"A": 0.0, "B": 4.4, "C": 62.1},
    "Alaska_11052024_StateHouseD9.csv": {"A": 0.0, "B": 8.9, "C": 59.3},
    "Alaska_11052024_StateHouseD10.csv": {"A": 0.0, "B": 24.5, "C": 69.9},
    "Alaska_11052024_StateHouseD11.csv": {"A": 0.0, "B": 5.3, "C": 61.1},
    "Alaska_11052024_StateHouseD12.csv": {"A": 0.0, "B": 22.0, "C": 67.0},
    "Alaska_11052024_StateHouseD13.csv": {"A": 0.0, "B": 6.9, "C": 60.2},
    "Alaska_11052024_StateHouseD14.csv": {"A": 0.0, "B": 56.6, "C": 82.7},
    "Alaska_11052024_StateHouseD15.csv": {"A": 0.0, "B": 4.6, "C": 42.8, "D": 62.2},
    "Alaska_11052024_StateHouseD16.csv": {"A": 0.0, "B": 14.0, "C": 63.3},
    "Alaska_11052024_StateHouseD17.csv": {"A": 0.0, "B": 87.6},
    "Alaska_11052024_StateHouseD18.csv": {"A": 0.0, "B": 0.5, "C": 57.8},
    "Alaska_11052024_StateHouseD19.csv": {"A": 0.0, "B": 30.5, "C": 49.7, "D": 67.3},
    "Alaska_11052024_StateHouseD20.csv": {"A": 0.0, "B": 28.5, "C": 69.2},
    "Alaska_11052024_StateHouseD21.csv": {"A": 0.0, "B": 10.8, "C": 60.0},
    "Alaska_11052024_StateHouseD22.csv": {"A": 0.0, "B": 4.9, "C": 60.9},
    "Alaska_11052024_StateHouseD23.csv": {"A": 0.0, "B": 24.0, "C": 67.9},
    "Alaska_11052024_StateHouseD24.csv": {"A": 0.0, "B": 93.0},
    "Alaska_11052024_StateHouseD25.csv": {"A": 0.0, "B": 91.5},
    "Alaska_11052024_StateHouseD26.csv": {"A": 0.0, "B": 92.2},
    "Alaska_11052024_StateHouseD27.csv": {"A": 0.0, "B": 2.5, "C": 61.4},
    "Alaska_11052024_StateHouseD28.csv": {"A": 0.0, "B": 0.1, "C": 10.7, "D": 52.3},
    "Alaska_11052024_StateHouseD29.csv": {"A": 0.0, "B": 90.8},
    "Alaska_11052024_StateHouseD30.csv": {"A": 0.0, "B": 9.9, "C": 62.0},
    "Alaska_11052024_StateHouseD31.csv": {"A": 0.0, "B": 9.0, "C": 59.9},
    "Alaska_11052024_StateHouseD32.csv": {"A": 0.0, "B": 35.0, "C": 74.6},
    "Alaska_11052024_StateHouseD34.csv": {"A": 0.0, "B": 12.8, "C": 64.9},
    "Alaska_11052024_StateHouseD35.csv": {"A": 0.0, "B": 10.6, "C": 59.8},
    "Alaska_11052024_StateHouseD36.csv": {"A": 0.0, "B": 10.2, "C": 11.3, "D": 33.6, "E": 53.2},
    "Alaska_11052024_StateHouseD37.csv": {"A": 0.0, "B": 45.9, "C": 80.0},
    "Alaska_11052024_StateHouseD38.csv": {"A": 0.0, "B": 3.6, "C": 17.9, "D": 41.6, "E": 62.5},
    "Alaska_11052024_StateHouseD39.csv": {"A": 0.0, "B": 16.7, "C": 75.1},
    "Alaska_11052024_StateHouseD40.csv": {"A": 0.0, "B": 17.5, "C": 24.8, "D": 66.2},
    "Alaska_11052024_StateSenateB.csv": {"A": 0.0, "B": 92.7},
    "Alaska_11052024_StateSenateD.csv": {"A": 0.0, "B": 8.8, "C": 45.5, "D": 61.8},
    "Alaska_11052024_StateSenateF.csv": {"A": 0.0, "B": 5.5, "C": 43.2, "D": 60.3},
    "Alaska_11052024_StateSenateH.csv": {"A": 0.0, "B": 10.7, "C": 60.6},
    "Alaska_11052024_StateSenateJ.csv": {"A": 0.0, "B": 40.4, "C": 73.7},
    "Alaska_11052024_StateSenateL.csv": {"A": 0.0, "B": 9.9, "C": 41.0, "D": 61.7},
    "Alaska_11052024_StateSenateN.csv": {"A": 0.0, "B": 23.0, "C": 35.6, "D": 66.5},
    "Alaska_11052024_StateSenateP.csv": {"A": 0.0, "B": 3.0, "C": 57.8},
    "Alaska_11052024_StateSenateR.csv": {"A": 0.0, "B": 9.9, "C": 39.8, "D": 62.1},
    "Alaska_11052024_StateSenateT.csv": {"A": 0.0, "B": 94.2},
    "Alaska_11052024_US_House.csv": {"A": 0.0, "B": 2.4, "C": 42.5, "D": 45.3, "E": 57.4},
}


def detect_format(df):
    """Detect the format of the uploaded CSV."""
    cols = df.columns.tolist()
    if any(col.startswith('Choice_') for col in cols):
        return 'choice'
    elif any(col.startswith('rank') for col in cols):
        return 'rank'
    return None


def convert_rank_to_choice(df):
    """Convert rank1, rank2... format to Choice_1, Choice_2... format."""
    rename_map = {}
    for col in df.columns:
        if col.startswith('rank') and col[4:].isdigit():
            rename_map[col] = f"Choice_{col[4:]}"
    return df.rename(columns=rename_map)


def get_candidates_from_df(df):
    """Extract unique candidates from the dataframe."""
    choice_cols = [col for col in df.columns if col.startswith('Choice_')]
    all_candidates = set()
    for col in choice_cols:
        candidates = df[col].dropna().unique()
        all_candidates.update(candidates)

    exclude = {'', 'skipped', 'overvote', 'undervote', 'writein', 'exhausted', 'nan'}
    candidates = [c for c in all_candidates
                  if str(c).strip().lower() not in exclude
                  and pd.notna(c)
                  and str(c).strip() != '']
    return sorted(candidates, key=str)


def run_analysis(csv_path, k=1, budget_percent=100.0):
    """
    Run the same analysis as the webapp and return strategy results.
    Uses the same remapping logic: Winner→A, Runner-up→B, etc.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Detect and convert format
    data_format = detect_format(df)
    if data_format == 'rank':
        df = convert_rank_to_choice(df)
    elif data_format is None:
        raise ValueError(f"Could not detect data format for {csv_path}")

    # Get candidates
    candidates = get_candidates_from_df(df)
    if len(candidates) < 2:
        return None, "Less than 2 candidates"

    # Step 1: Initial mapping (alphabetical)
    initial_mapping = {name: ascii_uppercase[i] for i, name in enumerate(candidates)}
    initial_ballot_counts = get_ballot_counts_df(initial_mapping, df)
    total_votes = sum(initial_ballot_counts.values())

    if total_votes == 0:
        return None, "No valid ballots"

    initial_candidates_list = list(initial_mapping.values())

    # Step 2: First STV run to get social choice order
    Q = round(total_votes / (k + 1) + 1, 3)
    if k == 1:
        Q = Q * (k + 1)

    rt_initial, dt_initial, _ = STV_optimal_result_simple(
        initial_candidates_list, initial_ballot_counts, k, Q
    )
    initial_results, _ = return_main_sub(rt_initial)

    # Step 3: CRITICAL REMAPPING - Winner→A, Runner-up→B, etc.
    initial_reverse = {v: k for k, v in initial_mapping.items()}
    ordered_candidate_names = [initial_reverse[code] for code in initial_results]

    # Create final mapping based on social choice order
    final_mapping = {name: ascii_uppercase[i] for i, name in enumerate(ordered_candidate_names)}

    # Rebuild ballot counts with new mapping
    ballot_counts = get_ballot_counts_df(final_mapping, df)
    candidates_list = list(final_mapping.values())

    # Step 4: Run full analysis
    analysis_result = process_ballot_counts_post_elim_no_print(
        ballot_counts=ballot_counts,
        k=k,
        candidates=candidates_list,
        elim_cands=[],
        check_strats=True,
        budget_percent=budget_percent,
        check_removal_here=(len(candidates_list) > 9),
        keep_at_least=8,
        rigorous_check=True
    )

    return analysis_result, None


def compare_strategies(computed, expected, tolerance=0.15):
    """
    Compare computed strategies with expected values.
    Returns (match, details) where match is True if within tolerance.
    """
    if computed is None:
        return False, "No computed results"

    strategies = computed.get("Strategies", {})
    results = computed.get("overall_winning_order", [])

    discrepancies = []
    matches = []

    for code, expected_gap in expected.items():
        if code in strategies:
            strat = strategies[code]
            if isinstance(strat, (list, tuple)) and len(strat) > 0:
                computed_gap = strat[0]
            else:
                computed_gap = None
        else:
            computed_gap = None

        if computed_gap is None:
            discrepancies.append(f"{code}: expected {expected_gap}%, got None")
        else:
            diff = abs(computed_gap - expected_gap)
            if diff <= tolerance:
                matches.append(f"{code}: {computed_gap:.1f}% (expected {expected_gap}%) ✓")
            else:
                discrepancies.append(f"{code}: computed {computed_gap:.2f}%, expected {expected_gap}% (diff: {diff:.2f}%)")

    all_match = len(discrepancies) == 0
    return all_match, {"matches": matches, "discrepancies": discrepancies, "social_order": results}


def test_elections(data_dir, expected_dict, region_name, tolerance=0.15):
    """Test all elections in a directory against expected values."""
    print(f"\n{'='*60}")
    print(f"TESTING {region_name} ELECTIONS")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    skipped = 0
    all_results = {}

    for filename, expected in expected_dict.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            print(f"\n⚠️  SKIP: {filename} - file not found")
            skipped += 1
            continue

        print(f"\n📊 Testing: {filename}")

        try:
            result, error = run_analysis(csv_path, k=1, budget_percent=100.0)

            if error:
                print(f"   ❌ Error: {error}")
                failed += 1
                continue

            match, details = compare_strategies(result, expected, tolerance)

            all_results[filename] = {
                "computed": result.get("Strategies", {}),
                "expected": expected,
                "match": match,
                "details": details
            }

            if match:
                print(f"   ✅ PASS - All values match within {tolerance}% tolerance")
                for m in details["matches"][:3]:  # Show first 3 matches
                    print(f"      {m}")
                if len(details["matches"]) > 3:
                    print(f"      ... and {len(details['matches'])-3} more")
                passed += 1
            else:
                print(f"   ❌ FAIL - Discrepancies found:")
                for d in details["discrepancies"]:
                    print(f"      {d}")
                print(f"   Social Choice Order: {details['social_order']}")
                failed += 1

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'-'*60}")
    print(f"{region_name} SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'-'*60}")

    return passed, failed, skipped, all_results


def main():
    """Run all tests."""
    print("="*60)
    print("RCV WEBAPP ACCURACY TEST")
    print("Comparing computed victory gaps against paper appendix tables")
    print("="*60)

    base_path = Path(__file__).parent

    # Test NYC elections
    nyc_data_dir = base_path / "case_studies" / "nyc" / "data"
    nyc_passed, nyc_failed, nyc_skipped, nyc_results = test_elections(
        nyc_data_dir, NYC_EXPECTED, "NYC", tolerance=0.15
    )

    # Test Alaska elections
    alaska_data_dir = base_path / "case_studies" / "alaska" / "data"
    alaska_passed, alaska_failed, alaska_skipped, alaska_results = test_elections(
        alaska_data_dir, ALASKA_EXPECTED, "ALASKA", tolerance=0.15
    )

    # Overall summary
    total_passed = nyc_passed + alaska_passed
    total_failed = nyc_failed + alaska_failed
    total_skipped = nyc_skipped + alaska_skipped
    total_tests = total_passed + total_failed

    print("\n" + "="*60)
    print("OVERALL TEST RESULTS")
    print("="*60)
    print(f"Total tests run: {total_tests}")
    print(f"  ✅ Passed: {total_passed} ({100*total_passed/total_tests:.1f}%)" if total_tests > 0 else "  No tests run")
    print(f"  ❌ Failed: {total_failed}")
    print(f"  ⚠️  Skipped: {total_skipped}")

    if total_failed == 0 and total_tests > 0:
        print("\n🎉 ALL TESTS PASSED! Webapp logic matches paper values.")
    elif total_failed > 0:
        print(f"\n⚠️  {total_failed} test(s) failed. Review discrepancies above.")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
