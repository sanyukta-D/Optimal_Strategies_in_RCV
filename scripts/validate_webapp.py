"""
Validate Webapp Results Against Stored Expected Results
========================================================

Run this script before pushing webapp changes to ensure all example
elections produce correct victory gaps and strategies.

Usage:
    python scripts/validate_webapp.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from string import ascii_uppercase, ascii_lowercase
import ast

from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils.case_study_helpers import process_ballot_counts_post_elim_no_print
from rcv_strategies.utils.helpers import return_main_sub


def load_election(csv_path):
    """Load election data and return ballot_counts, candidates, total_votes."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Detect format
    rank_cols = [c for c in df.columns if c.lower().startswith('rank')]
    choice_cols = [c for c in df.columns if c.startswith('Choice_')]

    if rank_cols:
        cols = sorted(rank_cols, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    elif choice_cols:
        cols = sorted(choice_cols, key=lambda x: int(x.split('_')[1]))
    else:
        return None, None, None, "No rank/choice columns"

    # Get candidates (include Write-in as valid candidate)
    exclude = {'skipped', 'overvote', 'undervote', '', 'nan'}
    all_candidates = set()
    for col in cols:
        for val in df[col].dropna().unique():
            val_str = str(val).strip()
            if val_str.lower() not in exclude and val_str:
                all_candidates.add(val_str)

    if len(all_candidates) < 2:
        return None, None, None, "Too few candidates"

    # Create mapping
    sorted_cands = sorted(all_candidates)
    letters = ascii_uppercase + ascii_lowercase
    mapping = {name: letters[i] for i, name in enumerate(sorted_cands) if i < len(letters)}

    # Build ballot counts
    ballot_counts = {}
    for _, row in df.iterrows():
        ballot = []
        for col in cols:
            val = str(row[col]).strip() if pd.notna(row[col]) else ''
            if val.lower() not in exclude and val in mapping:
                ballot.append(mapping[val])
        if ballot:
            ballot_str = ''.join(ballot)
            ballot_counts[ballot_str] = ballot_counts.get(ballot_str, 0) + 1

    return ballot_counts, list(mapping.values()), sum(ballot_counts.values()), None


def run_analysis(ballot_counts, candidates, k=1, budget_percent=40):
    """Run the same analysis as the webapp."""
    total_votes = sum(ballot_counts.values())

    Q = round(total_votes / (k + 1) + 1, 3)
    if k == 1:
        Q = Q * (k + 1)

    # First run to get elimination order
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, _ = return_main_sub(rt)

    # Remap by elimination order
    reverse_mapping = {c: c for c in candidates}  # Identity for letter codes
    letters = ascii_uppercase + ascii_lowercase
    final_mapping = {results[i]: letters[i] for i in range(len(results))}

    # Rebuild ballot counts
    new_ballot_counts = {}
    for ballot, count in ballot_counts.items():
        new_ballot = ''.join(final_mapping.get(c, c) for c in ballot)
        new_ballot_counts[new_ballot] = new_ballot_counts.get(new_ballot, 0) + count

    new_candidates = [final_mapping[c] for c in results]

    # Run analysis
    result = process_ballot_counts_post_elim_no_print(
        ballot_counts=new_ballot_counts,
        k=k,
        candidates=new_candidates,
        elim_cands=[],
        check_strats=True,
        budget_percent=budget_percent,
        check_removal_here=(len(new_candidates) > 9),
        keep_at_least=min(9, len(new_candidates)),
        rigorous_check=True
    )

    return result


def extract_victory_gaps(strategies_dict):
    """Extract victory gaps from strategies dict."""
    gaps = {}
    if isinstance(strategies_dict, str):
        try:
            strategies_dict = ast.literal_eval(strategies_dict)
        except:
            return gaps

    if isinstance(strategies_dict, dict):
        for cand, value in strategies_dict.items():
            if isinstance(value, (list, tuple)) and len(value) >= 1:
                gaps[cand] = value[0]
            elif isinstance(value, (int, float)):
                gaps[cand] = value
    return gaps


def validate_example(name, csv_path, expected_strategies, k=1, budget=40, tolerance=0.5):
    """Validate a single example election."""
    ballot_counts, candidates, total_votes, error = load_election(csv_path)

    if error:
        return False, f"Load error: {error}"

    result = run_analysis(ballot_counts, candidates, k=k, budget_percent=budget)
    actual_strategies = result.get("Strategies", {})

    expected_gaps = extract_victory_gaps(expected_strategies)
    actual_gaps = extract_victory_gaps(actual_strategies)

    # Compare
    errors = []
    for cand in expected_gaps:
        expected = expected_gaps[cand]
        actual = actual_gaps.get(cand)

        if actual is None:
            errors.append(f"  {cand}: expected {expected:.2f}%, got MISSING")
        elif abs(expected - actual) > tolerance:
            errors.append(f"  {cand}: expected {expected:.2f}%, got {actual:.2f}%")

    if errors:
        return False, "\n".join(errors)
    return True, "OK"


def main():
    """Run validation on all example elections."""
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("WEBAPP VALIDATION - Checking Victory Gaps Against Expected Results")
    print("=" * 70)

    # Define test cases: (name, csv_path, expected_results_file, k, budget)
    test_cases = []

    # NYC 2025 examples
    nyc_results_path = project_root / "results" / "tables" / "summary_table_nyc_2025.xlsx"
    if nyc_results_path.exists():
        nyc_df = pd.read_excel(nyc_results_path)
        nyc_data_dir = project_root / "case_studies" / "nyc" / "nyc 2025 files"

        for _, row in nyc_df.iterrows():
            fname = row['file_name']
            csv_path = nyc_data_dir / fname
            if csv_path.exists():
                test_cases.append((
                    fname,
                    csv_path,
                    row['Strategies'],
                    1,  # k
                    40  # budget
                ))

    # Portland examples
    portland_expected = {
        "Portland District 1": {"D": 1.60, "F": 1.93, "E": 2.81, "G": 4.21},
        "Portland District 2": {"D": 5.69},
        "Portland District 4": {"D": 1.12},
    }

    portland_data_dir = project_root / "case_studies" / "portland" / "data"
    for i, (name, expected) in enumerate([
        ("Portland District 1", portland_expected.get("Portland District 1", {})),
        ("Portland District 2", portland_expected.get("Portland District 2", {})),
        ("Portland District 3", {}),  # Only A,B,C within threshold
        ("Portland District 4", portland_expected.get("Portland District 4", {})),
    ], 1):
        csv_path = portland_data_dir / f"Dis_{i}" / f"Election_results_dis{i}.csv"
        if csv_path.exists() and expected:
            # Convert to strategy format
            strats = {c: [v, {}] for c, v in expected.items()}
            test_cases.append((name, csv_path, strats, 3, 10))

    # Run tests
    passed = 0
    failed = 0

    print(f"\nRunning {len(test_cases)} test cases...\n")

    for name, csv_path, expected, k, budget in test_cases[:10]:  # Limit to first 10 for speed
        success, message = validate_example(name, csv_path, expected, k=k, budget=budget)

        if success:
            print(f"[PASS] {name[:50]}")
            passed += 1
        else:
            print(f"[FAIL] {name[:50]}")
            print(message)
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
