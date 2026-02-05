"""
Archived: analyze_portland_exhaustion wrapper function.
Originally in case_studies/portland/exhaustion_analysis.py.

This was a debug wrapper around STV_ballot_exhaust that was never called
externally. Now redundant since STV_ballot_exhaust produces correct values.

Archived: 2026-02-03
"""
import pandas as pd
import os
from rcv_strategies.core.stv_irv import STV_ballot_exhaust, STV_optimal_result_simple
from case_studies.portland.load_district_data import district_data
from rcv_strategies.utils import case_study_helpers

def analyze_portland_exhaustion(district_number, k=3):
    """
    Analyze ballot exhaustion for a Portland district's STV election.

    Args:
        district_number (int): District number (1-4)
        k (int): Number of winners

    Returns:
        tuple: (exhausted_ballots_list, exhausted_ballots_dict)
    """
    # Ensure k is an integer
    k = int(k)
    # Get district data
    df = district_data[district_number]['df']
    candidates_mapping = district_data[district_number]['candidates_mapping']
    print(f"\nDebug: Candidates mapping for District {district_number}:")
    for kmap, v in candidates_mapping.items():
        print(f"  {kmap!r} -> {v!r}")
    print(f"\nDebug: DataFrame columns for District {district_number}:")
    print(df.columns.tolist())
    print(f"Debug: First 3 rows of DataFrame:")
    print(df.head(3))

    # Convert ballot data using the proper helper function
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)

    print(f"\nDebug: Total ballots parsed: {sum(ballot_counts.values())}")
    print(f"Debug: Number of unique ballot patterns: {len(ballot_counts)}")
    print(f"Debug: Sample ballot patterns:")
    for ballot, count in list(ballot_counts.items())[:5]:
        print(f"  {ballot}: {count} votes")

    # Calculate quota (Droop quota)
    total_votes = sum(ballot_counts.values())
    quota = (total_votes // (k + 1)) + 1
    print(f"Debug: Total votes: {total_votes}")
    print(f"Debug: Quota: {quota}")

    # Get list of candidates
    candidates = list(candidates_mapping.values())
    print(f"Debug: Candidates: {candidates}")

    # Calculate exhaustion - STV_ballot_exhaust returns 3 values
    exhausted_ballots_list, exhausted_ballots_dict, winners = STV_ballot_exhaust(
        cands=candidates,
        ballot_counts=ballot_counts,
        k=k,
        Q=quota
    )

    return exhausted_ballots_list, exhausted_ballots_dict
