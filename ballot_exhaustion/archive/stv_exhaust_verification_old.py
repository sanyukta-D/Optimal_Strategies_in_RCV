"""
Archived exhaustion verification functions.
Originally in rcv_strategies/core/stv_irv.py.

These were experimental/verification functions that are never called anywhere.
STV_ballot_exhaust_verification has a subtle bug: uses a set (loses counts)
for used_ballots and never updates current_ballots from round_history.

The correct implementation is STV_ballot_exhaust() in stv_irv.py, which
replicates the proven print_exhaustion_analysis approach.

Archived: 2026-02-03
"""
from copy import deepcopy
from rcv_strategies.core.stv_irv import STV_optimal_result_simple


def verify_exhaustion_matches_no_winners(ballot_counts, winners, total_exhausted):
    """
    Verify that the final exhaustion rate matches ballots that didn't rank any winners.

    Args:
        ballot_counts (dict): Original ballot counts
        winners (list): List of winning candidates
        total_exhausted (int): Total number of exhausted ballots

    Returns:
        tuple: (bool, int) - Whether verification passed and count of ballots with no winners
    """
    ballots_with_no_winners = 0
    for ballot, count in ballot_counts.items():
        if not any(winner in ballot for winner in winners):
            ballots_with_no_winners += count

    return ballots_with_no_winners == total_exhausted, ballots_with_no_winners

def STV_ballot_exhaust_verification(cands, ballot_counts, k, Q):
    """
    Verify ballot exhaustion using the following logic:
    1. Initialize "used ballots" and "exhausted ballots" empty
    2. For each round:
       - If there's a win, move ballots ranking winner first to "used ballots"
       - Update set of active candidates
       - Exhausted ballots = ballots (excluding used) that don't rank any active candidates
    3. For ballots ranking winners at later positions, treat as if they didn't rank the winner
       if the winner is already elected when that position is reached

    NOTE: This has a subtle bug — uses used_ballots as a set (loses counts)
    and never updates current_ballots from round_history.

    Args:
        cands (list): List of candidates
        ballot_counts (dict): Dictionary of ballot counts
        k (int): Number of winners
        Q (float): Quota threshold

    Returns:
        tuple: (exhausted_ballots_list, used_ballots, winners)
    """
    candidates_remaining = deepcopy(cands)
    used_ballots = set()  # Set of ballots that contributed to a win
    exhausted_ballots_list = []
    winners = []
    total_ballots = sum(ballot_counts.values())

    # Get the social choice order first
    event_log, result_dict, _ = STV_optimal_result_simple(cands, ballot_counts, k, Q)

    # Process each round based on the event log
    for candidate, is_winner in event_log:
        if is_winner:
            # This is a winning round
            winners.append(candidate)
            candidates_remaining.remove(candidate)

            # Move ballots ranking this winner first to used_ballots
            for ballot in ballot_counts:
                if ballot.startswith(candidate):
                    used_ballots.add(ballot)
        else:
            # This is an elimination round
            candidates_remaining.remove(candidate)

        # Count exhausted ballots (excluding used ballots)
        exhausted_count = 0
        for ballot, count in ballot_counts.items():
            if ballot not in used_ballots:
                # Check if ballot has any remaining active candidates
                if not any(c in candidates_remaining for c in ballot):
                    exhausted_count += count

        exhausted_ballots_list.append(exhausted_count)

    return exhausted_ballots_list, used_ballots, winners

def verify_exhaustion_implementation(ballot_counts, winners, used_ballots, total_exhausted):
    """
    Verify the exhaustion implementation matches the described logic.

    Args:
        ballot_counts (dict): Original ballot counts
        winners (list): List of winning candidates
        used_ballots (set): Set of ballots that contributed to wins
        total_exhausted (int): Total number of exhausted ballots

    Returns:
        tuple: (bool, dict) - Whether verification passed and detailed counts
    """
    # Count ballots that don't rank any winners and aren't used
    ballots_with_no_winners = 0
    for ballot, count in ballot_counts.items():
        if ballot not in used_ballots and not any(winner in ballot for winner in winners):
            ballots_with_no_winners += count

    # Count used ballots
    used_ballots_count = sum(ballot_counts[ballot] for ballot in used_ballots)

    # Count total ballots
    total_ballots = sum(ballot_counts.values())

    return {
        'verification_passed': ballots_with_no_winners == total_exhausted,
        'ballots_with_no_winners': ballots_with_no_winners,
        'used_ballots_count': used_ballots_count,
        'total_ballots': total_ballots,
        'total_exhausted': total_exhausted
    }
