from copy import deepcopy
from utils import get_new_dict


def IRV_optimal_result(cands, ballot_counts):
    """
    Produce optimal social choice order for Instant Runoff Voting (IRV) or Single-winner RCV.
    
    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        
    Returns:
        list: Ordered list of candidates from last eliminated to first.
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    
    # Eliminate candidates one by one
    for _ in range(len(candidates_remaining)):
        # Count first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }
        
        # Find candidate with fewest votes
        worst_candidate = min(vote_counts, key=vote_counts.get)
        
        # Remove candidate from remaining list and add to results
        candidates_remaining.remove(worst_candidate)
        results.insert(0, worst_candidate)
        
        # Redistribute votes from ballots containing eliminated candidates
        filtered_ballots = {}
        for ballot, count in ballot_counts.items():
            # Remove all eliminated candidates from ballots
            new_ballot = ''.join(char for char in ballot if char not in results)
            filtered_ballots[new_ballot] = filtered_ballots.get(new_ballot, 0) + count
        
        # Remove empty ballots
        filtered_ballots.pop('', None)
        
        # Recalculate aggregated votes
        aggregated_votes = get_new_dict(filtered_ballots)
        
    return results


def STV_optimal_result_simple(cands, ballot_counts, k, Q):
    """
    Compute the optimal social choice order for Single Transferable Vote (STV).

    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        k (int): Number of winners.
        Q (float): Quota threshold for winning.

    Returns:
        tuple: Contains three elements:
            - List of events with winners (1) and eliminations (0)
            - Dictionary summarizing results for each candidate
            - List of ballot states in each round
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    event_log = []
    result_dict = {candidate: 0 for candidate in cands}
   
    # Track ballot state in each round
    round_history = []
    current_round = 0
    round_history.append([ballot_counts, current_round])

    # Process candidates until all are either winners or eliminated
    while candidates_remaining:
        current_round += 1
        
        # Calculate first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }

        # Check if any candidate meets the quota
        winner = None
        for candidate, votes in vote_counts.items():
            if votes >= Q:
                # Select the candidate with the most votes who meets quota
                winner = max(vote_counts, key=vote_counts.get)
                
                # Record winner
                event_log.append([winner, 1])
                result_dict[winner] = 1
                candidates_remaining.remove(winner)
                results.append(winner)

                # Distribute surplus votes proportionally
                surplus = votes - Q
                if surplus > 0:
                    transfer_weight = surplus / votes
                    new_ballots = {}

                    # Redistribute votes
                    for ballot, count in ballot_counts.items():
                        if ballot.startswith(winner):
                            # Remove the winner from the ballot
                            new_ballot = ballot[1:]
                            new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count * transfer_weight
                        else:
                            if winner in ballot:
                                # Remove winner from anywhere in ballot
                                new_ballot = ''.join(char for char in ballot if char != winner)
                                new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                            else:
                                # Keep ballot as is
                                new_ballots[ballot] = new_ballots.get(ballot, 0) + count
                    
                    # Clean up and update
                    new_ballots.pop('', None)
                    ballot_counts = new_ballots
                break

        # If no candidate meets the quota, eliminate the lowest
        if winner is None:
            if not vote_counts:  # Check if no candidates remain
                break
            
            # Find candidate with fewest votes
            loser = min(vote_counts, key=vote_counts.get)
            
            # Record elimination
            candidates_remaining.remove(loser)
            event_log.append([loser, 0])
            result_dict[loser] = 0
            results.append(loser)

            # Redistribute votes of the eliminated candidate
            new_ballots = {}
            for ballot, count in ballot_counts.items():
                if ballot.startswith(loser):
                    # Remove the loser from the start of the ballot
                    new_ballot = ballot[1:]
                    new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                else:
                    if loser in ballot:
                        # Remove loser from anywhere in ballot
                        new_ballot = ''.join(char for char in ballot if char != loser)
                        new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                    else:
                        # Keep ballot as is
                        new_ballots[ballot] = new_ballots.get(ballot, 0) + count

            # Clean up and update
            new_ballots.pop('', None)
            ballot_counts = new_ballots

        # Recalculate votes and record round state
        aggregated_votes = get_new_dict(ballot_counts)
        round_history.append([ballot_counts, current_round])

    return event_log, result_dict, round_history