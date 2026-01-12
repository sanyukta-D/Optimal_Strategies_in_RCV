from copy import deepcopy
from rcv_strategies.utils.helpers import get_new_dict


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

def IRV_ballot_exhaust(cands, ballot_counts):
    """
    Produce ballot exhaustion for Instant Runoff Voting (IRV) or Single-winner RCV.
    
    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        
    Returns:
        list: Ordered list of candidates from last eliminated to first.
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    number_of_ballots = sum(ballot_counts.values())
    ballot_count_list = [number_of_ballots]    
    exhausted_ballots_list = []
    exhausted_ballots_dict = {}
    filtered_ballots = ballot_counts.copy()
    # Eliminate candidates one by one
    for _ in range(len(candidates_remaining)):
        # Count first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }
        
        # Find candidate with fewest votes
        worst_candidate = min(vote_counts, key=vote_counts.get)

        ballot_count_list.append(sum(filtered_ballots.values()))
        # Check if any ballots are exhausted
        exhausted_ballots_list.append(ballot_count_list[0] - ballot_count_list[-1])
        exhausted_ballots_dict[worst_candidate] = ballot_count_list[0] - ballot_count_list[-1]
        
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
        
    return exhausted_ballots_list, exhausted_ballots_dict


def STV_ballot_exhaust(cands, ballot_counts, k, Q):
    """
    Produce ballot exhaustion for Single Transferable Vote (STV).
    
    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        k (int): Number of winners.
        Q (float): Quota threshold for winning.
        
    Returns:
        tuple: Contains three elements:
            - List of exhausted ballots at each round
            - Dictionary mapping eliminated candidates to number of exhausted ballots
            - List of winners (for verification)
    """
    candidates_remaining = deepcopy(cands)
    aggregated_votes = get_new_dict(ballot_counts)
    results = []
    winners = []
    
    # Track total ballots at start
    total_ballots = sum(ballot_counts.values())
    ballot_count_list = [total_ballots]    
    exhausted_ballots_list = []
    exhausted_ballots_dict = {}
    filtered_ballots = ballot_counts.copy()
    
    # Process until we have k winners
    while len(winners) < k and candidates_remaining:
        # Count first-choice votes for remaining candidates
        vote_counts = {
            candidate: aggregated_votes.get(candidate, 0) 
            for candidate in candidates_remaining
        }
        
        if not vote_counts:
            break
            
        # Track current ballot count before any changes
        current_ballot_count = sum(filtered_ballots.values())
        ballot_count_list.append(current_ballot_count)
            
        # Check if any candidate meets the quota
        winner = None
        for candidate, votes in vote_counts.items():
            if votes >= Q:
                # Select the candidate with the most votes who meets quota
                winner = max(vote_counts, key=vote_counts.get)
                
                # Record winner
                candidates_remaining.remove(winner)
                winners.append(winner)
                results.append(winner)

                # Count exhausted ballots after processing winner
                exhausted_count = 0
                new_ballots = {}
                
                # First pass: remove winner and count exhausted
                for ballot, count in filtered_ballots.items():
                    if ballot.startswith(winner):
                        # For ballots starting with winner, check if they have valid choices after removal
                        new_ballot = ballot[1:]
                        if not any(c in candidates_remaining for c in new_ballot):
                            exhausted_count += count
                        elif new_ballot:  # Only keep non-empty ballots
                            new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                    else:
                        if winner in ballot:
                            # Remove winner from anywhere in ballot
                            new_ballot = ''.join(char for char in ballot if char != winner)
                            if not any(c in candidates_remaining for c in new_ballot):
                                exhausted_count += count
                            elif new_ballot:  # Only keep non-empty ballots
                                new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
                        else:
                            # Keep ballot as is
                            new_ballots[ballot] = new_ballots.get(new_ballot, 0) + count

                # Distribute surplus votes proportionally
                surplus = votes - Q
                if surplus > 0:
                    transfer_weight = surplus / votes
                    final_ballots = {}
                    
                    # Second pass: distribute surplus
                    for ballot, count in new_ballots.items():
                        if ballot.startswith(winner):
                            # Remove the winner from the ballot
                            new_ballot = ballot[1:]
                            if new_ballot:  # Only keep non-empty ballots
                                final_ballots[new_ballot] = final_ballots.get(new_ballot, 0) + count * transfer_weight
                        else:
                            final_ballots[ballot] = final_ballots.get(ballot, 0) + count
                    
                    # Clean up and update
                    filtered_ballots = final_ballots
                else:
                    filtered_ballots = new_ballots

                # Add exhausted ballots for this round
                exhausted_ballots_list.append(exhausted_count)
                
                # Track which candidates caused ballot exhaustion
                if exhausted_count > 0:
                    exhausted_ballots_dict[winner] = exhausted_count
                
                break
                
        # If no candidate meets the quota, eliminate the lowest
        if winner is None:
            # Find candidate with fewest votes
            worst_candidate = min(vote_counts, key=vote_counts.get)
            
            # Count exhausted ballots after elimination
            exhausted_count = 0
            new_ballots = {}
            
            for ballot, count in filtered_ballots.items():
                # Remove the eliminated candidate from anywhere in ballot
                new_ballot = ''.join(char for char in ballot if char != worst_candidate)
                if not any(c in candidates_remaining for c in new_ballot):
                    exhausted_count += count
                elif new_ballot:  # Only keep non-empty ballots
                    new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count
            
            # Remove candidate from remaining list and add to results
            candidates_remaining.remove(worst_candidate)
            results.append(worst_candidate)
            
            filtered_ballots = new_ballots
            
            # Add exhausted ballots for this round
            exhausted_ballots_list.append(exhausted_count)
            
            # Track which candidates caused ballot exhaustion
            if exhausted_count > 0:
                exhausted_ballots_dict[worst_candidate] = exhausted_count
        
        # Recalculate aggregated votes
        aggregated_votes = get_new_dict(filtered_ballots)
    
    return exhausted_ballots_list, exhausted_ballots_dict, winners


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
                # Remove the loser from anywhere in the ballot
                new_ballot = ''.join(char for char in ballot if char != loser)
                new_ballots[new_ballot] = new_ballots.get(new_ballot, 0) + count

            # Clean up empty ballots
            new_ballots.pop('', None)
            ballot_counts = new_ballots

        # Recalculate votes and record round state
        aggregated_votes = get_new_dict(ballot_counts)
        round_history.append([ballot_counts, current_round])

    return event_log, result_dict, round_history


def create_STV_round_result_given_structure(structure, checked_candidates, remaining_candidates, ballot_counts, Q):
    """
    Compute the round-counts for a given structure (social choice order) using Single Transferable Vote (STV).

    Args:
        stdt (dict): Description of the structure.
        checked_candidates (list): List of candidates already checked.  
        remaining_candidates (list): List of remaining candidates.
        currentdict (dict): Dictionary with current vote counts.
        Q (float): Quota threshold for winning.

    Returns:
        new currentdict: Updated dictionary with current vote counts.
    """
    if len(checked_candidates) == 0:
        return ballot_counts
    currentdict = get_new_dict(ballot_counts)
    
    last_candidate = checked_candidates[-1]  # The most recently processed candidate
    new_ballot_counts = {last_candidate: currentdict[last_candidate] }
    if structure[last_candidate] == 0: 
        # Remove eliminated candidates while retaining the rest of the string
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char != last_candidate)
            new_ballot_counts[new_key] = new_ballot_counts.get(new_key, 0) + value
        new_ballot_counts.pop('', None)
        return new_ballot_counts
    
    else:
        # Distribute surplus votes proportionally
        surplus = currentdict[last_candidate] - Q
        if surplus > 0:
            transfer_weight = surplus / currentdict[last_candidate]

            # Redistribute votes
            for ballot, count in ballot_counts.items():
                if ballot.startswith(last_candidate):
                    # Remove the winner from the ballot
                    new_ballot = ballot[1:]
                    new_ballot_counts[new_ballot] = new_ballot_counts.get(new_ballot, 0) + count * transfer_weight
                else:
                    if last_candidate in ballot:
                        # Remove winner from anywhere in ballot
                        new_ballot = ''.join(char for char in ballot if char != last_candidate)
                        new_ballot_counts[new_ballot] = new_ballot_counts.get(new_ballot, 0) + count
                    else:
                        # Keep ballot as is
                        new_ballot_counts[ballot] = new_ballot_counts.get(ballot, 0) + count
            
            # Clean up and update
            new_ballot_counts.pop('', None)
            return new_ballot_counts
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in [last_candidate])
            new_ballot_counts[new_key] = new_ballot_counts.get(new_key, 0) + value
        new_ballot_counts.pop('', None)
        return new_ballot_counts

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



    