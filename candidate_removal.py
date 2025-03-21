from utils import get_new_dict
from operator import itemgetter

def strict_support(ballot_counts, lower_group, upper_group, candidate):
    """
    Calculate strict support for a candidate based on ballot preferences.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        lower_group: Group of candidates to consider for first choices
        upper_group: Group of candidates to exclude
        candidate: The specific candidate to calculate support for
        
    Returns:
        Integer count of strict support for the candidate
    """
    letter_counts = {}
    total_cands = lower_group + upper_group
    
    # Calculate strict support
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in total_cands:
            while key[i] in lower_group and key[i] not in upper_group:
                newset.append(key[i])
                i += 1
                if i >= len(key):
                    break

        new_key = ''.join(char for char in newset)
        if candidate in new_key:
            if candidate in letter_counts:
                letter_counts[candidate] += value
            else:
                letter_counts[candidate] = value

    return letter_counts.get(candidate, 0)


def check_removal(candidates, group, ballot_counts, budget):
    """
    Check if a group of candidates can be removed while retaining other candidates.
    
    Args:
        candidates: List of candidates to retain
        group: String of candidates to potentially remove
        ballot_counts: Dictionary of ballots and their counts
        budget: Available budget
        
    Returns:
        Boolean indicating if removal is possible
    """
    # Calculate strict support for each candidate in group
    strict_support_dict = {}

    for key, value in ballot_counts.items():
        i = 0
        newset = []
        while i < len(key) and key[i] in group:
            newset.append(key[i])
            i += 1
        
        new_key = ''.join(char for char in newset)
        for letter in new_key:
            if letter in strict_support_dict:
                strict_support_dict[letter] += value
            else:
                strict_support_dict[letter] = value
    
    can_remove = False
   
    for best_c_irrelevant in strict_support_dict.keys():
        groupcopy = group
        mostly_irrelevant = groupcopy.replace(best_c_irrelevant, "")

        # Filter data by removing mostly_irrelevant letters
        filtered_data = {}
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in mostly_irrelevant)
            filtered_data[new_key] = filtered_data.get(new_key, 0) + value
        
        filtered_data.pop('', None)  # Remove empty key if exists
        aggre_v_dict = get_new_dict(filtered_data)
    
        # Get aggregated votes for relevant candidates
        relevant_aggre_dict = {c: aggre_v_dict[c] for c in candidates}
        worst_c_relevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
        
        if int(relevant_aggre_dict[worst_c_relevant] - strict_support_dict[best_c_irrelevant]) + 1 >= budget:
            can_remove = True
        else:
            # Check if addition changes who drops out after worst candidate removal
            last_three = sorted(relevant_aggre_dict.items(), key=itemgetter(1))[:3]
            
            # Budget not enough to benefit 2 candidates at the bottom
            if 2 * last_three[2][1] - last_three[1][1] - last_three[0][1] > budget:
                maybe_irrelevant = mostly_irrelevant + worst_c_relevant
                
                # Filter data again with new irrelevant set
                filtered_data = {}
                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in maybe_irrelevant)
                    filtered_data[new_key] = filtered_data.get(new_key, 0) + value
                
                filtered_data.pop('', None)
                aggre_v_dict = get_new_dict(filtered_data)

                # Add budget votes to best_c_irrelevant
                aggre_v_dict[best_c_irrelevant] = aggre_v_dict[best_c_irrelevant] + budget

                # New candidate list that removes worst and adds best_c_irrelevant
                candidates_temp = candidates.copy()
                candidates_temp.remove(worst_c_relevant)
                candidates_temp.append(best_c_irrelevant)

                relevant_aggre_dict = {c: aggre_v_dict[c] for c in candidates_temp}
                new_best_c_irrelevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)

                if new_best_c_irrelevant == best_c_irrelevant:  # Still gets eliminated
                    can_remove = True
                else:
                    can_remove = False
                    return False
            else:
                return False
                
    return can_remove


def remove_irrelevent(ballot_counts, rt, startcandidates, budget, fullgroup):
    """
    Remove irrelevant candidates iteratively.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        rt: Unknown parameter (not used in function)
        startcandidates: Initial list of candidates to consider
        budget: Available budget
        fullgroup: Complete set of all candidates
        
    Returns:
        Tuple of (remaining candidates, group of removed candidates, stop flag)
    """
    candidatesnew = startcandidates
    group = ''.join(char for char in fullgroup if char not in candidatesnew)
   
    while not check_removal(candidatesnew, group, ballot_counts, budget):
        candidatesnew = candidatesnew[:-1]
        group = ''.join(char for char in fullgroup if char not in candidatesnew)
        
        if len(candidatesnew) <= 2:
            stop = False
            return candidatesnew, group, stop

    return candidatesnew, group, check_removal(candidatesnew, group, ballot_counts, budget)


def predict_wins(ballot_counts, candidates, k, Q, budget):
    """
    Predict number of winning candidates.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        candidates: List of candidate identifiers
        k: Number of seats/positions
        Q: Quota value
        budget: Available budget
        
    Returns:
        Number of predicted winners (U_W)
    """
    C_W = []
    for cand in candidates:
        if budget + strict_support(ballot_counts, candidates, [], cand) > Q + budget/(k+1):
            C_W.append(cand)

    C_bar_W = [cand for cand in candidates if cand not in C_W]
    number_unique_ballots = 0
    
    for cand in C_W:
        # Call strict_support only for new unique ballots
        c_new = C_bar_W + [cand]
        support_count = strict_support(ballot_counts, c_new, [], cand)
        number_unique_ballots += support_count  # Accumulate only unique contribution

    U_W = min(k, int((budget + number_unique_ballots)/(Q + budget/(k+1))))
    return U_W


def predict_losses(ballot_counts, candidates, k, Q, budget):
    """
    Predict number of losing candidates.
    
    Args:
        ballot_counts: Dictionary of ballots and their counts
        candidates: List of candidate identifiers
        k: Number of seats/positions
        Q: Quota value
        budget: Available budget
        
    Returns:
        Number of predicted losers (i_L)
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    first_choice_votes = [aggre_v_dict[i] for i in candidates]
    first_choice_votes = sorted(first_choice_votes, reverse=True)

    C_L = []
    for cand in candidates:
        if budget + strict_support(ballot_counts, candidates, [], cand) <= first_choice_votes[k-1]:
            C_L.append(cand)

    T = []
    for c_i in C_L:
        t_i = 0
        for ballot, count in ballot_counts.items():
            # Check if this ballot starts with c_i
            if len(ballot) > 0 and ballot[0] == c_i:
                # Look at subsequent candidates in the ballot
                for cand in ballot[1:]:
                    if cand not in C_L:
                        t_i += count  # Found a candidate outside C_L -> add once
                        break         # Don't check further; we already transferred
        T.append(t_i)
    
    T = sorted(T, reverse=True)
    
    i_L = 1
    if len(T) == 0:
        return 0
    
    while i_L < len(T) and sum(T[:i_L]) + first_choice_votes[0] < Q + budget/(k+1):
        i_L += 1
        
    return i_L