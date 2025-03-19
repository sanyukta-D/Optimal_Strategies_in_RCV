from copy import deepcopy
from utils import get_new_dict



##produce optimal social choice order for IRV or Single-winner RCV

def IRV_optimal_result(cands, ballot_counts):
    
    candsnew= deepcopy(cands)
    aggre_v_dict_mynew = get_new_dict(ballot_counts)
    results=[]
    
    for i in range(len(candsnew)):
        relevant_aggre_dict={}
    
        for c in candsnew:
        
            relevant_aggre_dict[c] =  aggre_v_dict_mynew.get(c, 0)
            
        worst_c = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
        
        candsnew.remove(worst_c)
        results.insert(0, worst_c)
        # Initialize a dictionary for filtered data
        filtered_data = {}

        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in results)

            filtered_data[new_key] =   filtered_data.get(new_key, 0) + value
            
        filtered_data.pop('', None)
        aggre_v_dict_mynew={}

        aggre_v_dict_mynew = get_new_dict(filtered_data)
        
    return results
    

def STV_optimal_result_simple(cands, ballot_counts, k, Q):
    """
    Compute the optimal social choice order for Single Transferable Vote (STV).

    Args:
        cands (list): List of candidates.
        ballot_counts (dict): Dictionary where keys are ranked ballots (as strings) and values are counts.
        k (int): Number of winners.

    Returns:
        list, dict: List of events with winners (1) and eliminations (0), and a dictionary summarizing the same.
        collection/list of updated dicts of remaining votes in every round
    """
    candsnew = deepcopy(cands)
    aggre_v_dict_mynew = get_new_dict(ballot_counts)
    results = []
    event_log = []
    result_dict = {c: 0 for c in cands}
   
    collection = []
    current_round = 0
    collection.append([ballot_counts, current_round])

    # Calculate the Droop quota
    total_votes = sum(ballot_counts.values())

    while candsnew:  # Process all candidates until all are either winners or eliminated
        current_round = current_round + 1 
        relevant_aggre_dict = {}

        # Calculate first-choice votes for remaining candidates
        for c in candsnew:
            relevant_aggre_dict[c] = aggre_v_dict_mynew.get(c, 0)

        # Check if any candidate meets the quota
        winner = None
        for candidate, votes in relevant_aggre_dict.items():
            if votes >= Q:  # Ensure only `k` winners
                winner = max(relevant_aggre_dict, key=relevant_aggre_dict.get)
            
                event_log.append([winner, 1])
                result_dict[winner] = 1
                candsnew.remove(winner)

                # Distribute surplus votes proportionally
                surplus = votes - Q
                if surplus > 0:
                    transfer_weight = surplus / votes
                    filtered_data_1 = {}

                    for key, value in ballot_counts.items():
                        if key.startswith(winner):
                            new_key = key[1:]  # Remove the winner from the ballot
                            filtered_data_1[new_key] = filtered_data_1.get(new_key, 0) + value * transfer_weight
                        else:
                            if winner in key:
                                newkey = ''.join(char for char in key if char not in [winner])
                                filtered_data_1[newkey] = filtered_data_1.get(newkey, 0) + value
                            else:
                                filtered_data_1[key] = filtered_data_1.get(key, 0) + value
                        

                filtered_data_1.pop('', None)
                results.append(winner)
                break

        if winner is None:  # No candidate meets the quota, eliminate the lowest
            if not relevant_aggre_dict:  # Check if relevant_aggre_dict is empty
                break  # Exit the loop if no candidates remain
        

            loser = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
            candsnew.remove(loser)
            event_log.append([loser, 0])
            result_dict[loser] = 0
            results.append(loser)

            # Redistribute votes of the eliminated candidate
            filtered_data_1 = {}

            for key, value in ballot_counts.items():
                if key.startswith(loser):
                    new_key = key[1:]  # Remove the loser from the ballot
                    filtered_data_1[new_key] = filtered_data_1.get(new_key, 0) + value
                else:
                    if loser in key:
                        newkey = ''.join(char for char in key if char not in [loser])
                        filtered_data_1[newkey] = filtered_data_1.get(newkey, 0) + value
                    else:
                        filtered_data_1[key] = filtered_data_1.get(key, 0) + value

            filtered_data_1.pop('', None)

        aggre_v_dict_mynew={}
        ballot_counts = filtered_data_1
        aggre_v_dict_mynew = get_new_dict(ballot_counts)
        collection.append([ballot_counts, current_round])


    return event_log, result_dict, collection
