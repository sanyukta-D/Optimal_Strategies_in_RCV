from copy import deepcopy
from utils import (
    get_new_dict, 
    clean_aggre_dict_diff, 
    create_structure, 
    give_winners, 
    str_for_given_winners, 
    return_main_sub, 
    main_structures, 
    sub_structures, 
    sub_structures_at_most_k_ones_fixed_last, 
    campaign_addition_dict_simple
)
from RCV_optimization_processing import roundupdate, decode_dict, STV_optimal_result
from STVandIRV_results import STV_optimal_result_simple
from itertools import combinations
import math


def add_campaign(log_campaign_list, main_st, remaining_candidates, decoded_dict, Q, k, t, stdt, budget):
    """
    Returns the campaign investment needed for a round, given budget.
    
    Args:
        log_campaign_list: List of campaign dictionaries for previous rounds
        main_st: Main structure
        remaining_candidates: List of candidates still in the race
        decoded_dict: Dictionary with current vote counts
        Q: Quota value
        k: Number of seats
        t: Current candidate being evaluated
        stdt: Structure dictionary indicating win/loss status
        budget: Available budget
        
    Returns:
        Tuple of (updated campaign list, boolean indicating success)
    """
    current_campaign_dict = {j: 0 for j in main_st} 
          
    if stdt[t] == 1:  # If candidate is a winner
        v = []
        for candidate in remaining_candidates:
            v.append(decoded_dict[candidate])
        t_update = max(0, (max(v) - decoded_dict[t]) + 1, (Q - decoded_dict[t]) + 1)
        if t_update > budget:
            return {}, False
        current_campaign_dict[t] = t_update
    else:  # If candidate is a loser
        for candidate in remaining_candidates:
            if decoded_dict[candidate] >= Q:
                return {}, False
            else:
                can_update = max(0, decoded_dict[t] - decoded_dict[candidate] + 1)
                if can_update > budget:
                    return {}, False
                current_campaign_dict[candidate] = can_update 
            
    log_campaign_list.append(deepcopy(current_campaign_dict))
    return log_campaign_list, True


def process_campaign_STV(candidates, main, sub, k, Q, aggre_v_dict, budget):
    """
    Given a structure and voter data and an updated Q, this gives a round-wise campaign 
    allocation requirement to make the structure feasible.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q: Quota value
        aggre_v_dict: Dictionary of aggregate votes
        budget: Available budget
        
    Returns:
        Tuple containing (decoded dictionaries, structure, campaign list, status list)
    """
    strt, stdt = create_structure(main, sub)
    currentdict = {can: [[can]] for can in candidates}

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    
    CheckDicts = []  # List of additions each round should have
    DecodedDicts = []  # List of how votes look in each round
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates) - 1):
        # Update according to current round of elimination/win
        currentdict = roundupdate(stdt, checked_candidates, remaining_candidates, currentdict)
        
        # Convert into numerical data
        decoded_dict = decode_dict(currentdict, candidates, Q, aggre_v_dict)

        # Perform the next elimination/win
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        # Check if the next elimination/win is alright and calculate required additions
        log_campaign_list, status = add_campaign(
            log_campaign_list, main, remaining_candidates, decoded_dict, Q, k, t, stdt, budget
        )
        if status == False:
            return {}, [], [], [False]
            
        status_list.append(status)
        DecodedDicts.append(decoded_dict)
        
    return DecodedDicts, strt, log_campaign_list, status_list


def process_campaign_STV_simple(candidates, main, sub, k, Q, ballot_counts, budget, collections):
    """
    Simplified version of process_campaign_STV that uses pre-calculated ballot counts.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        collections: Pre-calculated ballot collections
        
    Returns:
        Tuple containing (decoded dictionaries, structure, campaign list, status list)
    """
    strt, stdt = create_structure(main, sub)
    currentdict = {can: [[can]] for can in candidates}

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    DecodedDicts = []
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates) - 1):
        ballot_counts_at_time = collections[i][0]

        full_decoded_dict = get_new_dict(ballot_counts_at_time)
        decoded_dict = {cand: full_decoded_dict.get(cand, 0) for cand in list(stdt.keys())}
        
        # Perform the next elimination/win
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        # Check and calculate required campaign additions
        log_campaign_list, status = add_campaign(
            log_campaign_list, main, remaining_candidates, decoded_dict, Q, k, t, stdt, budget
        )
        if status == False:
            return {}, [], [], [False]
            
        status_list.append(status)
        DecodedDicts.append(decoded_dict)
        
    return DecodedDicts, strt, log_campaign_list, status_list


def smart_campaign(candidates, log_campaign_list, strt, stdt, Q, DecodedDicts, budget):
    """
    Takes the round-wise requirement dict and returns the optimized investment strategy.
    
    Args:
        candidates: List of candidate identifiers
        log_campaign_list: List of campaign dictionaries by round
        strt: Structure list
        stdt: Structure dictionary
        Q: Quota value
        DecodedDicts: List of decoded vote dictionaries by round
        budget: Available budget
        
    Returns:
        Tuple of (investment dictionary, amount spent)
    """
    log = deepcopy(log_campaign_list)
    dict1 = log[0]  # First round allocations
    invest_dict = {can: 0 for can in candidates}  # Current round investments
    total_investment_dict = {can: 0 for can in candidates}  # Total investments
    
    # Available funds from previous investments
    avl_dict = {can: 0 for can in candidates} 
    
    # First round: everyone gets what's needed
    for can in dict1.keys():
        if dict1[can] > 0:
            invest_dict[can] = dict1[can]
            total_investment_dict[can] = dict1[can]
             
    t = strt[0][0]  # First candidate for elimination or win
    decoded_dict = DecodedDicts[0]
    
    # Process subsequent rounds with reallocation
    for rd in range(len(log) - 1):
        # Update log of checked candidates and dissipate investment
        if stdt[t] == 1 and invest_dict[t] > 0:
            dee_invest = deepcopy(invest_dict)
            avl_dict[t] = math.floor(
                (decoded_dict[t] + dee_invest[t] - Q) * 
                (dee_invest[t] / (decoded_dict[t] + dee_invest[t]))
            )
            invest_dict[t] = 0
            
        if stdt[t] == 0 and invest_dict[t] > 0:
            for rich_can in total_investment_dict.keys():
                if rich_can[len(rich_can) - 1] == t:  # Investments owned by t
                    avl_dict[rich_can] = deepcopy(total_investment_dict)[rich_can]
            invest_dict[t] = 0
        
        # Process the next round
        dict_rd = log[rd + 1]
        decoded_dict = DecodedDicts[rd + 1]
        
        for can in dict_rd.keys():
            if dict_rd[can] > invest_dict[can]:
                needed_extra = dict_rd[can] - invest_dict[can]
                invest_dict[can] = dict_rd[can]
                
                # Check if previously checked candidates can help
                for rich_can in avl_dict.keys():
                    if avl_dict[rich_can] > 0:
                        rich_can_gives = min(avl_dict[rich_can], needed_extra)
                        avl_dict[rich_can] -= rich_can_gives
                        total_investment_dict[rich_can] -= rich_can_gives
                        # Ballots with multiple choices
                        total_investment_dict[str(rich_can) + str(can)] = rich_can_gives
                        needed_extra -= rich_can_gives
                        
                if needed_extra > 0:
                    total_investment_dict[can] += needed_extra
                    
        t = strt[rd + 1][0]
        amount_spent = sum(total_investment_dict[key] for key in total_investment_dict.keys())
        
        if amount_spent > budget:
            return {}, amount_spent
                    
    return total_investment_dict, amount_spent


def reach_a_structure_check(candidates, main, sub, k, Q_new, ballot_counts, budget):
    """
    Given main and sub structure specifications, finds optimal investment to reach that structure.
    
    Args:
        candidates: List of candidate identifiers
        main: Main structure specification
        sub: Sub structure specification
        k: Number of seats
        Q_new: Updated quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Tuple of (success boolean, updated ballot counts, amount spent)
    """
    amount_check = 1
    aggre_v_dict = get_new_dict(ballot_counts)
    check_aggre_v_dict = deepcopy(aggre_v_dict)
    check_ballot_counts = deepcopy(ballot_counts)
    strt, stdt = create_structure(main, sub)
    
    # Iterate until no more investment is needed
    while amount_check > 0:
        DecodedDicts, strt, log_campaign_list, status_list = process_campaign_STV(
            candidates, main, sub, k, Q_new, check_aggre_v_dict, budget
        )

        if all(status_list) == True:  # If campaigning is feasible
            total_investment_dict, amount_check = smart_campaign(
                candidates, log_campaign_list, strt, stdt, Q_new, DecodedDicts, budget
            )
            
            # Update ballot counts with new investment
            check_ballot_counts = campaign_addition_dict_simple(
                total_investment_dict, candidates, check_ballot_counts
            )
            
            if amount_check > budget:
                return False, {}, 0
        else:
            return False, {}, 0
            
        check_aggre_v_dict = get_new_dict(check_ballot_counts)
        
    # Calculate total amount spent
    amount_spent = sum(check_aggre_v_dict[candidate] for candidate in candidates) - \
                   sum(aggre_v_dict[candidate] for candidate in candidates)
    
    return amount_spent < budget, check_ballot_counts, amount_spent


def flip_order_campaign(candidates, k, Q, ballot_counts, budget):
    """
    Finds minimum investment to flip the current election order.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Tuple of (updated counts dictionary, new quota, difference dictionary)
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    budget_list_flip = []
    campaigned_dict_list = []
    
    # Try each possible sub-structure
    for sub in sub_structures(candidates):
        status_final, check_aggre_v_dict, amount_spent = reach_a_structure_check(
            candidates, list(reversed(original_main)), sub, k, Q_new, aggre_v_dict, budget
        )
        
        if status_final == True:
            campaigned_dict_list.append(check_aggre_v_dict)
            budget_list_flip.append(amount_spent)
    
    if len(budget_list_flip) == 0:
        print('increase the budget')
        return {}, 0, {}
    else:        
        min_budget = min(budget_list_flip)
        print(budget_list_flip)
        min_index = budget_list_flip.index(min_budget)

        check_aggre_v_dict = campaigned_dict_list[min_index]
        new_ballot_counts = clean_aggre_dict_diff(check_aggre_v_dict)
        
        strt_new, stdt_new, collection = STV_optimal_result_simple(
            candidates, new_ballot_counts, 2, Q_new
        )
        new_main, new_sub = return_main_sub(strt_new)
        
        # Calculate vote difference
        C = {x: check_aggre_v_dict[x] - aggre_v_dict[x] for x in check_aggre_v_dict if x in aggre_v_dict}
        diff = {x: y for x, y in C.items() if y != 0}
        
        print('New votes to be added = ', clean_aggre_dict_diff(diff))
        print('original order = ', original_main, original_sub)
        print('new order = ', new_main, new_sub)
        print('budget used = ', min_budget)
        return check_aggre_v_dict, Q_new, clean_aggre_dict_diff(diff)


def reach_any_winners_campaign(candidates, k, Q, ballot_counts, budget):
    """
    Finds minimum investment required for each possible combination of winners.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Dictionary mapping winner combinations to [budget, additions]
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Try each possible combination of k winners
    for comb in combinations(candidates, k):
        if set(comb) != set(og_winners):
            main_set = str_for_given_winners(comb, candidates)
            
            for current_main in main_set:
                budget_list_flip = []
                campaigned_dict_list = []
                
                for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k):
                    status_final, check_ballot_counts, amount_spent = reach_a_structure_check(
                        candidates, current_main, sub, k, Q_new, ballot_counts, budget
                    )

                    if status_final == True:
                        campaigned_dict_list.append(check_ballot_counts)
                        budget_list_flip.append(amount_spent)

                if len(budget_list_flip) > 0:
                    min_budget = min(budget_list_flip)
                    min_index = budget_list_flip.index(min_budget)

                    check_ballot_counts = campaigned_dict_list[min_index]
                    check_aggre_v_dict = get_new_dict(check_ballot_counts)
                    
                    strt_new, stdt_new, collection = STV_optimal_result_simple(
                        candidates, check_ballot_counts, k, Q_new
                    )
                    new_main, new_sub = return_main_sub(strt_new)
                    
                    # Verify winners
                    if set(give_winners(new_main, k)) != set(comb):
                        print(set(give_winners(new_main, k)), set(comb))
                        print(strt_new)
                        print('error!')
                        
                    # Calculate vote difference
                    C = {
                        x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                        for x in check_aggre_v_dict if x in aggre_v_dict
                    }
                    diff = {x: y for x, y in C.items() if y > 0}
                    
                    if strats_frame.get(''.join(comb), [budget, {}])[0] > min_budget:
                        strats_frame[''.join(comb)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x: y for x, y in strats_frame.items() if y[0] >= 0}


def reach_any_order_campaign(candidates, k, Q, ballot_counts, budget):
    """
    Finds minimum investment needed to reach any possible structure.
    
    Args:
        candidates: List of candidate identifiers
        k: Number of seats
        Q: Quota value
        ballot_counts: Dictionary of ballot counts
        budget: Available budget
        
    Returns:
        Dictionary mapping orders to [budget, additions]
    """
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og = STV_optimal_result(candidates, k, Q, aggre_v_dict)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q + budget / (k + 1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    # Try each possible main structure
    main_set = main_structures(candidates)
    
    for current_main in main_set:
        budget_list_flip = []
        campaigned_dict_list = []
        
        for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k):
            status_final, check_ballot_counts, amount_spent = reach_a_structure_check(
                candidates, current_main, sub, k, Q_new, ballot_counts, budget
            )

            if status_final == True:
                campaigned_dict_list.append(check_ballot_counts)
                budget_list_flip.append(amount_spent)

        if len(budget_list_flip) > 0:
            min_budget = min(budget_list_flip)
            min_index = budget_list_flip.index(min_budget)

            check_ballot_counts = campaigned_dict_list[min_index]
            check_aggre_v_dict = get_new_dict(check_ballot_counts)
            
            strt_new, stdt_new = STV_optimal_result(candidates, k, Q_new, check_aggre_v_dict)
            new_main, new_sub = return_main_sub(strt_new)
            
            if current_main != new_main:
                print('error')
           
            # Calculate vote difference
            C = {
                x: check_aggre_v_dict[x] - aggre_v_dict[x] 
                for x in check_aggre_v_dict if x in aggre_v_dict
            }
            diff = {x: y for x, y in C.items() if y > 0}
            
            if strats_frame.get(''.join(new_main), [budget, {}])[0] > min_budget:
                strats_frame[''.join(new_main)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x: y for x, y in strats_frame.items() if y[0] >= 0}