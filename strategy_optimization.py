from copy import deepcopy
from utils import get_new_dict, clean_aggre_dict_diff, create_structure, give_winners, str_for_given_winners, return_main_sub, main_structures, sub_structures, sub_structures_at_most_k_ones_fixed_last, campaign_addition_dict_simple
from processing import roundupdate, decode_dict, STV_optimal_result
from STVandIRV_results import STV_optimal_result_simple
from itertools import combinations
import math



#  Returns the campaign investment needed for a round, given budget. Returns false, if this is not possible.
def add_campaign(log_campaign_list, main_st,  remaining_candidates, decoded_dict, Q, k, t, stdt, budget):
    current_campaign_dict = {j:0 for j in main_st} 
          
    if stdt[t]==1:
        v = []
        for candidate in remaining_candidates:
            v.append(decoded_dict[candidate])
        t_update = max(0,(max(v) - decoded_dict[t])+1, (Q - decoded_dict[t])+1)
        if t_update>budget:
            return {}, False
        current_campaign_dict[t] = t_update
    else: 
        for candidate in remaining_candidates:
            
            if decoded_dict[candidate] >= Q:
                
                return {}, False
            else:
                can_update = max(0, decoded_dict[t] - decoded_dict[candidate]+1)
                if can_update>budget:
                    return {}, False
                current_campaign_dict[candidate] = can_update 
            
    log_campaign_list.append(deepcopy(current_campaign_dict))

    return log_campaign_list, True

# Given a structure and voter data and an updated Q, this gives a round-wise campaign allocation requirement to 
# make the structure feasible. Provides fixed addition to be made in each round, by calling add_campaign.
# This function doesn't utilitze the budget to fulfil the structural requirements. Just lists those.
def process_campaign_STV(candidates, main, sub, k, Q, aggre_v_dict, budget):    
    
    strt, stdt = create_structure(main, sub)
    currentdict  = {can: [[can]] for can in candidates}

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    
    CheckDicts = [] # list of additions each round should have, with each round addition in dict format for candidates
    DecodedDicts = [] # list of how votes look according to given main and sub structure, in each round
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates)-1):
        
        # for i'th round, takes the currentdict and updates according to current round of elimination/win
        currentdict = roundupdate(stdt, checked_candidates, remaining_candidates, currentdict)
        
        # once the variables are obtained for next round, this converts into numerical data
        decoded_dict = decode_dict(currentdict, candidates, Q, aggre_v_dict)

        
        # performs the next elimination/win and puts it in the checked_list
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        # checks if the next elimination/win is alright and if not, gives the addition that can make it right
        log_campaign_list, status = add_campaign(log_campaign_list, main, remaining_candidates, decoded_dict, Q, k, t, stdt, budget)
        if status==False:
            return {}, [], [], [False]
        status_list.append(status)
        DecodedDicts.append(decoded_dict)
        
    return DecodedDicts, strt, log_campaign_list, status_list

# Given a structure and voter data and an updated Q, this gives a round-wise campaign allocation requirement to 
# make the structure feasible. Provides fixed addition to be made in each round, by calling add_campaign.
# This function doesn't utilitze the budget to fulfil the structural requirements. Just lists those.
def process_campaign_STV_simple(candidates, main, sub, k, Q, ballot_counts, budget, collections):    
    
    strt, stdt = create_structure(main, sub)
    currentdict  = {can: [[can]] for can in candidates}

    remaining_candidates = list(stdt.keys())
    checked_candidates = []
    DecodedDicts = [] # list of how votes look according to given main and sub structure, in each round
    status_list = []
    log_campaign_list = []

    for i in range(len(candidates)-1):
        
        # # for i'th round, takes the currentdict and updates according to current round of elimination/win
        # currentdict = roundupdate(stdt, checked_candidates, remaining_candidates, currentdict)
        
        # # once the variables are obtained for next round, this converts into numerical data
        # decoded_dict = decode_dict(currentdict, candidates, Q, get_new_dict(ballot_counts))
        # print(decoded_dict)

        ballot_counts_at_time = collections[i][0]

        full_decoded_dict = get_new_dict(ballot_counts_at_time)
        decoded_dict = {cand: full_decoded_dict.get(cand, 0) for cand in list(stdt.keys())}
        #print(decoded_dict)
        
        # performs the next elimination/win and puts it in the checked_list
        t = remaining_candidates.pop(0) 
        checked_candidates.append(t)
        
        # checks if the next elimination/win is alright and if not, gives the addition that can make it right
        log_campaign_list, status = add_campaign(log_campaign_list, main, remaining_candidates, decoded_dict, Q, k, t, stdt, budget)
        if status==False:
            return {}, [], [], [False]
        status_list.append(status)
        DecodedDicts.append(decoded_dict)
        
    return DecodedDicts, strt, log_campaign_list, status_list



## This takes the round-wise requirement dict and returns the smart investment.
## This may not be very precise. Hence, we loop over this function later to final optimal investment.
## The problem arises when we invest in a losing candidate and latter rounds require investment in others to beat this
## candidate. Since we don't update log_campaign_dict while running this, we need loops outside the function.

def smart_campaign(candidates, log_campaign_list, strt, stdt, Q, DecodedDicts, budget):
    
    log = deepcopy(log_campaign_list)
    dict1 = log[0]  #for later distributing in the first round, where everyone gets what needed
    invest_dict = {can: 0 for can in candidates} #investment the candidates have in current rounds.
    total_investment_dict = {can: 0 for can in candidates} #total investment until this round
    
    #old investment in checked candidates, currently available for further dissipation 
    avl_dict = {can: 0 for can in candidates} 
    
    
    for can in dict1.keys(): #first round everyone gets what needed
        if dict1[can]>0:
            invest_dict[can] = dict1[can]
            total_investment_dict[can] = dict1[can]
             
    t = strt[0][0]      ## the first candidate for elimination or win
    decoded_dict = DecodedDicts[0]
    
    for rd in range(len(log)-1): # next rounds, with possible re-allocation of investment for efficiency 
        
        #first update the log of checked candidates. Dissipate investment into available dict.
        if stdt[t] == 1 and invest_dict[t]>0:
            dee_invest = deepcopy(invest_dict)
            avl_dict[t] = math.floor((decoded_dict[t]+ dee_invest[t]-Q)*(dee_invest[t]/(decoded_dict[t]+dee_invest[t])))
            invest_dict[t] = 0
            
        if stdt[t] == 0 and invest_dict[t]>0:
            for rich_can in total_investment_dict.keys():
                if rich_can[len(rich_can)-1]==t: #for all investments that are currently owned by t
                    avl_dict[rich_can] = deepcopy(total_investment_dict)[rich_can]
                
            invest_dict[t] = 0
        
        # Then, process the next checking of candidates.
        dict_rd = log[rd+1] #current round requirement 
        decoded_dict = DecodedDicts[rd+1] #this is needed for the updated votes info
        for can in dict_rd.keys():
            if dict_rd[can]>invest_dict[can]:
                needed_extra =  dict_rd[can] - invest_dict[can]
                invest_dict[can] = dict_rd[can] #investment currently the candidate has
                for rich_can in avl_dict.keys(): #check if old checked candidates can help
                    if avl_dict[rich_can]>0:  
                        rich_can_gives = min(avl_dict[rich_can], needed_extra)
                        avl_dict[rich_can] = avl_dict[rich_can] - rich_can_gives
                        total_investment_dict[rich_can] = total_investment_dict[rich_can]-rich_can_gives
                        total_investment_dict[str(rich_can)+str(can)] = rich_can_gives # ballots with >1 choices.
                        needed_extra = needed_extra - rich_can_gives
                if needed_extra>0:
                    total_investment_dict[can] = total_investment_dict[can]+needed_extra
        t = strt[rd+1][0]   
        amount_spent = sum(total_investment_dict[key] for key in total_investment_dict.keys())   
        if amount_spent >budget:
            return {}, amount_spent
                    
    return total_investment_dict, amount_spent



### Given main and sub str specifications, this function gives the updated dict and amount that does the job.
### Uses the smart campaign function to find optimal investment dict and uses addition to give final dict.
### Uses the while loop to repeatedly do the additions, until the given str becomes optimal. 

def reach_a_structure_check(candidates, main, sub, k, Q_new, ballot_counts, budget):
    amount_check = 1
    aggre_v_dict = get_new_dict(ballot_counts)
    check_aggre_v_dict = deepcopy(aggre_v_dict)
    check_ballot_counts = deepcopy(ballot_counts)
    #results_list , results_dict, collections = STV_optimal_result_simple(candidates, ballot_counts, k, Q_new)
    strt, stdt = create_structure(main, sub)
    
    while amount_check > 0: #until the additional amount that needs to be spent becomes zero
        
        DecodedDicts, strt, log_campaign_list, status_list = process_campaign_STV(candidates, main, sub, k, Q_new, check_aggre_v_dict, budget)
        #DecodedDicts, strt, log_campaign_list, status_list = process_campaign_STV_simple(candidates, main, sub, k, Q_new, check_ballot_counts, budget, collections)

        if all(status_list)==True: # proceed only if campaigning is feasible. 
            total_investment_dict, amount_check = smart_campaign(candidates, log_campaign_list, strt, stdt, Q_new, DecodedDicts, budget)
            #check_aggre_v_dict = campaign_addition_dict(total_investment_dict, candidates, check_aggre_v_dict)
            check_ballot_counts = campaign_addition_dict_simple(total_investment_dict, candidates, check_ballot_counts)            
            if amount_check>budget:
                return False, {}, 0
      
        else:
            return False, {}, 0
    #total amount spent in the process  
        check_aggre_v_dict = get_new_dict(check_ballot_counts)   
    amount_spent = sum(check_aggre_v_dict[candidate] for candidate in candidates) - sum(aggre_v_dict[candidate] for candidate in candidates)
    
    return amount_spent<budget, check_ballot_counts, amount_spent


### This function uses the 'reach_a_structure_check' function and goes over all sub structures to find the minimum amount
### to reach the flipped order. Can be generalized to reach any desired main str. 

def flip_order_campaign(candidates, k, Q, ballot_counts, budget):
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q+budget/(k+1)
    budget_list_flip= []
    campaigned_dict_list = []
    for sub in sub_structures(candidates):

        status_final, check_aggre_v_dict, amount_spent = reach_a_structure_check(candidates, list(reversed(original_main)), sub, k, Q_new, aggre_v_dict, budget)
        
        if status_final==True:
            
            campaigned_dict_list.append(check_aggre_v_dict)
            budget_list_flip.append(amount_spent)
    
    if len(budget_list_flip)==0:
        print('increase the budget')
        return {}, 0, {}
    else:        
        min_budget = min(budget_list_flip)
        print(budget_list_flip)
        min_index=budget_list_flip.index(min_budget)

        check_aggre_v_dict =  campaigned_dict_list[min_index]
        new_ballot_counts = clean_aggre_dict_diff(check_aggre_v_dict)
        
        strt_new, stdt_new, collection = STV_optimal_result_simple(candidates, new_ballot_counts, 2, Q_new)
        new_main, new_sub = return_main_sub(strt_new)
        C = {x: check_aggre_v_dict[x] - aggre_v_dict[x] for x in check_aggre_v_dict if x in aggre_v_dict}
        diff = {x:y for x,y in C.items() if y!=0}
        
        print('New votes to be added = ', clean_aggre_dict_diff(diff))
        print('original order = ', original_main, original_sub)
        print('new order = ', new_main, new_sub)
        print('budget used = ', min_budget)
        return check_aggre_v_dict, Q_new, clean_aggre_dict_diff(diff)


## This function finds minimum additions required for each combination of top k positions, i.e., winners

def reach_any_winners_campaign(candidates, k, Q, ballot_counts, budget):
    aggre_v_dict = get_new_dict(ballot_counts)
    #strt_og, stdt_og = STV_optimal_result(candidates, k, Q, aggre_v_dict)
    strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q+budget/(k+1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    for comb in combinations(candidates, k):
      
      if set(comb)!=set(og_winners):  
        main_set = str_for_given_winners(comb, candidates)
        #strats_frame[''.join(comb)] = [-Q_new, []]
    
        for current_main in main_set:
            budget_list_flip= []
            campaigned_dict_list = []
            for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k): #sub_structures(candidates)

                status_final, check_ballot_counts, amount_spent = reach_a_structure_check(candidates, current_main, sub, k, Q_new, ballot_counts, budget)


                if status_final==True:

                    campaigned_dict_list.append(check_ballot_counts)
                    budget_list_flip.append(amount_spent)

            if len(budget_list_flip)>0:       
                min_budget = min(budget_list_flip)
              
                min_index=budget_list_flip.index(min_budget)

                check_ballot_counts =  campaigned_dict_list[min_index]
                check_aggre_v_dict = get_new_dict(check_ballot_counts)
                #strt_new, stdt_new = STV_optimal_result(candidates, k, Q_new, check_aggre_v_dict)
                strt_new, stdt_new, collection = STV_optimal_result_simple(candidates, check_ballot_counts, k, Q_new)
                new_main, new_sub = return_main_sub(strt_new)
                if set(give_winners(new_main, k))!=set(comb):
                    print(set(give_winners(new_main, k)),set(comb))
                    print(strt_new)
                    print('error!')
                C = {x: check_aggre_v_dict[x] - aggre_v_dict[x] for x in check_aggre_v_dict if x in aggre_v_dict}
                diff = {x:y for x,y in C.items() if y>0}
                if strats_frame.get(''.join(comb), [budget, {}])[0] > min_budget:
        
                    strats_frame[''.join(comb)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x:y for x,y in strats_frame.items() if y[0]>=0}


## This function finds minimum additions needed to reach any structure. Produces a dict of all reachable orders 
# within the specified budget.

def reach_any_order_campaign(candidates, k, Q, ballot_counts, budget):
    aggre_v_dict = get_new_dict(ballot_counts)
    strt_og, stdt_og = STV_optimal_result(candidates, k, Q, aggre_v_dict) 
    #strt_og, stdt_og, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    original_main, original_sub = return_main_sub(strt_og)
    Q_new = Q+budget/(k+1)
    
    og_winners = give_winners(original_main, k)
    strats_frame = {}
    strats_frame[''.join(og_winners)] = [0, []]
    
    main_set = main_structures(candidates)
    
    for current_main in main_set:
        budget_list_flip= []
        campaigned_dict_list = []
        for sub in sub_structures_at_most_k_ones_fixed_last(candidates, k): #sub_structures(candidates)

            status_final, check_ballot_counts, amount_spent = reach_a_structure_check(candidates, current_main, sub, k, Q_new, ballot_counts, budget)

            if status_final==True:

                campaigned_dict_list.append(check_ballot_counts)
                budget_list_flip.append(amount_spent)


        if len(budget_list_flip)>0:       
            min_budget = min(budget_list_flip)
        
            min_index=budget_list_flip.index(min_budget)

            check_ballot_counts =  campaigned_dict_list[min_index]
            check_aggre_v_dict = get_new_dict(check_ballot_counts)
            strt_new, stdt_new = STV_optimal_result(candidates, k, Q_new, check_aggre_v_dict)
            #strt_new, stdt_new, collection = STV_optimal_result_simple(candidates, check_ballot_counts, k, Q_new)
            new_main, new_sub = return_main_sub(strt_new)
            if current_main != new_main:
                print('error')
           
            C = {x: check_aggre_v_dict[x] - aggre_v_dict[x] for x in check_aggre_v_dict if x in aggre_v_dict}
            diff = {x:y for x,y in C.items() if y>0}
            if strats_frame.get(''.join(new_main), [budget, {}])[0] > min_budget:

                strats_frame[''.join(new_main)] = [min_budget, clean_aggre_dict_diff(diff)]
           
    return {x:y for x,y in strats_frame.items() if y[0]>=0}