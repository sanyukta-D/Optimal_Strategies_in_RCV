from utils import get_new_dict
from operator import itemgetter

## To check if we can remove group, and retain candidates

def check_removal(candidates, group, ballot_counts, budget):
    #strict support check
    #group = 'FGHIJKLM'

    strict_support = {}

    for key, value in ballot_counts.items():
        i = 0
        newset = []
        while key[i] in group:
            newset.append(key[i])
            i=i+1
            if i >= len(key):
                break
        new_key = ''.join(char for char in newset)
        for letter in new_key:
            if letter in strict_support:
                strict_support[letter] += value
            else:
                strict_support[letter] = value
    can_remove = False
   
    for best_c_irrelevant in strict_support.keys():

        #best_c_irrelevant = max(strict_support, key=strict_support.get)
        groupcopy = group

        mostly_irrelevant = groupcopy.replace(best_c_irrelevant, "")

        # Initialize a dictionary for filtered data
        filtered_data = {}

        # Remove letters maybe_irrelevant {G, H, I, J, K, L}? while retaining the rest of the string
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in mostly_irrelevant)

            filtered_data[new_key] =   filtered_data.get(new_key, 0) + value
        filtered_data.pop('', None)

        aggre_v_dict = get_new_dict(filtered_data)

    
        relevant_aggre_dict = {}
        #candidates = ['A', 'B', 'C', 'D', 'E']

        for c in candidates:
            relevant_aggre_dict[c] =  aggre_v_dict[c]

        worst_c_relevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
        #print(int(relevant_aggre_dict[worst_c_relevant]-strict_support[best_c_irrelevant]))
        if int(relevant_aggre_dict[worst_c_relevant]-strict_support[best_c_irrelevant])+1>= budget:
            can_remove = True
        # else:
        #     can_remove = False
        #     return can_remove
        
        else: #again check if the addition really changes who drops out after E removal
            last_three= sorted(relevant_aggre_dict.items(), key=itemgetter(1))[:3] 
            #print(last_three, 'last three', group)
            #budget not enough to benefit 2 candidates at the bottom
            
            if 2*last_three[2][1]-last_three[1][1]-last_three[0][1] >budget:
                # Remove letters maybe_irrelevant {E, G, H, I, J, K, L}? while retaining the rest of the string

                maybe_irrelevant = mostly_irrelevant+ worst_c_relevant
                    # Initialize a dictionary for filtered data
                filtered_data = {}

                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in maybe_irrelevant)

                    filtered_data[new_key] =   filtered_data.get(new_key, 0) + value
                filtered_data.pop('', None)

                aggre_v_dict = get_new_dict(filtered_data)

                #add budget votes to F
                aggre_v_dict[best_c_irrelevant] = aggre_v_dict[best_c_irrelevant]+ budget

                # new candidate list that removes E and adds F
                relevant_aggre_dict = {}
                candidates_temp = candidates.copy()
                candidates_temp.remove(worst_c_relevant)
                candidates_temp.append(best_c_irrelevant)

                for c in candidates_temp:
                    relevant_aggre_dict[c] =  aggre_v_dict[c]

                new_best_c_irrelevant = min(relevant_aggre_dict, key=relevant_aggre_dict.get)
                #print(new_best_c_irrelevant, best_c_irrelevant)


                if new_best_c_irrelevant == best_c_irrelevant: #F still gets eliminated
                    can_remove = True
                    #return True
                else:
                    #print(new_best_c_irrelevant)
                    can_remove = False
                    return False
            else:
                return False
    return can_remove

def remove_irrelevent( ballot_counts, rt, startcandidates, budget, fullgroup):

    candidatesnew = startcandidates
    #candidatesnew = candidatesnew[:6]
    group = ''.join(char for char in fullgroup if char not in candidatesnew)
   
    while check_removal(candidatesnew, group, ballot_counts, budget)!=True:

        candidatesnew = candidatesnew[:-1]
        
        #print(candidatesnew)
        group = ''.join(char for char in fullgroup if char not in candidatesnew)
        if len(candidatesnew)<=2:
            stop =False
            return candidatesnew, group, stop
        

    return candidatesnew, group, check_removal(candidatesnew, group, ballot_counts, budget)
       

def strict_support(ballot_counts, lower_group, upper_group, candidate):

    letter_counts ={}
    total_cands = lower_group + upper_group
    
    #strict_support
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in total_cands:
            while key[i] in lower_group and key[i] not in upper_group:
                newset.append(key[i])
                i=i+1
                if i >= len(key):
                    break

        new_key = ''.join(char for char in newset)
        if candidate in new_key:
            if candidate in letter_counts:
                letter_counts[candidate] += value
            else:
                letter_counts[candidate] = value

    return letter_counts.get(candidate, 0)


def predict_wins(ballot_counts, candidates, k, Q, budget):
    C_W = []
    for cand in candidates:
        if budget + strict_support(ballot_counts,  candidates, [], cand)> Q+budget/(k+1):
            C_W.append(cand)

    C_bar_W = [cand for cand in candidates if cand not in C_W]
    number_unique_ballots = 0
    for cand in C_W:
        # Call strict_support only for new unique ballots
        c_new = C_bar_W +[cand]
        support_count = strict_support(ballot_counts, c_new, [], cand)
        
        number_unique_ballots += support_count  # Accumulate only unique contribution


    U_W = min (k, int(budget + number_unique_ballots)/(Q+budget/(k+1)))
    #print((budget + number_unique_ballots)/(Q+budget/(k+1)), number_unique_ballots)
    return U_W

def predict_losses(ballot_counts, candidates, k , Q, budget):
    aggre_v_dict = get_new_dict(ballot_counts)
    first_choice_votes = [aggre_v_dict[i] for i in candidates]
    first_choice_votes=sorted(first_choice_votes, reverse=True)
    # print(first_choice_votes)

    C_L =[]
    for cand in candidates:
        if budget + strict_support(ballot_counts,  candidates, [], cand) <= first_choice_votes[k-1]:
            C_L.append(cand)
    # print(C_L)

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
    T= sorted(T,reverse=True)
    
    i_L = 1
    if len(T)==0:
        return 0
    
    while sum(T[:i_L]) + first_choice_votes[0] < Q+budget/(k+1):
        i_L = i_L+1
        if i_L>=len(T):
            return i_L

    return i_L