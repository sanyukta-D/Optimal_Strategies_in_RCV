from copy import deepcopy
from utils import common_member, make_perms_transfer

### Core functions of STV elections- coding up information in round-update, decoding later from their format

# After a round result (W/L) is reflected in checked_candidates, 
# this step updates the remaining vote banks by transfering the unused votes
def roundupdate(structure, checked_candidates, remaining_candidates, olddict):
    if len(checked_candidates)==0:
        return olddict
    currentdict = deepcopy(olddict)
    last_candidate = checked_candidates[-1] #depends on whether the last result is W or L, update changes
    
    perms_checked_candidates = make_perms_transfer(checked_candidates)
    
    stvotelist = currentdict[last_candidate] #last candidate's total vote bank
    for I_c in stvotelist:  #each set of votes that has a separate path to the last candidate
        I = deepcopy(I_c) #copying so that the Q updates don't change the original list accounts
        q = []
        while I[-1][0]==1: #we remove the lists with [1,..] from vote banks because these are fractions
            q1 = I.pop()
            q = q+[q1] #all removed fractions are stored here and appended later
        for j in remaining_candidates: # the vote banks of remaining candidates are updated here
            for l in perms_checked_candidates: #for each permutation of checked candidates that gets appended in middle
                if common_member(I, l) == False: #making sure the permutation candidates and the sets are distinct
                    if structure[last_candidate]==0: #last round is an elimination
                        Inew = I + l +[j]+ q 
                    else:    # last round is a win
                        Inew = I + l +[j] + q + [[1, 'Q', stvotelist]] #append a list for the surplus fraction
                    currentdict[j] =  currentdict[j]+[Inew]
    return currentdict


##Used to decode the previously coded descriptions in round_update. Gives dict containing numbers for current round.
# using aggregated data, converts each path into a number that measures the votes this path contributes
def decode_list(pathcopy, Q, aggre_v_dict):
    path = deepcopy(pathcopy)
    if path[-1][0]!=1:
        value =  aggre_v_dict.get(''.join(path), 0)  
        return value
    Qlist = []
    path2 = deepcopy(path)
    while path2[-1][0]==1:
        q_item = path2.pop()
        Qlist.append(q_item)
    tot = decode_list(path2, Q, aggre_v_dict)
    for q_item in Qlist:
        value = 0
        for part in q_item[-1]:
            value = value + decode_list(part, Q, aggre_v_dict)
        tot = tot *max(0, ( 1 - Q/(value+0.001)))
        #tot = tot *( 1 - Q/value)
    return tot


# using aggregated data, converts a dict with complete variable description to numerical data
def decode_dict(currentdict, candidates, Q, aggre_v_dict):
    newcurrentdict = {}
    for candidate in candidates:
        currentdict2 = deepcopy(currentdict)
        v_list = currentdict2[candidate]
        newcurrentdict[candidate] = round(sum(decode_list(path, Q, aggre_v_dict) for path in v_list),3)

    return newcurrentdict
        

### Produce optimal STV result ###


# creates a short dict from decoded_dict-for only remaining candidates- and returns the next elimination/win
# used in finding optimal STV result
def give_next_candidate(decoded_dict, remaining_candidates, Q, rt, dt):    
    short_decoded_dict = {can: decoded_dict[can] for can in remaining_candidates}
        
    # performs the next elimination/win and puts it in the checked_list
    t1 = max(short_decoded_dict, key=short_decoded_dict.get)
    t2 = min(short_decoded_dict, key=short_decoded_dict.get)
    if decoded_dict[t1]>=Q:
        t = t1
        rt.append([t, 1])
        dt[t] = 1
    else:
        t = t2
        rt.append([t, 0])
        dt[t] = 0
    return t, rt, dt


# for a given voter data, this produces the resulting optimal structure
# a bit similar to process_structure_STV
def STV_optimal_result(candidates, k, Q, aggre_v_dict):    

    currentdict  = {can: [[can]] for can in candidates}

    remaining_candidates = deepcopy(candidates)
    checked_candidates = []
    dt = {}
    rt = []
    
    for i in range(len(candidates)):
       
        # for i'th round, takes the currentdict and updates according to current round of elimination/win
        currentdict = roundupdate(dt, checked_candidates, remaining_candidates, currentdict)
        
        # once the variables are obtained for next round, this converts into numerical data
        decoded_dict = decode_dict(currentdict, candidates, Q, aggre_v_dict)
        
        t, rt, dt = give_next_candidate(decoded_dict, remaining_candidates, Q, rt, dt)
        
        remaining_candidates.remove(t) 
        checked_candidates.append(t)
        
    return rt, dt