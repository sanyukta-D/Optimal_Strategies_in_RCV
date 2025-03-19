from itertools import permutations, product
from copy import deepcopy
from collections import Counter
import math

### Basic operations- generating voter data, doing permutations etc

#a function for checking if two lists have any common elements
def common_member(a, b): 
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True)
    return(False)  

#given a list of candidates, this gives all possible orderings- n! in total
def main_structures(candidates):
    PotentialSets = [list(perm) for perm in permutations(candidates, len(candidates))]
    return PotentialSets

#returns a list of main strs for given list of winners
def str_for_given_winners(winners, candidates):
    losers = [item for item in candidates if item not in winners]
    PotentialSets=[]

    for perm_w in permutations(winners, len(winners)):
        for perm_l in permutations(losers, len(losers)):
            PotentialSets.append(list(perm_w)+list(perm_l))
    return PotentialSets

#returns a list of main strs for given list of losers
def str_for_given_losers(losers, candidates):
    winners = [item for item in candidates if item not in losers]
    PotentialSets=[]

    for perm_w in permutations(winners, len(winners)):
        for perm_l in permutations(losers, len(losers)):
            PotentialSets.append(list(perm_w)+list(perm_l))
    return PotentialSets
    
#returns a list of main strs for given list of winners and losers
def str_for_given_winners_losers(winners, candidates, losers):
    middle = [item for item in candidates if item not in winners or losers]
    PotentialSets=[]

    for perm_w in permutations(winners, len(winners)):
        for perm_l in permutations(losers, len(losers)):
            for perm_m in permutations(middle, len(middle)):
                PotentialSets.append(list(perm_w)+list(perm_m)+list(perm_l))
    return PotentialSets
    

#given a list of candidates, this gives all possible W/L round result types- 2^(n-1) in total
def sub_structures(candidates):
    l = len(candidates)-1
    results = list(product([0,1],repeat=l))
    roundresults = [list(ele)+[1] for ele in results]
    return roundresults

# Generate all possible combinations of 0 and 1 with z zeros at the beginning and (l - z) zeros in the middle
# thm that bounds the min number of initial losses
def sub_structures_specific(candidates, z):
    l = len(candidates) - 1
    
    prefix = [0] * z
    middle = list(product([0, 1], repeat=l - z))
    suffix = [1]
    
    # Combine the prefix, middle, and suffix to form the final combinations
    results = [prefix + list(ele) + suffix for ele in middle]
    
    return results

def sub_structures_at_most_k_ones_fixed_last(candidates, k):
    """
    Returns all binary sequences of length len(candidates)
    whose last bit is forced to 1 and which have at most k ones.
    """
    n = len(candidates)
    results = []
    # We'll generate sequences of length n-1, then append 1
    for bits in product([0, 1], repeat=n-1):
        # The total number of ones includes the appended 1
        if sum(bits) + 1 <= k:
            results.append(list(bits) + [1])
    return results

# given a main and sub structure, this gives a dict and a list showing who wins/loses in each round, in the order.
def create_structure(main, sub):
    maincopy = deepcopy(main)
    results = []
    redict = {} # mapping candidates to their results in the round order.
    for i in sub:
        if i == 0:
            res = maincopy.pop(-1)
            results.append([res, i])
        else:
            res = maincopy.pop(0)
            results.append([res, i])
        redict[res] = i     
    return results, redict

# given a list showing who wins/loses in each round, this gives the main and sub structure
def return_main_sub(results_list):
    strt = deepcopy(results_list)
    sub = [] # mapping candidates to their results in the round order.
    win=[]
    lose=[]
    for i in strt:
        if i[1]>0:
            win= win +[i[0]]
        else:
            lose= [i[0]] +lose
        sub.append(i[1])
    main = win+lose
    return main, sub


#while transferring from the last candidate to the current ones, we may pass those through checked candidates
#so here, we make permutations of checked candidates and add these paths between the transfers
def make_perms_transfer(candidates):
    perms_candidates = [] 
    for l in range(len(candidates)):
        perms = [list(perm) for perm in permutations(candidates, l)]
        perms_candidates = perms_candidates+perms
    return perms_candidates

# This is a general function for making all possible vote patterns of specified length, given the set of candidates. 
def make_perms(candidates, min_ballot_length=0):
    perms_candidates = [] 
    for l in range(min_ballot_length, len(candidates)):
        perms = [list(perm) for perm in permutations(candidates, l+1)]
        perms_candidates = perms_candidates+perms
    return perms_candidates

# initiates a dict of all possible winner types
def dict_perm_k(candidates, k):
    dict_new = {}
    for perm in permutations(candidates, k):
        dict_new[''.join(list(perm))] = 0
        
    return dict_new

# Returns the list of winners, given main str.
def give_winners(main_st, k):
    winners = main_st[0:k]
    return winners


### Basic dict operations- aggregation, non-aggregation, converting investment into complete aggr-data etc

# takes data in (type, num) format through lists and returns it in a dict format, but non-aggregated. 
def non_agg_dict(Ballottypes, NumVoters):
    stringballots = []
    for ballot in Ballottypes:
        ballotnew = ''.join(ballot)
        stringballots.append(ballotnew)

    non_aggre_voterdata_dict = dict(zip(stringballots, NumVoters))
    return non_aggre_voterdata_dict

#Take non-agg data and returns the complete aggr_dict.
#creates a dict with all V variables, where V_{AB} includes total votes that have first and second choices as A and B.
def agg_dict(candidates, non_aggre_voterdata):
    aggre_v_dict = Counter()
    permset = make_perms(candidates)

    # Calculate the aggregation using Counter
    for ballot in permset:
        for i in range(len(ballot)):
            v = ''.join(ballot[0:i+1])
            aggre_v_dict[v] += non_aggre_voterdata.get(''.join(ballot), 0)

    # Filter out keys with zero counts
    final_dict = {x: y for x, y in aggre_v_dict.items() if y > 0}
    return final_dict

# converts any investment or ballot counts into aggregated investment
def get_new_dict(my_dict):
    new_dict = {}
    
    # Iterate through the keys and values in my_dict
    for key, value in my_dict.items():
        # Generate all possible substrings of the key
        substrings = [key[:i+1] for i in range(len(key))]
        
        # Update the new_dict with the aggregated values for each substring
        for substring in substrings:
            new_dict[substring] = new_dict.get(substring, 0) + value
    
    # Filter out keys with zero counts
    final_dict = {x: y for x, y in new_dict.items() if y > 0}
    return final_dict


## Given the investment and the original aggr_dict, this returns the final aggr_dict.
def campaign_addition_dict(my_total_investment_dict, candidates, aggre_v_dict):
    
    my_total_investment_dict_d = deepcopy(my_total_investment_dict)
    aggre_v_dict_d = deepcopy(aggre_v_dict)
  
    agg_total_investment_dict = get_new_dict(my_total_investment_dict_d)

    for key in agg_total_investment_dict.keys():

        aggre_v_dict_d[key] = aggre_v_dict_d.get(key,0) + math.ceil(agg_total_investment_dict.get(key,0))
    
    return aggre_v_dict_d

## Given the investment and the original aggr_dict, this returns the final aggr_dict.
def campaign_addition_dict_simple(my_total_investment_count, candidates, ballot_counts):
    
    my_total_investment_count_d = deepcopy(my_total_investment_count)
    ballot_counts_d = deepcopy(ballot_counts)

    for key in my_total_investment_count_d.keys():

        ballot_counts_d[key] = ballot_counts_d.get(key,0) + math.ceil(my_total_investment_count_d.get(key,0))
    
    return ballot_counts_d

## Reverse of aggregation operation. Given an aggregated dict, this gives a non-agg dict.
def clean_aggre_dict_diff(anydict):
    cleandict=deepcopy(anydict)
    for key1 in cleandict.keys():
        for key2 in cleandict.keys():
            if key2[0:len(key1)]==key1 and key1!=key2:
               
                cleandict[key1] = cleandict[key1]-cleandict[key2]
    final_dict = {x:y for x,y in cleandict.items() if y>0}
    return final_dict

