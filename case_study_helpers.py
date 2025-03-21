from STVandIRV_results import STV_optimal_result_simple
from strategy_optimization import reach_any_winners_campaign
from candidate_removal import remove_irrelevent, strict_support
import time
from copy import deepcopy
import utils
import os
import pandas as pd
import json


# ============================================================================
# BALLOT COUNTING FUNCTIONS
# ============================================================================

def get_ballot_counts_df(candidates_mapping, df):
    """
    Convert ranked choice dataframe to ballot counts dictionary for standard format.
    
    Parameters:
    candidates_mapping (dict): Mapping from candidate names to identifiers
    df (DataFrame): DataFrame containing voter choices
    
    Returns:
    dict: Counts of each ballot type
    """
    # Initialize a dictionary to store the counts of each ballot type as strings
    ballot_counts = {}
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Initialize an empty list to store valid candidates
        valid_candidates = []
        
        # Iterate through columns Choice_1 to Choice_6
        for i in range(1, 7):
            candidate = row[f'Choice_{i}']
            if candidate in candidates_mapping.keys():
                valid_candidates.append(candidates_mapping[candidate])
        
        # Check if the ballot is empty (no valid candidates)
        if valid_candidates:  # Skip empty ballots
            ballot_type = ''.join(valid_candidates)

            # Add the weight to the count for this ballot type in the dictionary
            if ballot_type not in ballot_counts:
                ballot_counts[ballot_type] = 1
            else:
                ballot_counts[ballot_type] += 1
                
    return ballot_counts


def get_ballot_counts_df_republican_primary(candidates_mapping, df):
    """
    Convert ranked choice dataframe to ballot counts dictionary for Republican primary format.
    
    Parameters:
    candidates_mapping (dict): Mapping from candidate names to identifiers
    df (DataFrame): DataFrame containing voter choices with weights
    
    Returns:
    dict: Weighted counts of each ballot type
    """
    # Initialize a dictionary to store the counts of each ballot type as strings
    ballot_counts = {}
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Initialize an empty list to store valid candidates
        valid_candidates = []

        # Iterate through columns rank1 to rank13
        for i in range(1, 14):
            candidate = row[f'rank{i}']
            valid_candidates.append(candidates_mapping[candidate])

        ballot_type = ''.join(valid_candidates)

        # Use the 'weight' column to determine the number of voters for this ballot type
        weight = row['weight']

        # Add the weight to the count for this ballot type in the dictionary
        if ballot_type not in ballot_counts:
            ballot_counts[ballot_type] = weight
        else:
            ballot_counts[ballot_type] += weight
    
    return ballot_counts


# ============================================================================
# BALLOT PROCESSING FUNCTIONS
# ============================================================================

def process_ballot_counts_post_elim(ballot_counts, k, candidates, elim_cands, check_strats=False, 
                                   budget_percent=0, check_removal_here=False, keep_at_least=8):
    """
    Process ballot counts after eliminating certain candidates.
    
    Parameters:
    ballot_counts (dict): Dictionary of ballot counts
    k (int): Number of winners to select
    candidates (list): List of candidate identifiers
    elim_cands (list): List of candidates to eliminate
    check_strats (bool): Whether to check strategies
    budget_percent (float): Budget percentage for strategy calculation
    check_removal_here (bool): Whether to check candidate removal
    keep_at_least (int): Minimum number of candidates to keep
    
    Returns:
    None: Results are printed rather than returned
    """
    # Initialize a dictionary for filtered data
    filtered_data = {}
    elim_strings = ''.join(char for char in elim_cands)

    # Remove eliminated candidates while retaining the rest of the string
    for key, value in ballot_counts.items():
        new_key = ''.join(char for char in key if char not in elim_strings)
        filtered_data[new_key] = filtered_data.get(new_key, 0) + value
    filtered_data.pop('', None)

    # Get remaining candidates
    elec_cands = [cand for cand in candidates if cand not in elim_cands]
    
    # Calculate vote counts
    full_aggre_v_dict = utils.get_new_dict(ballot_counts)
    aggre_v_dict = utils.get_new_dict(filtered_data)

    # Calculate quota
    Q = round(sum(full_aggre_v_dict[cand] for cand in candidates)/(k+1)+1, 3)
    print("\n" + "="*50)
    print(f"Q = {Q}")
    print("="*50 + "\n")
    
    # Calculate strict support within eliminated candidates
    letter_counts = {}
    for key, value in ballot_counts.items():
        i = 0
        newset = []
        if key and key[i] in candidates:
            while key[i] in elim_strings:
                newset.append(key[i])
                i=i+1
                if i >= len(key):
                    break

        new_key = ''.join(char for char in newset)
        for letter in new_key:
            if letter in letter_counts:
                letter_counts[letter] += value
            else:
                letter_counts[letter] = value

    # Print vote counts after elimination
    print("\n" + "-"*50)
    print(f"Total votes if {elim_cands} are eliminated:")
    print("-"*50)
    
    wins_during_elims = []
    for c in elec_cands:
        print(f"  {c}: {aggre_v_dict[c]}")
        if aggre_v_dict[c] >= Q:
            wins_during_elims.append(c)

    if len(wins_during_elims) > 0:
        print("\nWinners during elimination of lower group:")
        print(f"  {wins_during_elims}")
    print("-"*50 + "\n")

    # Print strict support
    print("-"*50)
    print(f"Strict support within {elim_cands}:")
    for letter, count in letter_counts.items():
        print(f"  {letter}: {count}")
    
    best_c_irrelevant = max(letter_counts, key=letter_counts.get) if letter_counts else None
    if best_c_irrelevant:
        print(f"\nMax strict support is with: {best_c_irrelevant}")
    print("-"*50 + "\n")

    # Run STV to determine winners
    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, subresults = utils.return_main_sub(rt)
    print("="*50)
    print(f"Overall winning order is: {results}")
    print("="*50 + "\n")

    # Check candidate removal if requested
    budget = budget_percent * sum(full_aggre_v_dict[cand] for cand in candidates) * 0.01
    if check_removal_here:
        candidates_reduced, group_remaining, stop = remove_irrelevent(
            ballot_counts, rt, results[:keep_at_least], budget, ''.join(candidates)
        )
        
        print("-"*50)
        if stop:
            print(f"We can remove {group_remaining} and keep {candidates_reduced}")
        else: 
            print("We cannot remove any more candidates")
        print("-"*50 + "\n")

    # Check strategies if requested
    if check_strats:
        print("="*50)
        print(f"Checking strategies for {elec_cands}")
        start = time.time()
        strats_frame = reach_any_winners_campaign(elec_cands, k, Q, filtered_data, budget)
        end = time.time()
        print(f"Total time: {end - start:.2f} seconds")
        print("\nStrategy frame:")
        print(strats_frame)
        print("="*50)


# ============================================================================
# BOOTSTRAP FUNCTIONS
# ============================================================================

def generate_bootstrap_samples(data, n_samples=1000, save=False):
    """
    Generates bootstrap samples from the given dataset.

    Parameters:
    data (DataFrame): The original dataset containing RCV rankings.
    n_samples (int): The number of bootstrap samples to generate.
    save (bool): Whether to save the bootstrap samples.

    Returns:
    List[DataFrame]: A list containing the bootstrap samples.
    """
    bootstrap_samples = []
    n = len(data)
    for _ in range(n_samples):
        sample = data.sample(n, replace=True)  # Sampling with replacement
        bootstrap_samples.append(sample)

    if save:
        # Creating a directory to store all the bootstrap sample files
        output_dir = 'bootstrap_samples_new'
        os.makedirs(output_dir, exist_ok=True)

        # Saving each bootstrap sample as a separate CSV file
        for i, sample in enumerate(bootstrap_samples):
            sample_file_path = os.path.join(output_dir, f'bootstrap_sample_{i+1}.csv')
            sample.to_csv(sample_file_path, index=False)
        
    return bootstrap_samples


def process_bootstrap_samples(k, candidates_mapping, bootstrap_samples_dir, bootstrap_files, 
                              budget_percent, keep_at_least, iters=10, loopy_removal=False, 
                              want_strats=False, save=False, spl_check=False):
    """
    Process multiple bootstrap samples to test algorithm efficiency.
    
    Parameters:
    k (int): Number of winners to select
    candidates_mapping (dict): Mapping from candidate names to identifiers
    bootstrap_samples_dir (str): Directory containing bootstrap samples
    bootstrap_files (list): List of bootstrap sample filenames
    budget_percent (float): Budget percentage for strategy calculation
    keep_at_least (int): Minimum number of candidates to keep
    iters (int): Maximum number of iterations
    loopy_removal (bool): Whether to try decreasing keep_at_least
    want_strats (bool): Whether to calculate strategies
    save (bool): Whether to save results
    spl_check (bool): Whether to do special check
    
    Returns:
    tuple: (algo_works, data_samples) - Algorithm success count and collected data
    """
    candidates = list(candidates_mapping.values())
    it = 0
    algo_works = 0
    data_samples = []
    
    for file in bootstrap_files:
        start = time.time()
        
        stop = False
        it += 1
    
        if it > iters:
            break

        sample_file_path = os.path.join(bootstrap_samples_dir, file)
        df = pd.read_csv(sample_file_path)
    
        # Determine ballot counting method based on column names
        if 'rank1' in df.columns:
            ballot_counts = get_ballot_counts_df_republican_primary(candidates_mapping, df)
            Q = 800
            budget = budget_percent * sum(ballot_counts.values()) * 0.01
        else:
            ballot_counts = get_ballot_counts_df(candidates_mapping, df)
            Q = round(sum(ballot_counts.values())/(k+1)+1, 3) 
            budget = budget_percent * sum(ballot_counts.values()) * 0.01
    
        # Compute results using STV
        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
        results, subresults = utils.return_main_sub(rt)

        # Handle candidate removal based on approach
        if not loopy_removal:
            candidates_reduced, group, stop = remove_irrelevent(
                ballot_counts, rt, results[:keep_at_least], budget, ''.join(candidates)
            )
        else:
            # Start with original keep_at_least value and decrease until removal fails
            current_keep = keep_at_least
            while current_keep >= k:
                candidates_reduced, group, stop = remove_irrelevent(
                    ballot_counts, rt, results[:current_keep], budget, ''.join(candidates)
                )

                if not stop:
                    # If removal failed, revert to previous successful value
                    current_keep += 1
                    candidates_reduced, group, stop = remove_irrelevent(
                        ballot_counts, rt, results[:current_keep], budget, ''.join(candidates)
                    )
                    break
                current_keep -= 1
        
        print(f"Iteration {it}: Candidates = {candidates_reduced}")

        if stop:
            algo_works += 1

            if want_strats:
                # Initialize a dictionary for filtered data
                filtered_data = {}

                # Remove eliminated candidates while retaining the rest of the string
                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in group)
                    filtered_data[new_key] = filtered_data.get(new_key, 0) + value
                filtered_data.pop('', None)

                agg_v_dict = utils.get_new_dict(filtered_data)

                # Check for immediate winners
                for cand_winner in candidates_reduced:
                    if agg_v_dict[cand_winner] >= Q and k > 1:   
                        removal_permitted = permit_STV_removal(
                            cand_winner, ballot_counts, Q, candidates_reduced, 
                            group, budget_percent, spl_check=spl_check
                        )
                        
                        if not removal_permitted:
                            print(f"Error. Candidate {cand_winner} wins during elimination")
                        else:
                            small_election_number = len(candidates) - len(candidates_reduced)
                            ballot_counts_short = collection[small_election_number][0]
                            test = [rt[i][0] for i in range(small_election_number, len(rt))]
                            ordered_test = sorted(test, key=lambda x: results.index(x))
                            strats_frame = reach_any_winners_campaign(
                                ordered_test, k, Q, ballot_counts_short, budget
                            )
                            break
                
                # If no immediate winners, check strategies for all remaining candidates
                if all(agg_v_dict[cand_winner] < Q for cand_winner in candidates_reduced):
                    strats_frame = reach_any_winners_campaign(
                        candidates_reduced, k, Q, filtered_data, budget
                    )
                
                data_samples.append(strats_frame)

                # Save results if requested
                if save:
                    output_dir = 'strategy_optimization_results'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    save_path = os.path.join(output_dir, f"iteration_{it}.json")
                    with open(save_path, "w") as f:
                        json.dump({"iteration": it, "strats_frame": strats_frame}, f, indent=4)

                    print(f"Iteration {it}: strats_frame = {strats_frame}")
            
        end = time.time()    
        # print('Total time =', end - start)
    
    print(algo_works, it-1, ' Removal Efficiency is ', algo_works/(it-1)*100, '%')
    return algo_works, data_samples


# ============================================================================
# PERMITTING FUNCTIONS
# ============================================================================

def permit_STV_removal(cand_winner, ballot_counts, Q, candidates_reduced, group, 
                      budget_percent, spl_check=False):
    """
    Check if a candidate's removal is permitted under STV rules.
    
    Parameters:
    cand_winner (str): Winning candidate identifier
    ballot_counts (dict): Dictionary of ballot counts
    Q (float): Quota value
    candidates_reduced (list): Reduced list of candidates
    group (str): Group of candidates
    budget_percent (float): Budget percentage
    spl_check (bool): Whether to do special check
    
    Returns:
    bool: Whether removal is permitted
    """
    surplusA = {}
    A_original = utils.get_new_dict(ballot_counts)[cand_winner]
    A_needs = Q - A_original
    candidates_still_in_elec = ''.join([c for c in candidates_reduced if c != cand_winner])
    
    if spl_check:
        probable_elec = min(candidates_still_in_elec, key=lambda x: utils.get_new_dict(ballot_counts)[x])
    else: 
        probable_elec = ''
        
    groupcopy = group + probable_elec
    groupcopy2 = deepcopy(groupcopy)

    # Calculate surplus for each candidate in group
    for cand in group:
        fil_data = {}
        for key, value in ballot_counts.items():
            new_key = ''.join(char for char in key if char not in groupcopy.replace(cand, ""))
            fil_data[new_key] = fil_data.get(new_key, 0) + value
        fil_data.pop('', None)
        
        C_set = {}
        for cand_spl in candidates_still_in_elec.replace(probable_elec, ""):
            for key, value in ballot_counts.items():
                if key[0] in groupcopy2.replace(cand, ""):
                    if cand_spl in key:
                        if cand_winner in key and key.index(cand_winner) > key.index(cand_spl):
                            C_set[cand_spl] = C_set.get(cand_spl, 0) + value
        
        AtransfersL = 0 
        for key, value in fil_data.items():
            if key[0] == cand_winner:
                if len(key) > 1 and key[1] in groupcopy:
                    AtransfersL = AtransfersL + value

        SV0 = min(10, utils.get_new_dict(fil_data)[cand_winner] - Q) 
        min_from_elec_cands = min([
            utils.get_new_dict(fil_data)[cande] + C_set.get(cande, 0) 
            for cande in candidates_still_in_elec.replace(probable_elec, "")
        ])
        surplusA[cand] = SV0 * (AtransfersL) / (SV0 + A_original) - min_from_elec_cands + A_needs
    
    # Calculate final dictionaries for decision
    final_dict = {}
    another_dict = {}
    for cand in group:
        letter_counts = {}
        
        # Calculate strict support
        for key, value in ballot_counts.items():
            i = 0
            newset = []
            if key and key[0] not in [cand_winner]:
                while key[i] in groupcopy2 + cand_winner:
                    newset.append(key[i])
                    i = i + 1
                    if i >= len(key):
                        break

            new_key = ''.join(char for char in newset)
            if cand in new_key:
                if cand in letter_counts:
                    letter_counts[cand] += value
                else:
                    letter_counts[cand] = value
                    
        final_dict[cand] = -round(surplusA[cand] + letter_counts[cand], 2)
        another_dict[cand] = 2 * (
            letter_counts[cand] - strict_support(ballot_counts, group, '', cand)
        )
    
    # Make final decision
    maxdev = min(final_dict, key=final_dict.get)   
    dev_in_percent = round(final_dict[maxdev] / sum(ballot_counts.values()) * 100, 2)
    print(dev_in_percent)
    
    return dev_in_percent > budget_percent