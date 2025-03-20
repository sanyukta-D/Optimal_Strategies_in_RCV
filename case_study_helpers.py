from STVandIRV_results import STV_optimal_result_simple
from strategy_optimization import reach_any_winners_campaign
from candidate_removal import remove_irrelevent
import time 
import utils
import os
import pandas as pd
import json

def get_ballot_counts_df(candidates_mapping, df):
    # Initialize a dictionary to store the counts of each ballot type as strings
    ballot_counts = {}
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Initialize an empty list to store valid candidates
        valid_candidates = []
        
        # Iterate through columns rank1 to rank5
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
    #print('Total votes: ', sum(ballot_counts.values()))
    return ballot_counts

def get_ballot_counts_df_republican_primary(candidates_mapping, df):
    # Initialize a dictionary to store the counts of each ballot type as strings
    ballot_counts = {}
    # Iterate through the DataFrame rows
    #df['weight'] = df['weight']*800/df['weight'].sum()
    for index, row in df.iterrows():
        # Initialize an empty list to store valid candidates
        valid_candidates = []

        # Iterate through columns rank1 to rank5
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

def process_ballot_counts_post_elim(ballot_counts, k, candidates, elim_cands, check_strats=False, budget_percent=0, check_removal_here=False, keep_at_least=8):
    # Initialize a dictionary for filtered data
    filtered_data = {}
    elim_strings = ''.join(char for char in elim_cands)

    # Remove letters {G, H, I, J, K, L} while retaining the rest of the string
    for key, value in ballot_counts.items():
        new_key = ''.join(char for char in key if char not in elim_strings)
        
        filtered_data[new_key] = filtered_data.get(new_key, 0) + value
    filtered_data.pop('', None)

    elec_cands = [cand for cand in candidates if cand not in elim_cands]
    
    full_aggre_v_dict = utils.get_new_dict(ballot_counts)
    aggre_v_dict = utils.get_new_dict(filtered_data)

    Q = round(sum(full_aggre_v_dict[cand] for cand in candidates)/(k+1)+1, 3)
    print("\n" + "="*50)
    print(f"Q = {Q}")
    print("="*50 + "\n")
    
    letter_counts = {}

    # strict_support
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

    print("-"*50)
    print(f"Strict support within {elim_cands}:")
    for letter, count in letter_counts.items():
        print(f"  {letter}: {count}")
    
    best_c_irrelevant = max(letter_counts, key=letter_counts.get) if letter_counts else None
    if best_c_irrelevant:
        print(f"\nMax strict support is with: {best_c_irrelevant}")
    print("-"*50 + "\n")

    rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
    results, subresults = utils.return_main_sub(rt)
    print("="*50)
    print(f"Overall winning order is: {results}")
    print("="*50 + "\n")

    budget = budget_percent * sum(full_aggre_v_dict[cand] for cand in candidates) * 0.01
    if check_removal_here:
        candidates_reduced, group_remaining, stop = remove_irrelevent(ballot_counts, rt, 
                results[:keep_at_least], budget, ''.join(candidates))
        
        print("-"*50)
        if stop:
            print(f"We can remove {group_remaining} and keep {candidates_reduced}")
        else: 
            print("We cannot remove any more candidates")
        print("-"*50 + "\n")

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



def generate_bootstrap_samples(data, n_samples=1000, save = False):
    """
    Generates bootstrap samples from the given dataset.

    Parameters:
    data (DataFrame): The original dataset containing RCV rankings.
    n_samples (int): The number of bootstrap samples to generate.

    Returns:
    List[DataFrame]: A list containing the bootstrap samples.
    """
    bootstrap_samples = []
    n = len(data)
    for _ in range(n_samples):
        sample = data.sample(n, replace=True)  # Sampling with replacement
        bootstrap_samples.append(sample)

    if save == True:
        # Creating a directory to store all the bootstrap sample files
        output_dir = 'bootstrap_samples_new'
        os.makedirs(output_dir, exist_ok=True)

        # Saving each bootstrap sample as a separate CSV file
        for i, sample in enumerate(bootstrap_samples):
            sample_file_path = os.path.join(output_dir, f'bootstrap_sample_{i+1}.csv')
            sample.to_csv(sample_file_path, index=False)
        
    return bootstrap_samples

def process_bootstrap_samples(k, candidates_mapping, bootstrap_samples_dir, bootstrap_files, budget_percent, keep_at_least, iters = 10, want_strats = False, save = False):
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
    
        if 'rank1' in df.columns:
            ballot_counts = get_ballot_counts_df_republican_primary(candidates_mapping, df)
            Q = 800
            budget = budget_percent*sum(ballot_counts.values())*0.01
        else:
            ballot_counts = get_ballot_counts_df(candidates_mapping, df)
            Q = round(sum(ballot_counts.values())/(k+1)+1,3) 
            budget = budget_percent*sum(ballot_counts.values())*0.01
    
        # Compute results using STV
        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, Q)
        results, subresults = utils.return_main_sub(rt)
        candidates_reduced, group, stop = remove_irrelevent(ballot_counts, rt, 
                results[:keep_at_least], budget, ''.join(candidates))
    
        #print(f"Iteration {it}: Candidates = {candidates_reduced}")

        if stop:
            algo_works += 1

            if want_strats == True:

                # Initialize a dictionary for filtered data
                filtered_data = {}

                # Remove letters {G, H, I, J, K, L} while retaining the rest of the string
                for key, value in ballot_counts.items():
                    new_key = ''.join(char for char in key if char not in group)
                    filtered_data[new_key] = filtered_data.get(new_key, 0) + value
                filtered_data.pop('', None)
        
                strats_frame = reach_any_winners_campaign(candidates_reduced, k, Q, filtered_data, budget)
                data_samples.append(strats_frame)

                if save == True:
                    # Creating a directory to store all the strategy optimization results
                    output_dir = 'strategy_optimization_results'
                    os.makedirs(output_dir, exist_ok=True)
                    # Saving the strategy optimization results as a JSON file

                    save_path = os.path.join(output_dir, f"iteration_{it}.json")
                    with open(save_path, "w") as f:
                        json.dump({"iteration": it, "strats_frame": strats_frame}, f, indent=4)

                    print(f"Iteration {it}: strats_frame = {strats_frame}")
            
        end = time.time()    
        #print('Total time =', end - start)
    print(algo_works, it-1, ' Removal Efficiency is ', algo_works/(it-1)*100, '%')
    return algo_works, data_samples


