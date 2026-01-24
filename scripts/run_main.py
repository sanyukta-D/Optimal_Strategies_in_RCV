import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

#from case_studies.portland.load_district_data import district_data
#from case_studies.republican_primary.load_data import election_data
from rcv_strategies.core.stv_irv import STV_optimal_result_simple, IRV_ballot_exhaust
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.core import candidate_removal
from rcv_strategies.utils import helpers as utils
from rcv_strategies.utils.case_study_helpers import process_ballot_counts_post_elim_no_print, process_ballot_counts_post_elim, process_bootstrap_samples
from rcv_strategies.analysis.tools import comprehensive_voting_analysis
import pandas as pd
from case_studies.nyc.convert_data import process_single_file, create_candidate_mapping
import os
from string import ascii_uppercase


### NEW YORK CITY RCV elections case study ####
input_file = "NewYorkCity_06222021_DEMCouncilMember7thCouncilDistrict.csv"
# Use relative path from project root
project_root = Path(__file__).parent.parent
full_path = project_root / "case_studies" / "nyc" / "data" / input_file
k = 1
df, processed_file = process_single_file(full_path)

candidates_mapping = create_candidate_mapping(processed_file)
ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)

candidates = list(candidates_mapping.values())
rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
results, subresults = utils.return_main_sub(rt)
# Create a reverse mapping to go from letter codes to candidate names.
reverse_mapping = {code: name for name, code in candidates_mapping.items()}
ordered_candidate_names = [reverse_mapping[code] for code in results]

# Now create a final mapping where the ordered candidate names are assigned new letter codes.
final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}

ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
candidates = list(final_mapping.values())


candidates = list(candidates_mapping.values())
print("Number of candidates: ", len(candidates))
rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
results, subresults = utils.return_main_sub(rt)
elim_cands = results[-6:]
zeros , c_l = candidate_removal.predict_losses(ballot_counts, candidates, k, 11608, 1160)

# exhausted_ballots_list, exhausted_ballots_dict = IRV_ballot_exhaust(candidates, ballot_counts)
# ballot_counts['C'] = ballot_counts['C'] + 1490
# rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
# results, subresults = utils.return_main_sub(rt)
# print(results)   
# exhausted_ballots_dict_percent = {key: round(value/sum(ballot_counts.values())*100, 3) for key, value in exhausted_ballots_dict.items()}
# print("Exhausted ballots dictionary percentage: ", exhausted_ballots_dict_percent)


process_ballot_counts_post_elim(ballot_counts,
            k,
            results,
            elim_cands,
            check_strats=True,
            budget_percent=18,
            check_removal_here=True,
            keep_at_least=10, zeros= 0, rigorous_check=True
        )

# print(candidate_removal.predict_losses(ballot_counts, candidates, k, 11608, 1160))


# #nyc_folder = "Case_Studies/Alaska_RCV/alaska_files"
# nyc_folder = "Case_Studies/NewYork_City_RCV/nyc_files"
# results_list = []
# file_iter = 0
# for input_file in os.listdir(nyc_folder):
#     file_iter += 1
#     # input_file = "NewYorkCity_06222021_DEMCouncilMember40thCouncilDistrict.csv"
#     # if file_iter > 1:
#     #     break   
#     if input_file.endswith('.csv'):

#         full_path = os.path.join(nyc_folder, input_file)

#         df, processed_file = process_single_file(full_path)

#         k = 1
#         budget_percent = 15
#         # Initial processing: build original mapping and ballot counts
#         # Initial processing: build original mapping and ballot counts
#         candidates_mapping = create_candidate_mapping(processed_file)
#         ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)

#         candidates = list(candidates_mapping.values())
#         rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, k, sum(ballot_counts.values())/(k+1))
#         results, subresults = utils.return_main_sub(rt)
#         # Create a reverse mapping to go from letter codes to candidate names.
#         reverse_mapping = {code: name for name, code in candidates_mapping.items()}
#         ordered_candidate_names = [reverse_mapping[code] for code in results]

#         # Now create a final mapping where the ordered candidate names are assigned new letter codes.
#         final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}

#         ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
#         candidates = list(final_mapping.values())

#         #print("Final mapping: ", final_mapping)
#         exhausted_ballots_list, exhausted_ballots_dict = IRV_ballot_exhaust(candidates, ballot_counts)

#         exhausted_ballots_dict_percent = {key: round(value/sum(ballot_counts.values())*100, 3) for key, value in exhausted_ballots_dict.items()}
#         print("Exhausted ballots dictionary percentage: ", exhausted_ballots_dict_percent)

#         if len(candidates)>9:
#             print("Number of candidates: ", len(candidates))
#             elim_cands = []#results[-5:]
#             # process_ballot_counts_post_elim(ballot_counts,
#             #                        k, candidates, 
#             #                         elim_cands, 
#             #                         check_strats=True, 
#             #                         budget_percent = 10, 
#             #                         check_removal_here = True, 
#             #                         keep_at_least = 9)
#             # Get the results from the updated function.
#             file_result = process_ballot_counts_post_elim_no_print(ballot_counts,
#                 k,
#                 candidates,
#                 elim_cands,
#                 check_strats=False,
#                 budget_percent=budget_percent,
#                 check_removal_here=False,
#                 keep_at_least=9
#             )
            
#             file_result["file_name"] = input_file
#             file_result["exhaust_percents"] = exhausted_ballots_dict_percent
#             file_result["budget_percent"] = budget_percent  # Include filename for reference.
#             results_list.append(file_result)

# # Create a summary table from the collected results.
# df_results = pd.DataFrame(results_list)
# summary_columns = ["file_name", "exhaust_percents", "total_votes", "num_candidates", "candidates_removed",  
#                     "candidates_retained", "initial_zeros", "def_losing", "overall_winning_order", "Strategies", "budget_percent"]
# print("\nSummary Table:")
# pd.set_option('display.max_rows', None)
# print(df_results[summary_columns])
# output_filename = "summary_table.xlsx"
# df_results[summary_columns].to_excel(output_filename, index=False)
# print(f"Summary table exported to {output_filename}")


# # #### Portland City Council Case Study ####

# # dis_number = 4  #choose district number from 1,2,3 and 4

# # candidates_mapping = district_data[dis_number]['candidates_mapping']
# df = district_data[dis_number]['df']
# bootstrap_files = district_data[dis_number]['bootstrap_files']
# bootstrap_dir = district_data[dis_number]['bootstrap_dir']

# df.head()

# ballot_counts= case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
# k= 3
# candidates = list(candidates_mapping.values())
# elim_cands = candidates[-15:]

# # ### Do a full analysis of the Portland City Council data ####
# process_ballot_counts_post_elim(ballot_counts,
#                                k, candidates, 
#                                 elim_cands, 
#                                 check_strats=True, 
#                                 budget_percent = 4.15, 
#                                 check_removal_here = True, 
#                                 keep_at_least = 7)

# ### Do a full analysis of the bootstrap samples ####  
# algo_works, data_samples = process_bootstrap_samples(k, candidates_mapping, 
#                           bootstrap_dir, 
#                           bootstrap_files, budget_percent = 5, 
#                           keep_at_least= 7, iters = 5, loopy_removal= False,
#                           want_strats = True, save = False, spl_check=True)



# #### Do a full statistical analysis on bootstrap data ####

# results, figure = comprehensive_voting_analysis(
#     data_samples=data_samples,
#     total_votes=sum(ballot_counts.values()),
#     algo_works=algo_works,
#     budget_percent=4,
#     show_plots=False,   # Controls whether plots are shown
#     print_results=True  # Controls whether detailed results are printed
# )


# #### Republican Primary Case Study ####

# candidates_mapping = election_data['candidates_mapping']
# df = election_data['df']
# bootstrap_files = election_data['bootstrap_files']
# bootstrap_dir = election_data['bootstrap_dir']  

# df.head()   

# ballot_counts = case_study_helpers.get_ballot_counts_df_republican_primary(candidates_mapping, df)
# k = 1   
# candidates = list(candidates_mapping.values())
# elim_cands = candidates[-8:]

# # ### Do a full analysis of the Republican Primary data ####
# # process_ballot_counts_post_elim(ballot_counts,
# #                                k, candidates, 
# #                                 elim_cands, 
# #                                 check_strats=True, 
# #                                 budget_percent = 4.85, 
# #                                 check_removal_here = True, 
# #                                 keep_at_least = 6)

# #### Do a full analysis of the bootstrap samples ####
# algo_works, data_samples = process_bootstrap_samples(2, candidates_mapping, 
#                           bootstrap_dir, 
#                           bootstrap_files, budget_percent = 4.85, 
#                           keep_at_least= 8, iters = 100,
#                           want_strats = True, save = False)

