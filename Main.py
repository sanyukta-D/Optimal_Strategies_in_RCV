
from Case_Studies.Portland_City_Council_Data_and_Analysis.load_district_data import district_data
from Case_Studies.Republican_Primary_Case_Data_and_Analysis.load_data import election_data
import case_study_helpers
from case_study_helpers import process_ballot_counts_post_elim, process_bootstrap_samples
from case_study_analysis_tools import comprehensive_voting_analysis

#### Portland City Council Case Study ####

dis_number = 4  #choose district number from 1,2,3 and 4

candidates_mapping = district_data[dis_number]['candidates_mapping']
df = district_data[dis_number]['df']
bootstrap_files = district_data[dis_number]['bootstrap_files']
bootstrap_dir = district_data[dis_number]['bootstrap_dir']

df.head()

ballot_counts= case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
k= 3
candidates = list(candidates_mapping.values())
elim_cands = candidates[-15:]

# # ### Do a full analysis of the Portland City Council data ####
# process_ballot_counts_post_elim(ballot_counts,
#                                k, candidates, 
#                                 elim_cands, 
#                                 check_strats=True, 
#                                 budget_percent = 4.15, 
#                                 check_removal_here = True, 
#                                 keep_at_least = 7)

### Do a full analysis of the bootstrap samples ####  
algo_works, data_samples = process_bootstrap_samples(k, candidates_mapping, 
                          bootstrap_dir, 
                          bootstrap_files, budget_percent = 5, 
                          keep_at_least= 7, iters = 5, loopy_removal= False,
                          want_strats = True, save = False, spl_check=True)



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

