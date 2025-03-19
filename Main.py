
from Case_Studies.Portland_City_Council_Data_and_Analysis.load_district_data import district_data
import case_study_helpers
from case_study_helpers import process_ballot_counts_post_elim

dis_number = 1
# Get a district's data

candidates_mapping = district_data[dis_number]['candidates_mapping']
df = district_data[dis_number]['df']

df.head()

ballot_counts= case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
k= 3
candidates = list(candidates_mapping.values())
elim_cands = candidates[-10:]

process_ballot_counts_post_elim(ballot_counts,
                               k, candidates, 
                                elim_cands, 
                                check_strats=True, 
                                budget_percent = 4.15, 
                                check_removal_here = True, 
                                at_least = 7)
