# portland_analysis.py
# Analysis for Portland City Council data

# Standard library imports
import os
import json

# Local application/library specific imports
from Case_Studies.Portland_City_Council_Data_and_Analysis.load_district_data import district_data
import case_study_helpers
from case_study_helpers import process_ballot_counts_post_elim, process_bootstrap_samples
from case_study_analysis_tools import comprehensive_voting_analysis


def analyze_portland_district(district_number, k=3, budget_percent=4.15, keep_at_least=7, 
                             bootstrap_iters=20, check_strats=True, check_removal=True,
                             show_plots=True, print_results=True):
    """
    Analyze Portland City Council voting data for a specific district.
    
    Parameters:
    - district_number: District number (1-4)
    - k: Number of winners
    - budget_percent: Budget percentage for analysis
    - keep_at_least: Minimum number of candidates to keep
    - bootstrap_iters: Number of bootstrap iterations
    - check_strats: Whether to check strategies
    - check_removal: Whether to check removal
    - show_plots: Whether to show plots
    - print_results: Whether to print detailed results
    
    Returns:
    - results: Analysis results
    - figure: Generated figure
    - algo_works: Whether algorithm works
    - data_samples: Bootstrap data samples
    """
    # Validate district number
    if district_number not in [1, 2, 3, 4]:
        raise ValueError("District number must be 1, 2, 3, or 4")
    
    # Load district data
    candidates_mapping = district_data[district_number]['candidates_mapping']
    df = district_data[district_number]['df']
    bootstrap_files = district_data[district_number]['bootstrap_files']
    bootstrap_dir = district_data[district_number]['bootstrap_dir']
    
    print(f"Analyzing Portland City Council District {district_number}")
    print(f"Number of candidates: {len(candidates_mapping)}")
    
    # Process ballot counts
    ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
    candidates = list(candidates_mapping.values())
    elim_cands = candidates[-15:]
    
    # Analyze main dataset
    print("Analyzing main dataset...")
    process_ballot_counts_post_elim(
        ballot_counts,
        k, 
        candidates, 
        elim_cands, 
        check_strats=check_strats, 
        budget_percent=budget_percent, 
        check_removal_here=check_removal, 
        keep_at_least=keep_at_least
    )
    
    # Analyze bootstrap samples
    print("Analyzing bootstrap samples...")
    algo_works, data_samples = process_bootstrap_samples(
        k, 
        candidates_mapping, 
        bootstrap_dir, 
        bootstrap_files, 
        budget_percent=budget_percent, 
        keep_at_least=keep_at_least, 
        iters=bootstrap_iters,
        want_strats=check_strats, 
        save=False
    )
    
    # Comprehensive statistical analysis
    results = None
    figure = None
    if show_plots or print_results:
        print("Performing comprehensive statistical analysis...")
        results, figure = comprehensive_voting_analysis(
            data_samples=data_samples,
            total_votes=sum(ballot_counts.values()),
            algo_works=algo_works,
            budget_percent=budget_percent,
            show_plots=show_plots,
            print_results=print_results
        )
    
    return results, figure, algo_works, data_samples

def get_bootstrat_analysis_samples():
        # Define the output directory where JSON files were saved
    output_dir = "Case_Studies/Portland_City_Council_Data_and_Analysis/Dis_1/final_results_dis1"  # Replace with your actual directory

    # Initialize an empty list to store data samples
    data_samples = []

    # Get all JSON files in the output directory
    json_files = sorted([f for f in os.listdir(output_dir) if f.startswith("iteration_") and f.endswith(".json")])

    # Load data from each JSON file
    for file in json_files:
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            data_samples.append(data["strats_frame"])
    return data_samples

if __name__ == "__main__":
    # Example usage
    data_samples = get_bootstrat_analysis_samples()
    comprehensive_voting_analysis(
        data_samples=data_samples,
        total_votes=42686,
        algo_works=84,
        budget_percent=4,
        show_plots=False,
        print_results=True
    )
    district_number = 1
    # results, figure, algo_works, data_samples = analyze_portland_district(
    #     district_number=district_number,
    #     k=3,
    #     budget_percent=4.15,
    #     keep_at_least=7,
    #     bootstrap_iters=20,
    #     show_plots=True,
    #     print_results=True
    # )