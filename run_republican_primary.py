# republican_primary_analysis.py
# Analysis for Republican Primary data

# Standard library imports
import os

# Local application/library specific imports
from Case_Studies.Republican_Primary_Case_Data_and_Analysis.load_data import election_data
import case_study_helpers
from case_study_helpers import process_ballot_counts_post_elim, process_bootstrap_samples
from case_study_analysis_tools import comprehensive_voting_analysis


def analyze_republican_primary(k=1, budget_percent=4.85, keep_at_least=6, 
                              bootstrap_k=2, bootstrap_keep=8, bootstrap_iters=100,
                              check_strats=True, check_removal=True, 
                              show_plots=True, print_results=True):
    """
    Analyze Republican Primary voting data.
    
    Parameters:
    - k: Number of winners for main analysis
    - budget_percent: Budget percentage for analysis
    - keep_at_least: Minimum number of candidates to keep for main analysis
    - bootstrap_k: Number of winners for bootstrap analysis
    - bootstrap_keep: Minimum number of candidates to keep for bootstrap analysis
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
    # Load election data
    candidates_mapping = election_data['candidates_mapping']
    df = election_data['df']
    bootstrap_files = election_data['bootstrap_files']
    bootstrap_dir = election_data['bootstrap_dir']
    
    print("Analyzing Republican Primary Data")
    print(f"Number of candidates: {len(candidates_mapping)}")
    
    # Process ballot counts
    ballot_counts = case_study_helpers.get_ballot_counts_df_republican_primary(candidates_mapping, df)
    candidates = list(candidates_mapping.values())
    elim_cands = candidates[-8:]
    
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
        bootstrap_k, 
        candidates_mapping, 
        bootstrap_dir, 
        bootstrap_files, 
        budget_percent=budget_percent, 
        keep_at_least=bootstrap_keep, 
        iters=bootstrap_iters,
        want_strats=check_strats, 
        save=False
    )
    
    # Comprehensive statistical analysis
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


if __name__ == "__main__":
    # Run the Republican Primary analysis
    results, figure, algo_works, data_samples = analyze_republican_primary(
        k=1,
        budget_percent=4.85,
        keep_at_least=6,
        bootstrap_k=2,
        bootstrap_keep=8,
        bootstrap_iters=100,
        show_plots=True,
        print_results=True
    )