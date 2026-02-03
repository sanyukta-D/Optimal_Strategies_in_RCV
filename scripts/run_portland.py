# portland_analysis.py

'''
\section{2024 Portland City Council Elections: Details} \label{app:portland}
For this case study, we utilized the official cast vote record (CVR) released in December 2024. The record included improperly marked ballots, which were processed in accordance with the official ranked-choice voting (RCV) ballot adjudication procedures \cite{MultnomahCounty2024Turnout}. Each election followed the single transferable vote (STV) rules, electing exactly three winners using the Droop quota of 25\% \citep{multnomah_rcv2025}.
Applying \Cref{thm: stvprotocol}, our structural framework is applicable to analyzing these elections. Using our algorithms, we examined strategic behavior across all four elections. Our methodology follows the format established in the Republican primary case study in \Cref{app: casestudies}. Below, we provide a brief analysis of each district's elections and the subsequent bootstrap analysis.

\subsection{District 1}\label{app:portland_dis1}

District 1 includes 16 candidates, with the first candidate winning in the 13th round following the elimination of 11 candidates. Given \(n = 16\) and up to \(k = 3\) winners, the total number of possible orders is \(16!\), and the number of possible sequences is \(\sum_{j=1}^3 \binom{16}{j} = 696\). To analyze this district, we apply the methodology of \Cref{thm: remove_irrelevant_candidates} as detailed in \Cref{app: perfect_info}. If the standard removal criterion is not met for a bottom group \(L\), we further check whether the budget can preserve the candidate in \(L\) with the highest Strict-Support from elimination in the next round, thereby potentially influencing two eliminations simultaneously.
For \(B = 4.47\%\), these steps remove 8 candidates; among the remaining 8, only 7 are able to win by adding up to 4.47\% of additional votes. These are the highest allocation numbers for successful candidate removal. Note that candidate H, the 8th candidate, is included in the set of relevant candidates but remains unable to secure a win through strategic additions. This occurs because H has the potential to attain higher positions in the ranking, thereby influencing the election dynamics, yet ultimately falls short of meeting the necessary threshold to win. The subsequent strategic analysis of this set, performed via \Cref{thm: poly_efforts_B}, takes approximately 814 minutes on a modern laptop. The resulting optimal strategies for 7 candidates are presented in \Cref{tab:dis1_strats}. Reducing \(B\) to 4.17\% allows the removal of 9 candidates, cutting the corresponding analysis time to about 12 minutes.

For District 1 bootstrap analysis, we generated 1,000 samples using sampling with replacement, similar to our analysis in \Cref{sec: casestudies}. We then applied a slightly lower budget limit of \(B = 4\%\), as 4.17\% is precise for the original dataset. Under these conditions, 9 candidates were removed in 807 samples and 8 in 190 samples, leaving 3 samples unsolved. From those with 9-candidate removal, we randomly selected 84 for in-depth strategic analysis, summarized in \Cref{tab:summary_dis1}. This step required a total of 865 minutes on a modern laptop.

\subsection{Districts 2, 3 and 4}
District 2 had 22 candidates, with the first win occurring in the 20th round after the elimination of 18 candidates. The subsequent analysis, including both the election data and bootstrap procedures, follows the same methodology as in District 1 (\Cref{app:portland_dis1}). Given the lower competitiveness of the District 2 election, the algorithms achieved a higher margin in eliminating irrelevant candidates, reaching up to 6.5\% while removing 18 candidates.
We analyzed 100 bootstrap samples with a bound of 6\% on allowed additions. The candidate-elimination algorithm was effective for all samples, reducing the number of relevant candidates to four in each case. Candidate D secured a place in the winning set in 85\% of the samples, requiring an average of 5.64\% additional votes.


In Districts 3 and 4, the first election winner emerged in rounds 20 and 7, following the elimination of 18 and 5 candidates, respectively. In both cases, the first win occurred while at least 11 candidates remained active in the election. 
%This has two key implications: First, the remaining active candidates had vote totals comparable to each other and to those of the eliminated candidates, limiting the ability to eliminate candidates from the lower group despite allowed additions. Formally, the strict-support of the eliminated candidates remained close to the aggregated votes of the active candidates. Second, since a winner had already been determined, \Cref{thm: remove_irrelevant_candidates} could not be directly applied to later rounds, restricting its use for further candidate elimination.
Thus, for Districts 3 and 4, we apply \Cref{thm: irrelevant_extension}, which enables candidate elimination even when the removal set includes an election winner. Using the corresponding algorithm, we determine the highest number of additional votes—i.e., the bound—that satisfies the elimination condition. The computed bounds for Districts 3 and 4 are 12.36\% and 9.6\%, respectively.
For bootstrap samples, we used 11.5\% bound for District 3, and 9\% for District 4 and analyzed 100 bootstrap samples. Within these bounds, the algorithms achieve 100\% efficiency in eliminating irrelevant candidates and analyzing samples in both districts.

'''
# Analysis for Portland City Council data

# Standard library imports
import sys
from pathlib import Path
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local application/library specific imports
from case_studies.portland.load_district_data import district_data
from rcv_strategies.utils import case_study_helpers
from rcv_strategies.utils.case_study_helpers import process_ballot_counts_post_elim, process_bootstrap_samples
from rcv_strategies.analysis.tools import comprehensive_voting_analysis


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
    # elim_cands: efficiency shortcut - skip early removal iterations.
    # Instead of testing 16→15→14→...→8, skip to ~10-9 and let removal test from there.
    elim_cands = candidates[-9:]
    
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
        keep_at_least=keep_at_least, rigorous_check=True
    )
    # Analyze bootstrap samples
    # print("Analyzing bootstrap samples...")
    # algo_works, data_samples = process_bootstrap_samples(
    #     k, 
    #     candidates_mapping, 
    #     #bootstrap_dir, 
    #     df ,
    #     bootstrap_files, 
    #     budget_percent=budget_percent, 
    #     keep_at_least=keep_at_least, 
    #     iters=bootstrap_iters,
    #     want_strats=check_strats, 
    #     save=False,
    #     spl_check=True,
    #     rigorous_check=False
    # )
    
    #### Comprehensive statistical analysis
    results = None
    figure = None
    # if show_plots or print_results:
    #     print("Performing comprehensive statistical analysis...")
    #     results, figure = comprehensive_voting_analysis(
    #         data_samples=data_samples,
    #         total_votes=sum(ballot_counts.values()),
    #         algo_works=algo_works,
    #         budget_percent=budget_percent,
    #         show_plots=show_plots,
    #         print_results=print_results
    #     )
    
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
    ##### Example usage
    # data_samples = get_bootstrat_analysis_samples()
    # print(data_samples)
    # comprehensive_voting_analysis(
    #     data_samples=data_samples,
    #     total_votes=42686,
    #     algo_works=84,
    #     budget_percent=4,
    #     show_plots=False,
    #     print_results=True
    # )
    district_number = 1
    results, figure, algo_works, data_samples = analyze_portland_district(
        district_number=district_number,
        k=3,
        budget_percent=4,
        keep_at_least=8,
        bootstrap_iters=1,
        check_strats=True,
        show_plots=True,
        print_results=True
    )
    print(results, figure, algo_works, data_samples )

    #dis2 : 5.6 vs 6.5 with rigor
    #dis1 : 4.17 vs 4.7 with rigor


