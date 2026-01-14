import os
import json
import sys

from rcv_strategies.utils.case_study_helpers import process_bootstrap_samples, get_ballot_counts_df
from rcv_strategies.analysis.tools import (
    comprehensive_voting_analysis,
    analyze_detailed_vote_additions
)
from case_studies.portland.load_district_data import district_data

def find_true_multi_layer_files(final_dir, json_list):
    bad = []
    for fn in json_list:
        path = os.path.join(final_dir, fn)
        with open(path) as r:
            frame = json.load(r)["strats_frame"]
        detailed = analyze_detailed_vote_additions([frame])
        all_keys = [k for sub in detailed.values() for k in sub.keys()]
        if any(k.startswith("Multi-ranked:") or k.startswith("Combination:") for k in all_keys):
            bad.append(fn)
    return bad

if __name__ == "__main__":
    DIST      = 1
    INFO      = district_data[DIST]
    BOOT_DIR  = INFO["bootstrap_dir"]
    BASE_DIR  = os.path.dirname(BOOT_DIR)
    FINAL_DIR = os.path.join(BASE_DIR, "final_results_dis1")

    K      = 3
    BUDGET = 4
    KEEP   = 8

    # 1) Gather and sort your JSON iteration files
    json_list = [fn for fn in os.listdir(FINAL_DIR)
                 if fn.startswith("iteration_") and fn.endswith(".json")]
    json_list.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    N = len(json_list)
    print(f"🔍 Found {N} iteration JSONs in {FINAL_DIR}")

    # 2) Grab the first N bootstrap CSVs (so they align 1:1 with your JSONs)
    csv_list = sorted(INFO["bootstrap_files"])
    if len(csv_list) < N:
        print("❌ Not enough CSVs to match JSONs. Exiting.")
        sys.exit(1)
    csv_list = csv_list[:N]

    # 3) Detect which JSONs were truly multi-layer
    bad_jsons = find_true_multi_layer_files(FINAL_DIR, json_list)
    bad_idxs  = [json_list.index(fn) for fn in bad_jsons]
    good_idxs = [i for i in range(N) if i not in bad_idxs]

    print(f"🚩 {len(bad_idxs)} bad iterations (multi-layer):",
          [json_list[i] for i in bad_idxs])
    print(f"✅ {len(good_idxs)} good iterations (single-layer):",
          [json_list[i] for i in good_idxs])

    # 4) Load the original strats_frame for the good ones
    good_samples = []
    for i in good_idxs:
        fn = json_list[i]
        with open(os.path.join(FINAL_DIR, fn)) as r:
            good_samples.append(json.load(r)["strats_frame"])
    print(f"Loaded {len(good_samples)} good samples.")

    # 5) Re-run only the bad CSVs with allowed_length=1
    bad_csvs = [csv_list[i] for i in bad_idxs]
    print(f"Re-running {len(bad_csvs)} bad CSVs:", bad_csvs)

    algo_works_bad, bad_samples = process_bootstrap_samples(
        K,
        INFO["candidates_mapping"],
        BOOT_DIR,
        bad_csvs,
        budget_percent=BUDGET,
        keep_at_least=KEEP,
        iters=len(bad_csvs),
        want_strats=True,
        save=False,
        allowed_length=1
    )
    print(f"Produced {len(bad_samples)} corrected bad samples.")

    # 6) Combine and summarize
    combined = good_samples + bad_samples
    total_runs = len(combined)
    ballot_counts = get_ballot_counts_df(INFO["candidates_mapping"], INFO["df"])
    total_votes   = sum(ballot_counts.values())

    print(f"🔀 Combining {len(good_samples)} good + {len(bad_samples)} bad → {total_runs} total runs")
    comprehensive_voting_analysis(
        data_samples   = combined,
        total_votes    = total_votes,
        algo_works     = total_runs,
        budget_percent = BUDGET,
        show_plots     = True,
        print_results  = True
    )
