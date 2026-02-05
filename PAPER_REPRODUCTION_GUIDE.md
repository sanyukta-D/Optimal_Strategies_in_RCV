# Paper Reproduction Guide
## "Optimal Strategies in Ranked Choice Voting" — Complete Asset Map

This document maps every figure, table, data file, and code file needed to reproduce
ALL paper results across three LaTeX files.

---

## 0. LATEX FILES

There are **3 LaTeX files** in `codes/EC_codes/latex files/`:

| File | Paper | Content |
|------|-------|---------|
| `Final_draft.tex` | "Simpler Than You Think: The Practical Dynamics of Ranked Choice Voting" | NYC/Alaska single-winner + Portland multi-winner case studies, ballot exhaustion analysis, probability models, heatmap |
| `JAIR_submission.tex` | "Optimal Strategies in Ranked Choice Voting" | Theoretical framework, algorithms, 2024 US Republican Primary case study |
| `case_study_20240130.tex` | (Standalone section) | Republican Primary case study section — same content as Sec. 5 of `JAIR_submission.tex` and `\label{sec: casestudies}` |

`Final_draft.tex` contains 8 figures and 11 tables (NYC, Alaska, Portland).
`JAIR_submission.tex` contains 0 figures and ~17 tables (theory examples, Republican Primary).
`case_study_20240130.tex` is the Republican Primary case study section extracted for standalone use.

---

## 1. FIGURES (Final_draft.tex only)

### Fig 1 — NYC 2017 Plurality Mayoral Primary
- **Image:** `images/nyc_mayor_2017.png`
- **Label:** `fig:mayor_2017`
- **Source:** External screenshot from NYC Board of Elections website
- **No code needed** — manual screenshot

### Fig 2 — NYC 2021 RCV Mayoral Primary
- **Image:** `images/NYC_mayor_2021.png`
- **Label:** `fig:mayor_2021`
- **Source:** External screenshot from NYC Board of Elections website
- **No code needed** — manual screenshot

### Fig 3 — Margin of Victory Violin Plots (NYC + Alaska)
- **Images:** `images/violin_competitive_bands.pdf`, `images/violin_bands_alaska.pdf`
- **Label:** `fig:margins-violin-comparison`
- **Generating script:** `case_studies/nyc/violin_plot.py`
- **Input data:**
  - `results/tables/summary_table_nyc_final.xlsx` (NYC 2021 victory gaps)
  - `results/tables/summary_table_alska_lite.xlsx` (Alaska 2024 victory gaps)
  - External pre-RCV data files referenced inside violin_plot.py

### Fig 4 — NYC Ballot Exhaustion (Boxplot + Models by Ratio)
- **Images:** `images/nyc_exhaust_strategy_diff_boxplot.pdf`, `images/nyc_models_by_ratio.pdf`
- **Label:** `fig:exhaustion_nyc`
- **Generating scripts:**
  - Boxplot (left): `case_studies/nyc/analyze_exhaust_strategy_diff.py`
  - Models (right): `ballot_exhaustion/probability_models.py` → `create_nyc_model_comparison()`
- **Input data:**
  - `results/tables/summary_table_nyc_final.xlsx` (boxplot)
  - `ballot_exhaustion/model_comparison_results/all_elections_analysis.csv` (models)

### Fig 5 — Alaska Ballot Exhaustion (Boxplot + Models by Ratio)
- **Images:** `images/alaska_exhaust_strategy_diff_boxplot.pdf`, `images/alaska_models_by_ratio.pdf`
- **Label:** `fig:exhaustion_alaska`
- **Generating scripts:**
  - Boxplot (left): `case_studies/nyc/analyze_exhaust_strategy_diff.py`
  - Models (right): `ballot_exhaustion/probability_models.py` → `create_alaska_model_comparison()`
- **Input data:**
  - `results/tables/summary_table_alska_lite.xlsx` (boxplot)
  - `ballot_exhaustion/model_comparison_results/all_elections_analysis.csv` (models)

### Fig 6 — NYC District 1 Official Results
- **Image:** `images/NYC_1_dist_table.png`
- **Label:** `fig:dis_1_nyc_official`
- **Source:** External screenshot from NYC Board of Elections
- **No code needed** — manual screenshot

### Fig 7 — Portland District 1 Official Results
- **Image:** `images/Portland_dis1.png`
- **Label:** `fig:portland_dis1`
- **Source:** External screenshot from Multnomah County interactive results
- **No code needed** — manual screenshot

### Fig 8 — Ballot Exhaustion Heatmap
- **Image:** `images/heatmap.png`
- **Label:** `fig:heatmap_prb`
- **Generating script:** `rcv_strategies/analysis/heatmap.py`
- **Input data:** `ballot_exhaustion/model_comparison_results/all_elections_analysis.csv`

---

## 2. TABLES — Final_draft.tex (NYC/Alaska/Portland)

### Table 1 — NYC District 1 Victory Gap & Exhaustion (tab:victory-gap)
- **Values hardcoded in tex** (lines 435-443)
- **Source data:** `results/tables/summary_table_nyc_final.xlsx`
  - `exhaust_percents` column → exhaustion values
  - `Strategies` column → victory gap values
- **Verified:** IRV_ballot_exhaust matches all 9 candidates exactly

### Table 2 — Portland Summary (tab:portland_summary)
- **Values hardcoded in tex** (lines 554-573)
- **Source data:** Portland election analysis via `scripts/run_portland.py`

### Table 3 — Portland District 1 Strategies (tab:portland_dis1_strats)
- **Values hardcoded in tex** (lines 575-615)
- **Source data:** Portland District 1 strategy computation

### Table 4 — District 1 Bootstrap Summary (tab:summary_dis1)
- **Values hardcoded in tex** (lines 617-658)
- **Source data:** Portland bootstrap analysis

### Table 5 — Districts 2,3,4 Bootstrap Summary (tab:multi_district_summary)
- **Values hardcoded in tex** (lines 660-706)
- **Source data:** Portland bootstrap analysis

### Table 6 — Portland Exhaustion Probabilities (tab:portland_district_exhaustion)
- **Values hardcoded in tex** (lines 708-731)
- **Source data:** STV_ballot_exhaust + probability_models for Portland Districts 1, 2, 4
- **Key values:** D,Dis1=14.59% exhaust; D,Dis4=14.08% exhaust
- **Generating code:** `ballot_exhaustion/probability_models.py` + `rcv_strategies/core/stv_irv.py`

### Table 7 — NYC Elections Complete Breakdown (tab:nyc_elections)
- **Long table** (lines 1282-1367)
- **Source data:** `results/tables/summary_table_nyc_final.xlsx`
- **Generating script:** `scripts/run_main.py` (processes all NYC elections)

### Table 8 — Alaska Elections Complete Breakdown (tab:alaska_elections)
- **Long table** (lines 1369-1454)
- **Source data:** `results/tables/summary_table_alska_lite.xlsx`
- **Generating script:** `scripts/run_main.py` (processes all Alaska elections)

### Table 9 — RCV Election Attributes Summary (tab:side_by_side_rcv_summary)
- **Includes sub-tables:** tab:nyc_rcv_extended, tab:ak_rcv_extended
- **Values hardcoded in tex** (lines 1455-1525)
- **Source data:** Aggregated from xlsx summary tables

### Table 10 — District 23 Non-Selfish Strategies (tab:district23_detail)
- **Values hardcoded in tex** (lines 1528-1557)
- **Source data:** NYC District 23 strategy analysis

### Table 11 — Stronger Removal Performance (tab:performance_by_size)
- **Values hardcoded in tex** (lines 1560-1575)
- **Source data:** Candidate removal analysis

---

## 3. TABLES — JAIR_submission.tex (Theory + Republican Primary)

These tables appear in `JAIR_submission.tex` (and `case_study_20240130.tex` for the case study section).

### Theory & Example Tables (main body)

| Label | Caption | Line | Content |
|-------|---------|------|---------|
| `tab: sf_election` | SF District 7 2020 election | 83-93 | Motivating example: Engardio vs Melgar vs Nguyen |
| `tab: sub_st_example` | Structures for order A>B>C>D | 238-252 | 8 sequences for 4-candidate example |
| `tab: coalition_categories` | Strategy types | 444-458 | Selfish, Altruistic to winners, Altruistic to losers |
| `tab: ex_1_original` | Example 1 original profile | 562-573 | 4-candidate STV example (B wins via altruistic) |
| `tab: ex_1_campaign` | Example 1 after campaigning | 577-589 | B adds 2 votes to D |
| `tab: ex_2_original` | Example 2 original profile | 596-608 | Social choice order flip example |
| `tab: ex_2_flip` | Example 2 after 1 vote change | 613-625 | Entire order flipped by 1 vote |

### Republican Primary Case Study Tables (Sec 5 / Appendix)

| Label | Caption | Line | Content |
|-------|---------|------|---------|
| `tab: votestowin` | Strategic additions to win | 32-46 (case_study) | Min % votes to win for T,D,R,H,C |
| `tab: top2strategies_5` | Top-2 strategies (5% budget) | 56-70 (case_study) | R,D,C strategies to reach top 2 |
| `tab:strategy_comparison` | Strategy efficacy under bootstrap | 87-101 (case_study) | % reaching top 2 under uncertainty |
| `tab: bootstrap_summary_table` | Bootstrap summary (5%) | 104-116 (case_study) | Top-2 frequency for H,R,D,C,P,Sc |
| `tab: combination_percentages` | Strategy categorization (5%) | 118-143 (case_study) | Detailed strategy type distribution |
| `tab:gop_primary` | Full RCV results on poll | 971-994 (JAIR) | 13 candidates, 12 rounds |
| `tab: strict_support_primary` | Strict-Support vs Trump | 1005-1015 (JAIR) | Head-to-head for all 12 non-Trump |
| `tab: strict-supportFtoM` | Strict-Support P to Su | 1027-1041 (JAIR) | For candidate removal proof |
| `tab: strategy_4_1` | Bootstrap summary (4%) | 1051-1060 (JAIR) | Appendix: 4% budget analysis |
| `tab: strategy_4_2` | Strategy categorization (4%) | 1063-1085 (JAIR) | Appendix: detailed types at 4% |
| `tab: strategy_3_1` | Bootstrap summary (3%) | 1092-1101 (JAIR) | Appendix: 3% budget analysis |
| `tab: strategy_3_2` | Strategy categorization (3%) | 1105-1126 (JAIR) | Appendix: detailed types at 3% |
| `tab:strategy_modifications` | Selfish modifications | 1132-1143 (JAIR) | Adding self-ranking to altruistic strategies |

- **Source data:** FairVote/WPA Intelligence poll (801 respondents, 13 candidates)
- **Raw data:** `case_studies/republican_primary/` (see Section 4 below)
- **Generating script:** `scripts/run_republican_primary.py`
- **Bootstrap results:** `case_studies/republican_primary/final_results_5_percent/` and `final_results_4_percent/`

---

## 4. KEY DATA FILES (Source of Truth)

### Excel Summary Tables
| File | Location | Contents |
|------|----------|----------|
| `summary_table_nyc_final.xlsx` | `results/tables/` | All 54 NYC DEM elections: strategies, exhaust_percents, victory gaps |
| `summary_table_alska_lite.xlsx` | `results/tables/` | All 52 Alaska elections: strategies, exhaust_percents, victory gaps |
| `summary_table_nyc_2025.xlsx` | `results/tables/` | NYC 2025 elections (supplementary) |

### CSV Analysis Results
| File | Location | Contents |
|------|----------|----------|
| `all_elections_analysis.csv` | `ballot_exhaustion/model_comparison_results/` | Per-candidate exhaustion %, gap %, 6 probability model outputs |

### Raw Ballot Data
| Directory | Contents |
|-----------|----------|
| `case_studies/nyc/data/` | 54+ NYC 2021 RCV ballot CSVs (Choice_1..Choice_5 format) |
| `case_studies/alaska/data/` | 52+ Alaska 2024 ballot CSVs (rank1..rank4 format) |
| `case_studies/portland/` | Portland 2024 district data (loaded via load_district_data.py) |

### Republican Primary Data
| File/Directory | Contents |
|----------------|----------|
| `case_studies/republican_primary/load_data.py` | Loads poll_data.csv + bootstrap files; defines candidate mapping (Trump→A, Haley→B, ...) |
| `codes/EC_codes/republican_primary/poll_data.csv` | Raw FairVote/WPA poll data (801 respondents, 13 candidates) |
| `case_studies/republican_primary/FVA_National_Debate2_MQ_230930[86] (1).pdf` | Original FairVote poll report |
| `case_studies/republican_primary/final_results_5_percent/` | Bootstrap iteration JSON files (5% budget) |
| `case_studies/republican_primary/final_results_4_percent/` | Bootstrap iteration JSON files (4% budget) |

---

## 5. KEY FUNCTIONAL CODE FILES

### Core RCV Algorithms
| File | Key Functions |
|------|--------------|
| `rcv_strategies/core/stv_irv.py` | `STV_optimal_result_simple()` — main STV election runner with collection tracking |
| | `IRV_optimal_result()` — single-winner IRV social choice order |
| | `IRV_ballot_exhaust()` — single-winner exhaustion computation |
| | `STV_ballot_exhaust()` — multi-winner exhaustion computation |
| | `create_STV_round_result_given_structure()` — round-by-round STV for given structure |
| `rcv_strategies/core/strategy.py` | Strategy computation (reach_any_winners_campaign, etc.) |
| `rcv_strategies/core/optimization.py` | Optimization routines for strategy search |
| `rcv_strategies/core/candidate_removal.py` | Irrelevant candidate elimination algorithm |

### Utility Functions
| File | Key Functions |
|------|--------------|
| `rcv_strategies/utils/helpers.py` | `get_new_dict()` — first-choice vote aggregation |
| | `return_main_sub()` — convert event log to social choice order |
| `rcv_strategies/utils/case_study_helpers.py` | `get_ballot_counts_df()` — convert DataFrame to ballot_counts dict |
| | `get_ballot_counts_df_republican_primary()` — Republican Primary ballot conversion |
| | `process_ballot_counts_post_elim_no_print()` — full analysis pipeline |
| | `process_bootstrap_samples()` — bootstrap analysis |
| `rcv_strategies/constants.py` | `TRACTABILITY_CONSTANT` — computational budget limit |

### Analysis & Visualization
| File | Key Functions |
|------|--------------|
| `rcv_strategies/analysis/heatmap.py` | Generates Fig 8 (heatmap.png) |
| `rcv_strategies/analysis/tools.py` | `comprehensive_voting_analysis()` — full election analysis |
| `rcv_strategies/analysis/probability.py` | Portland probability tables |
| `rcv_strategies/analysis/high_candidate.py` | High-candidate-count election handling |

### Ballot Exhaustion & Probability Models
| File | Key Functions |
|------|--------------|
| `ballot_exhaustion/probability_models.py` | `beta_probability()` — gap-based beta model |
| | `direct_posterior_beta()` — similarity beta model |
| | `prior_posterior_beta()` — prior-posterior beta model |
| | `category_based_bootstrap()` — first-preference bootstrap |
| | `limited_ranking_bootstrap()` — rank-restricted bootstrap |
| | `unconditional_bootstrap()` — unconditional bootstrap |
| | `process_election_data()` — main pipeline for all_elections_analysis.csv |
| | `create_nyc_model_comparison()` — Fig 4 right |
| | `create_alaska_model_comparison()` — Fig 5 right |

### Figure-Generating Scripts
| File | Generates |
|------|-----------|
| `case_studies/nyc/violin_plot.py` | Fig 3: violin_competitive_bands.pdf, violin_bands_alaska.pdf |
| `case_studies/nyc/analyze_exhaust_strategy_diff.py` | Fig 4/5 left: boxplots |
| `ballot_exhaustion/probability_models.py` | Fig 4/5 right: models_by_ratio |
| `rcv_strategies/analysis/heatmap.py` | Fig 8: heatmap.png |

### Data Processing Scripts
| File | Purpose |
|------|---------|
| `case_studies/nyc/convert_data.py` | `process_single_file()`, `create_candidate_mapping()` — NYC data loading |
| `case_studies/portland/load_district_data.py` | Loads Portland district data into `district_data` dict |
| `case_studies/portland/exhaustion_analysis.py` | `print_exhaustion_analysis()`, `create_district_summary()` |
| `case_studies/republican_primary/load_data.py` | Loads Republican Primary poll data + bootstrap files |

### Election Runner Scripts
| File | Purpose |
|------|---------|
| `scripts/run_main.py` | Run analysis on individual NYC/Alaska elections |
| `scripts/run_portland.py` | Run Portland multi-winner STV analysis |
| `scripts/run_republican_primary.py` | Run Republican Primary analysis + bootstrap |
| `scripts/run_nyc_2025.py` | Run NYC 2025 elections batch |
| `scripts/run_nyc_2025_large.py` | Run large NYC 2025 elections |

### Webapp
| File | Purpose |
|------|---------|
| `webapp/app.py` | Streamlit webapp — interactive election analysis |

---

## 6. REPRODUCTION CHECKLIST

### Final_draft.tex (NYC/Alaska/Portland paper)

1. **NYC summary table** → `scripts/run_main.py` on all files in `case_studies/nyc/data/`
   → produces `results/tables/summary_table_nyc_final.xlsx`

2. **Alaska summary table** → `scripts/run_main.py` on all files in `case_studies/alaska/data/`
   → produces `results/tables/summary_table_alska_lite.xlsx`

3. **Probability analysis** → `ballot_exhaustion/probability_models.py`
   → reads xlsx files + raw ballot data
   → produces `ballot_exhaustion/model_comparison_results/all_elections_analysis.csv`
   → produces Fig 4/5 right panels

4. **Exhaustion boxplots** → `case_studies/nyc/analyze_exhaust_strategy_diff.py`
   → reads xlsx files
   → produces Fig 4/5 left panels

5. **Violin plots** → `case_studies/nyc/violin_plot.py`
   → reads xlsx files + pre-RCV election data
   → produces Fig 3

6. **Heatmap** → `rcv_strategies/analysis/heatmap.py`
   → reads all_elections_analysis.csv
   → produces Fig 8

7. **Portland analysis** → `scripts/run_portland.py`
   → reads Portland district data
   → produces Table 2-6 values

8. **Webapp verification** → `webapp/app.py`
   → upload any CSV from `case_studies/nyc/data/` or `case_studies/alaska/data/`
   → shows victory gaps, exhaustion, strategies, probability models

### JAIR_submission.tex (Republican Primary paper)

9. **Republican Primary analysis** → `scripts/run_republican_primary.py`
   → reads `case_studies/republican_primary/load_data.py` (poll_data.csv + bootstrap files)
   → produces Tables: `tab:votestowin`, `tab:top2strategies_5`, `tab:strategy_comparison`,
     `tab:bootstrap_summary_table`, `tab:combination_percentages`
   → pre-computed bootstrap results in `case_studies/republican_primary/final_results_5_percent/`
     and `final_results_4_percent/` (JSON iteration files)
   → `comprehensive_voting_analysis()` from `rcv_strategies/analysis/tools.py` produces
     summary statistics for all strategy tables
