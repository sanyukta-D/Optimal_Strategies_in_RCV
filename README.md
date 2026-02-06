# Optimal Strategies in Ranked Choice Voting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Web App](https://img.shields.io/badge/Web_App-Live-brightgreen)](https://rcv-analyzer.streamlit.app/)

In Ranked Choice Voting (RCV) elections, how many additional votes does a candidate actually need to win? Is the answer always "just vote for yourself," or are there cases where a candidate's best strategy is to boost a rival? This repository provides algorithms and tools to answer these questions — and validates them on **130+ real U.S. elections**.

**Key finding:** RCV is *simpler than you think*. Across NYC, Alaska, and Portland elections, the vast majority of candidates' optimal strategies are straightforward self-support. Complex cross-candidate strategies are rare and typically only arise for candidates who are already far from winning.

> **[Try the live web app](https://rcv-analyzer.streamlit.app/)** — upload any RCV election CSV and get instant analysis.

## Papers

This repository accompanies two research papers:

1. **"Optimal Strategies in Ranked Choice Voting"**
   Sanyukta Deshpande, Nikhil Garg, Sheldon H. Jacobson.
   Algorithmic framework for computing optimal vote-addition strategies in IRV and multi-winner STV elections. Includes theoretical analysis under perfect and imperfect polling, and a case study on the 2024 Republican Primary.

2. **"Simpler Than You Think: The Practical Dynamics of Ranked Choice Voting"**
   Sanyukta Deshpande, Nikhil Garg, Sheldon H. Jacobson.
   Empirical analysis of 130+ real elections across four U.S. jurisdictions, introducing four interpretability attributes: victory gap, ballot exhaustion impact, strategic complexity, and preference order alignment.

## What the Framework Computes

For any RCV/STV election, the framework answers four questions:

| Attribute | Question |
|-----------|----------|
| **Victory Gap** | How many additional votes does each candidate need to win (as % of total votes)? |
| **Ballot Exhaustion Impact** | Could completing exhausted ballots (voters who didn't rank all candidates) change the outcome? |
| **Strategic Complexity** | Is each candidate's optimal strategy simple self-support, or does it require boosting other candidates? |
| **Preference Order Alignment** | Does the elimination order match how competitive candidates actually are? |

## Installation

```bash
git clone https://github.com/sanyukta-D/Optimal_Strategies_in_RCV.git
cd Optimal_Strategies_in_RCV
pip install -e .
```

## Web Application

**[Live app: rcv-analyzer.streamlit.app](https://rcv-analyzer.streamlit.app/)**

Upload your own election data or try built-in examples (NYC, Alaska, Portland, Burlington, Minneapolis).

To run locally:
```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

The webapp is deployed from a [lightweight companion repo](https://github.com/sanyukta-D/RCV-Analyzer-Webapp).

## Usage

```python
from rcv_strategies.utils.case_study_helpers import get_ballot_counts_df, process_ballot_counts_post_elim_no_print
from rcv_strategies.core.stv_irv import IRV_optimal_result

# Load election data
ballot_counts = get_ballot_counts_df("path/to/election.csv", num_candidates=5)

# Run IRV to get the social choice order
results, event_log = IRV_optimal_result(ballot_counts)

# Compute optimal strategies for all candidates
strategies = process_ballot_counts_post_elim_no_print(ballot_counts, k=1, budget_percent=10.0)
```

See [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) for a detailed walkthrough.

## Repository Structure

```
├── rcv_strategies/           # Core Python package
│   ├── core/                 # STV/IRV algorithms, strategy computation, candidate removal
│   ├── utils/                # Ballot processing, case study helpers
│   └── analysis/             # Heatmaps, probability tables, comprehensive analysis
├── webapp/                   # Streamlit web application
├── case_studies/             # Election data and analysis scripts
│   ├── alaska/               # 53 statewide IRV elections (2024)
│   ├── nyc/                  # 78 municipal IRV elections (2021)
│   ├── portland/             # 4 city council districts, multi-winner STV (2024)
│   └── republican_primary/   # 2024 Republican Primary (FairVote poll)
├── ballot_exhaustion/        # 6 probability models for ballot exhaustion analysis
├── scripts/                  # Scripts to reproduce all paper results
├── notebooks/                # Interactive tutorial
└── results/                  # Generated figures and tables
```

## Reproducing Paper Results

```bash
# NYC and Alaska single-winner analysis (Tables 1, 7–11 in Paper 2)
python scripts/run_main.py

# Portland multi-winner STV analysis (Tables 2–6 in Paper 2)
python scripts/run_portland.py

# Republican Primary analysis (Tables in Paper 1)
python scripts/run_republican_primary.py
```

## Case Studies

Validated on **130+ real-world RCV elections** across four U.S. jurisdictions:

| Jurisdiction | Elections | Type | Year |
|-------------|-----------|------|------|
| **Alaska** | 53 | Single-winner IRV | 2024 |
| **New York City** | 78 | Single-winner IRV | 2021 |
| **Portland** | 4 districts | Multi-winner STV (k=3) | 2024 |
| **Republican Primary** | 1 (poll) | Single-winner IRV | 2024 |

## Algorithms

The core algorithmic contributions:

1. **Optimal strategy computation** — Enumerate all feasible elimination/winning structures to find the minimum-cost vote addition for any candidate to place in the top *k*.
2. **Candidate removal** — Trim irrelevant candidates in O(mn⁴) time while preserving optimization accuracy, making large elections tractable (Theorems 4.1, 4.3).
3. **Structure reduction** — Reduce the set of feasible sub-structures in O(mn²) time (Theorem 4.2).
4. **Ballot exhaustion models** — Six probability models (3 Beta-distribution, 3 Bootstrap) to estimate whether exhausted ballots could change outcomes.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{deshpande2024optimal,
  title={Optimal Strategies in Ranked Choice Voting},
  author={Deshpande, Sanyukta and Garg, Nikhil and Jacobson, Sheldon H.},
  year={2024}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
