# Optimal Strategies in Ranked Choice Voting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A computational framework for finding optimal vote-addition strategies in ranked-choice voting (RCV) elections.

## Overview

We develop algorithms to:
1. Find optimal vote-addition strategies for any candidate to place in the top k
2. Trim irrelevant candidates under budget uncertainty in O(mn^4) time
3. Reduce feasible sub-structures in O(mn^2) time

## Installation

```bash
git clone https://github.com/sanyukta-D/Optimal_Strategies_in_RCV.git
cd Optimal_Strategies_in_RCV
pip install -r requirements.txt
```

## Web Application

**Try it online:** Upload your election data and get instant analysis!

https://rcv-analyzer.streamlit.app/

See https://github.com/sanyukta-D/RCV-Analyzer-Webapp for details.

## Repository Structure

```
├── webapp/                   # Web application (Streamlit)
├── rcv_strategies/           # Core algorithms
│   ├── core/                 # Optimization, strategy computation, candidate removal
│   ├── utils/                # Helper functions, case study utilities
│   └── analysis/             # Analysis tools (heatmaps, probability tables)
├── case_studies/             # Empirical analyses with data and scripts
│   ├── alaska/               # Alaska statewide elections (53 elections)
│   ├── nyc/                  # NYC municipal elections (78 elections)
│   ├── portland/             # Portland city council (4 districts, multi-winner STV)
│   └── republican_primary/   # Republican primary analysis
├── ballot_exhaustion/        # Ballot exhaustion analysis and probability models
├── notebooks/                # Interactive tutorials
├── scripts/                  # Reproduction scripts
└── results/                  # Generated outputs (not tracked)
```

## Usage

```python
from rcv_strategies.core import optimization, strategy

# See notebooks/tutorial.ipynb for detailed examples
```

## Reproducing Results

```bash
python scripts/run_main.py
python scripts/run_portland.py
```

## Case Studies

We validate our framework on 130+ real-world RCV elections across:
- **Alaska** - 53 statewide elections (IRV)
- **New York City** - 78 municipal elections (IRV)
- **Portland** - 4 city council districts (multi-winner STV)
- **Republican Primary** - 2024 Republican Primary Fairvote Poll analysis


## License

MIT License - see [LICENSE](LICENSE) for details.
