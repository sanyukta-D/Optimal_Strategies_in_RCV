# Optimal Strategies in Ranked Choice Voting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A computational framework for finding optimal vote-addition strategies in ranked-choice voting (RCV) elections.

## Overview

This repository accompanies the paper:

> **Strategic Interactions in Modern Elections and Markets**
> Sanyukta Deshpande and Sheldon H. Jacobson
> University of Illinois at Urbana-Champaign

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

## Repository Structure

```
├── rcv_strategies/      # Core algorithms
│   ├── core/            # Optimization and strategy computation
│   ├── utils/           # Helper functions
│   └── analysis/        # Analysis tools (heatmaps, probability tables)
├── case_studies/        # Empirical analyses (Alaska, NYC, Portland)
├── notebooks/           # Interactive tutorials
├── scripts/             # Reproduction scripts
└── results/             # Generated outputs (not tracked)
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

We validate our framework on 110+ real-world RCV elections across:
- Alaska (statewide elections)
- New York City (municipal elections)
- Portland (city council elections)

## Citation

```bibtex
@phdthesis{deshpande2025strategic,
  title={Strategic Interactions in Modern Elections and Markets},
  author={Deshpande, Sanyukta},
  year={2025},
  school={University of Illinois at Urbana-Champaign},
  advisor={Jacobson, Sheldon H.}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
