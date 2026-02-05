#!/usr/bin/env python3
"""
Run a comprehensive analysis of all probability models across all RCV elections.

This script runs the analyze_all_elections function from probability_models.py
to compare six probability models:
1. Gap-Based Beta
2. Similarity Beta
3. Prior-Posterior Beta
4. Similarity Bootstrap (with 1000 iterations)
5. Rank-Restricted Bootstrap (with 1000 iterations)
6. Unconditional Bootstrap (with 1000 iterations)

It creates comprehensive visualizations comparing these models and outputs a detailed CSV
with all results.
"""

import probability_models as sraf
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive analysis of RCV elections')
    parser.add_argument('--bootstrap-iters', type=int, default=1000, 
                      help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--max-elections', type=int, default=None,
                      help='Maximum number of elections to analyze')
    args = parser.parse_args()
    
    print(f"Starting comprehensive analysis of all RCV elections")
    print(f"Bootstrap iterations: {args.bootstrap_iters}")
    print(f"Maximum elections: {args.max_elections if args.max_elections else 'All'}")
    
    # Record start time
    start_time = time.time()
    
    # Run the comprehensive analysis
    results = sraf.analyze_all_elections(bootstrap_iters=args.bootstrap_iters)
    
    # Calculate runtime
    runtime = time.time() - start_time
    print(f"\nAnalysis complete!")
    print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"Results saved to model_comparison_results/")

if __name__ == "__main__":
    main() 