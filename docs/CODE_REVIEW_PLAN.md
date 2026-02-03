# Code Review & Documentation Plan

## Status: COMPLETED

Documentation improvements have been implemented across all key files.

**Repository:** https://github.com/sanyukta-D/Optimal_Strategies_in_RCV

---

## Completed Changes

### 1. rcv_strategies/utils/case_study_helpers.py ✓

- **Module docstring**: Added comprehensive overview with key concepts, functions, and paper references
- **process_ballot_counts_post_elim_no_print()**: Enhanced docstring with:
  - Complete parameter documentation
  - Algorithm overview for multi-winner handling
  - Logic flow explanation (Steps 0/0b/A/B/C)
  - Tractability constraint explanation (< 9 candidates)
  - Example usage
- **Multi-winner early winner handling section**: Added detailed comments explaining:
  - Problem: filtered_data vs collection vote distributions
  - Solution: Using collection[small_num] for actual STV state
  - Key insight about k-1 seats (early winner occupies one seat)
  - Portland Dis 4 example with specific numbers
- **permit_STV_removal()**: Enhanced docstring explaining:
  - What the function checks (Theorem 4.2)
  - Parameters and return values
  - Usage pattern with small election method
- **convert_combination_strats_to_candidate_strats()**: Enhanced docstring with:
  - Input/output format examples
  - Explanation of why original winners always get gap=0

### 2. rcv_strategies/core/stv_irv.py ✓

- **Module docstring**: Added comprehensive documentation of:
  - All key functions
  - Critical data structures (ballot_counts, rt, dt, collection)
  - Quota calculation explanation
  - Paper references
- **STV_optimal_result_simple()**: Enhanced docstring explaining:
  - Return values (rt, dt, collection) with examples
  - Collection usage for small election method
  - Fractional vote counts from surplus transfers

### 3. rcv_strategies/core/candidate_removal.py ✓

- **Module docstring**: Added explanation of:
  - The tractability problem (> 8 candidates)
  - Theorem 4.1 (Basic Removal) and 4.3 (Rigorous Check)
  - Usage in strategy computation
- **remove_irrelevent()**: Enhanced docstring with:
  - Algorithm steps
  - Parameters and return values
  - Example from Portland Dis 4
  - Note about function name typo

### 4. rcv_strategies/core/strategy.py ✓

- **Module docstring**: Added overview of:
  - Core concept of structures (main/sub)
  - Algorithm steps
  - Output format for single vs multi-winner
  - Tractability constraint

### 5. webapp/app.py ✓

- **Analysis pipeline section**: Added overview comment explaining:
  - 5-step pipeline (convert, STV, remap, STV, strategies)
  - Large election handling
  - Multi-winner early winner reference
- **Divide-and-conquer section**: Added comments explaining:
  - Binary search for optimal budget
  - Two-phase approach (full set vs pre-filter)
  - Portland district examples

---

## Key Concepts Now Documented

### Tractability Constraint
- Strategy computation only works for < 9 candidates
- For larger elections, use remove_irrelevent() to reduce
- If reduction fails, webapp's binary search tries lower budget

### Multi-Winner Early Winner Handling (k > 1)
```
Steps when early winner detected:
Step 0:  Check collection[small_num] - if no early winner, use directly
Step 0b: If has winner, try one step back (collection[small_num-1])
Step A:  If has winner, try permit_STV_removal()
         If permitted → use "small election method" with k-1 seats
Step B:  If permit fails, try loopy (smaller candidate pools)
Step C:  If loopy exhausted, reset (budget doesn't work)
```

### Small Election Method (when permit passes)
```python
# Example: Portland Dis 4, k=3, A wins early
small_num = len(candidates) - len(candidates_retained)  # 26
ballot_counts_short = collection[small_num][0]  # A's surplus transferred
ordered_test = [rt[i][0] for i in range(small_num, len(rt))]  # [B, C, D, F]

# CRITICAL: Use k-1 because A already occupies one seat
strats_frame = reach_any_winners_campaign(ordered_test, k-1, Q, ballot_counts_short, budget)
strats_frame['A'] = [0, {}]  # Add A back with gap=0
```

### Collection Data Structure
```python
# collection[i] = ballot state after i STV events
# Events = eliminations + wins
# Vote counts may be fractional (Droop surplus transfers)

collection[0]  # Initial state
collection[26] # After 26 events (e.g., Portland Dis 4 reduced to 4 candidates)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `rcv_strategies/utils/case_study_helpers.py` | Module docstring, function docs, inline comments |
| `rcv_strategies/core/stv_irv.py` | Module docstring, STV_optimal_result_simple docs |
| `rcv_strategies/core/candidate_removal.py` | Module docstring, remove_irrelevent docs |
| `rcv_strategies/core/strategy.py` | Module docstring |
| `webapp/app.py` | Section comments for pipeline and binary search |

---

## Paper References Added

- Theorem 4.1 (Candidate Removal)
- Theorem 4.2 (Irrelevant Extension)
- Theorem 4.3 (Rigorous Check)
- Section 3 (Strategy Computation)
- Section 5.2 (Early Winner Handling)
