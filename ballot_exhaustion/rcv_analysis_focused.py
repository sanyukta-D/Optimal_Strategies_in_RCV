import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # Force Agg backend for better plot saving
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os
from case_studies.NewYork_City_RCV.convert_data import process_single_file, create_candidate_mapping
from rcv_strategies.utils import case_study_helpers
from string import ascii_uppercase
from rcv_strategies.core.stv_irv import STV_optimal_result_simple
from rcv_strategies.utils import helpers as utils
import math

def extract_strategy_dict(strategy_str):
    """Extract strategy dictionary from string"""
    try:
        strategy_dict = ast.literal_eval(strategy_str)
        if not isinstance(strategy_dict, dict):
            return {}
        
        # Extract main percentage for each candidate
        result = {}
        for candidate, data_list in strategy_dict.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                result[candidate] = data_list[0]
        return result
    except:
        return {}

def extract_exhaust_dict(exhaust_str):
    """Extract exhaust dictionary from string"""
    try:
        exhaust_dict = ast.literal_eval(exhaust_str)
        if not isinstance(exhaust_dict, dict):
            return {}
        return exhaust_dict
    except:
        return {}

def beta_parameters(gap_to_win_pct):
    """Calculate Beta parameters based on gap size"""
    # Base parameters for a perfectly tied race (centered at 0.5)
    base_param = 50.0
    
    # Maximum shift based on gap (capped at 40)
    max_shift = min(gap_to_win_pct * 0.5, 40)
    
    # Calculate parameters
    a = max(base_param - max_shift, 10.0)  # Parameter for B (challenger), minimum 10
    b = max(base_param + max_shift, 10.0)  # Parameter for A (leader), minimum 10
    
    return a, b

def beta_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using Beta distribution"""
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))
    
    # Convert from percentage to proportion
    required_proportion = required_preference_pct / 100
    
    a, b = beta_parameters(gap_to_win_pct)
    return 1 - stats.beta.cdf(required_proportion, a, b)

def normal_parameters(gap_to_win_pct):
    """Calculate Normal parameters based on gap size"""
    mean = 50 - gap_to_win_pct * 0.5
    std_dev = max(5, 10 - gap_to_win_pct * 0.5)
    return mean, std_dev

def normal_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using Normal distribution"""
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))
    
    mean, std_dev = normal_parameters(gap_to_win_pct)
    return 1 - stats.norm.cdf(required_preference_pct, mean, std_dev)

def uniform_probability(required_preference_pct, gap_to_win_pct):
    """Calculate probability using uniform normal distribution (normal centered at 50%)
    
    This models the assumption that exhaust preferences are distributed normally
    with mean at 50% (equal probability for either candidate) and standard deviation
    that varies based on the gap size.
    """
    # Ensure required_preference_pct is within bounds
    required_preference_pct = max(0, min(required_preference_pct, 100))
    
    # Fixed mean at 50% - unbiased preferences among exhausted ballots  
    estimated_preference = 50
    
    # Standard deviation based on gap size - tighter races have more variability
    std_dev = max(5, 10 - gap_to_win_pct * 0.3)
    
    # Calculate probability using normal distribution centered at 50%
    return 1 - stats.norm.cdf(required_preference_pct, estimated_preference, std_dev)

def extract_first_preferences(ballot_counts, candidates):
    """Extract first preferences from ballot counts"""
    first_prefs = {}
    for ballot, count in ballot_counts.items():
        if ballot:  # Non-empty ballot
            first_pref = ballot[0]
            if first_pref not in first_prefs:
                first_prefs[first_pref] = 0
            first_prefs[first_pref] += count
    return first_prefs

def calculate_preference_distributions(ballot_counts, candidates, first_pref):
    """Calculate B>A vs A>B distributions for ballots with given first preference"""
    b_over_a = 0
    a_over_b = 0
    total = 0
    
    for ballot, count in ballot_counts.items():
        if not ballot or ballot[0] != first_pref:
            continue
            
        # Check if both A and B are in ballot
        if 'A' in ballot and 'B' in ballot:
            total += count
            if ballot.index('B') < ballot.index('A'):
                b_over_a += count
            else:
                a_over_b += count
                
    return b_over_a, a_over_b, total

def calculate_similarity_score(exhausted_ballot, complete_ballot):
    """Calculate similarity between two ballots based on ranked positions"""
    # Convert ballots to sets for Jaccard similarity
    exhausted_set = set(exhausted_ballot)
    complete_set = set(complete_ballot)
    
    # Basic Jaccard similarity
    if not exhausted_set or not complete_set:
        return 0
    
    # Calculate overlap score
    overlap = len(exhausted_set.intersection(complete_set))
    union = len(exhausted_set.union(complete_set))
    jaccard = overlap / union
    
    # Position bonus - reward exact position matches
    position_bonus = 0
    for i, candidate in enumerate(exhausted_ballot):
        if i < len(complete_ballot) and complete_ballot[i] == candidate:
            position_bonus += 0.2  # Bonus for positional match
    
    # Combine scores (weighted toward exact positional matches)
    similarity = jaccard * 0.5 + position_bonus * 0.5
    return min(1.0, similarity)  # Cap at 1.0

def bayesian_beta_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using Beta-based Bayesian model with proper regularization
    
    This model applies empirical Bayes inference using ballot data to estimate the probability
    that candidate B is preferred over candidate A among exhausted ballots. It uses
    statistically principled regularization and uncertainty quantification.
    
    Parameters are aligned with the non-Bayesian beta model for consistency.
    """
    # Get gap size for parameter initialization
    gap_to_win_pct = abs(required_preference_pct - 50) * 2
    
    # Calculate theoretical parameters using the same function as non-Bayesian model
    a, b = beta_parameters(gap_to_win_pct)
    
    # Convert Beta parameters to prior mean and concentration
    prior_concentration = a + b
    prior_b = a / prior_concentration  # This is the prior probability that B is preferred
    
    # Extract data about B>A vs A>B preferences from all ballots for global prior
    all_b_over_a = 0
    all_a_over_b = 0
    
    # Get global preference data from all ballots
    for ballot, count in ballot_counts.items():
        if 'A' in ballot and 'B' in ballot:
            if ballot.index('B') < ballot.index('A'):
                all_b_over_a += count
            else:
                all_a_over_b += count
    
    # Calculate global empirical prior (with pseudocounts to avoid extremes)
    global_total = all_b_over_a + all_a_over_b
    if global_total > 0:
        # Get empirical proportion from data
        empirical_prior_b = all_b_over_a / global_total
        
        # Combine theoretical and empirical priors
        # Weight depends on data size - more data means more weight on empirical
        data_weight = min(0.8, global_total / 1000)
        prior_b = (1 - data_weight) * prior_b + data_weight * empirical_prior_b
    
    # Extract first preferences from exhausted ballots
    first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        return 0.5  # Neutral probability when no data

    # For each exhausted ballot type, estimate preference distributions
    weighted_b_over_a = 0
    weighted_a_over_b = 0
    
    # Process each first preference group in exhausted ballots
    for first_pref, count in first_prefs.items():
        if first_pref in ['A', 'B']:
            continue
        
        # Weight of this first preference among exhausted ballots
        weight = count / total_exhausted
        
        # Get preference distribution for this first preference group
        b_over_a, a_over_b, total = calculate_preference_distributions(ballot_counts, candidates, first_pref)
        
        # Set up the Beta prior based on gap and sample size
        if total > 0:
            # Calculate effective sample size - scales by log of actual sample size
            effective_size = 2 * math.log(total + 1)
        else:
            # With no data, rely more on prior
            effective_size = 2
            
        # Use the concentration from theoretical parameters
        prior_alpha = prior_concentration * prior_b
        prior_beta = prior_concentration * (1 - prior_b)
        
        # Update with observed data
        posterior_alpha = prior_alpha + b_over_a
        posterior_beta = prior_beta + a_over_b
        
        # Calculate posterior mean
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Add this group's contribution to the weighted total
        weighted_b_over_a += weight * posterior_mean * total_exhausted
        weighted_a_over_b += weight * (1 - posterior_mean) * total_exhausted
    
    # Calculate probability from weighted posteriors
    if weighted_b_over_a + weighted_a_over_b > 0:
        observed_b_pref_pct = 100 * weighted_b_over_a / (weighted_b_over_a + weighted_a_over_b)
        
        # Create full Beta distribution for final probability calculation
        n = weighted_b_over_a + weighted_a_over_b
        pseudocount_factor = 2 / (n + 2)  # Ensures influence scales with sample size
        
        final_alpha = weighted_b_over_a + pseudocount_factor
        final_beta = weighted_a_over_b + pseudocount_factor
        
        # Convert required_preference_pct to proportion
        required_proportion = required_preference_pct / 100
        
        # Calculate probability from Beta CDF
        raw_probability = 1 - stats.beta.cdf(required_proportion, final_alpha, final_beta)
        
        # Apply slight shrinkage toward 0.5 to avoid extreme probabilities
        # Higher shrinkage for large gaps where we have less certainty
        shrinkage_factor = min(1.0, n / (100 + gap_to_win_pct))
        probability = 0.5 + (raw_probability - 0.5) * shrinkage_factor
    else:
        probability = 0.5  # Default when no information
        observed_b_pref_pct = 50
    
    print(f"[Bayesian Beta] weighted_b_over_a={weighted_b_over_a:.2f}, weighted_a_over_b={weighted_a_over_b:.2f}, " +
          f"obs_pref={observed_b_pref_pct:.2f}%, req_pref={required_preference_pct:.2f}%, prob={probability:.4f}")
    
    return probability

def bayesian_normal_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using Normal-based Bayesian model with James-Stein shrinkage
    
    This model uses a normal distribution to model preferences between candidates,
    with parameters updated based on ballot data and regularized using proper 
    statistical principles for robust uncertainty estimation.
    
    Parameters are aligned with the non-Bayesian normal model for consistency.
    """
    # Get gap size for parameter initialization
    gap_to_win_pct = abs(required_preference_pct - 50) * 2
    
    # Calculate prior parameters using the same function as non-Bayesian model
    prior_mean, prior_std = normal_parameters(gap_to_win_pct)
    
    # Extract data for global empirical statistics
    all_b_over_a = 0
    all_a_over_b = 0
    group_preferences = {}
    
    # Get global preference data and group-specific data
    for ballot, count in ballot_counts.items():
        if 'A' in ballot and 'B' in ballot:
            if ballot.index('B') < ballot.index('A'):
                all_b_over_a += count
            else:
                all_a_over_b += count
                
            # Also track by first preference for group-level data
            if ballot:
                first_pref = ballot[0]
                if first_pref not in group_preferences:
                    group_preferences[first_pref] = {'b_over_a': 0, 'a_over_b': 0}
                
                if ballot.index('B') < ballot.index('A'):
                    group_preferences[first_pref]['b_over_a'] += count
                else:
                    group_preferences[first_pref]['a_over_b'] += count
    
    # Calculate global empirical mean (as percentage)
    global_total = all_b_over_a + all_a_over_b
    if global_total > 0:
        global_empirical_mean = 100 * (all_b_over_a / global_total)
        # Combine theoretical prior with empirical data
        # Weight depends on data size - more data means more weight on empirical
        data_weight = min(0.8, global_total / 1000)
        global_mean = (1 - data_weight) * prior_mean + data_weight * global_empirical_mean
    else:
        # Use theoretical prior when no data
        global_mean = prior_mean
    
    # Extract first preferences from exhausted ballots
    first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        return 0.5  # Neutral probability when no data

    # Calculate group-level parameters with shrinkage toward global mean
    group_params = {}
    for group, data in group_preferences.items():
        group_total = data['b_over_a'] + data['a_over_b']
        if group_total > 0:
            # Raw mean for this group
            group_mean = 100 * (data['b_over_a'] / group_total)
            
            # Apply James-Stein shrinkage toward global mean
            # The shrinkage factor is inversely proportional to group size
            js_factor = 1 / (1 + group_total / 100)  # 100 is the regularization strength
            shrunk_mean = global_mean + (1 - js_factor) * (group_mean - global_mean)
            
            # Standard deviation based on prior_std but adjusted for sample size
            # Start with prior std and reduce based on sample size
            std_dev = max(5, prior_std / math.sqrt(1 + group_total / 100))
            
            group_params[group] = {'mean': shrunk_mean, 'std': std_dev}
    
    # For each exhausted ballot type, get preference distributions
    final_mean = 0
    final_precision = 0  # Precision = 1/variance
    
    # Process each first preference group in exhausted ballots
    for first_pref, count in first_prefs.items():
        if first_pref in ['A', 'B']:
            continue
        
        # Weight of this first preference among exhausted ballots
        weight = count / total_exhausted
        
        # Get parameters for this group, or use global parameters if not available
        if first_pref in group_params:
            group_mean = group_params[first_pref]['mean']
            group_std = group_params[first_pref]['std']
        else:
            # If we don't have data for this first preference, use global with higher uncertainty
            group_mean = global_mean
            group_std = prior_std  # Use prior std instead of arbitrary value
        
        # Calculate precision (1/variance)
        precision = 1 / (group_std ** 2)
        
        # Add weighted contribution to final parameters
        # This is a standard Bayesian update for normal distribution with known variance
        final_mean += weight * group_mean * precision
        final_precision += weight * precision
    
    # Normalize the final mean and calculate standard deviation
    if final_precision > 0:
        final_mean = final_mean / final_precision
        final_std = math.sqrt(1 / final_precision)
    else:
        final_mean = prior_mean  # Use prior mean instead of 50
        final_std = prior_std    # Use prior std instead of arbitrary value
    
    # Apply a minimum standard deviation to prevent overconfidence
    final_std = max(final_std, 5)
    
    # Calculate probability using normal CDF
    # Convert required_preference_pct to standard score (z-score)
    z_score = (final_mean - required_preference_pct) / final_std
    
    # Calculate raw probability using normal CDF
    raw_probability = stats.norm.cdf(z_score)
    
    # Apply smoothing to prevent extreme probabilities
    # This gradually returns probability toward 0.5 for z-scores near zero
    smoothing_factor = 1 - math.exp(-abs(z_score) / 1.5)
    probability = 0.5 + (raw_probability - 0.5) * smoothing_factor
    
    print(f"[Bayesian Normal] final_mean={final_mean:.2f}%, final_std={final_std:.2f}, " +
          f"req_pref={required_preference_pct:.2f}%, z={z_score:.2f}, prob={probability:.4f}")
    
    return probability

def similarity_bayesian_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using similarity-based Bayesian model with proper statistical grounding
    
    This model uses ballot similarity metrics based on nearest-neighbor principles to weight 
    completed ballots based on their similarity to exhausted ballots, providing a theoretically 
    grounded approach to inferring preferences in exhausted ballots with appropriate uncertainty.
    """
    # Early return if no exhausted ballots
    if not exhausted_ballots:
        return 0.5
    
    # Get preferences from all exhausted ballots
    exhausted_first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in exhausted_first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        return 0.5  # Neutral probability when no data
    
    # Calculate similarity-weighted preferences
    b_over_a_weighted_sum = 0
    a_over_b_weighted_sum = 0
    total_weighted = 0
    
    # Get all complete ballots with both A and B ranked
    complete_ballots = {}
    for ballot, count in ballot_counts.items():
        if 'A' in ballot and 'B' in ballot:
            complete_ballots[ballot] = count
    
    # Get all candidates except A and B for similarity calculations
    other_candidates = [c for c in candidates if c not in ['A', 'B']]
    
    # Process each exhausted ballot type
    for exh_ballot, exh_count in exhausted_ballots.items():
        # Skip if the exhausted ballot already has both A and B
        if 'A' in exh_ballot and 'B' in exh_ballot:
            continue
            
        # Calculate similarities with complete ballots
        similarity_scores = []
        
        # For each complete ballot, calculate similarity score
        for comp_ballot, comp_count in complete_ballots.items():
            similarity = calculate_similarity_score(exh_ballot, comp_ballot)
            
            # Only consider ballots with non-zero similarity
            if similarity > 0:
                similarity_scores.append({
                    'ballot': comp_ballot,
                    'similarity': similarity,
                    'count': comp_count,
                    'b_over_a': 1 if comp_ballot.index('B') < comp_ballot.index('A') else 0
                })
        
        # If no similar ballots found, skip this exhausted ballot
        if not similarity_scores:
            continue
            
        # Sort by similarity (highest first)
        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Use k-nearest neighbor approach with weighted voting
        # Determine k based on ballot data size (more data = more neighbors)
        total_complete = sum(comp_count for _, comp_count in complete_ballots.items())
        k = min(len(similarity_scores), max(5, int(math.log(1 + total_complete))))
        
        # Get top-k most similar ballots
        top_k = similarity_scores[:k]
        
        # Calculate weighted contribution from top-k similar ballots
        # Use a distance-weighted voting scheme
        b_over_a_sum = 0
        a_over_b_sum = 0
        total_similarity = 0
        
        for entry in top_k:
            # Weight by both similarity and ballot count (frequency)
            # Square the similarity to emphasize closer matches
            weight = entry['similarity'] ** 2 * entry['count']
            total_similarity += weight
            
            # Add preference contribution
            if entry['b_over_a'] == 1:
                b_over_a_sum += weight
            else:
                a_over_b_sum += weight
        
        # Normalize and scale by exhausted ballot count
        if total_similarity > 0:
            # Calculate proportions with proper regularization
            # Add pseudocounts to avoid extreme values
            pseudocount = 1 * (k / 10)  # Scales with neighborhood size
            
            b_prop = (b_over_a_sum + pseudocount) / (total_similarity + 2 * pseudocount)
            a_prop = (a_over_b_sum + pseudocount) / (total_similarity + 2 * pseudocount)
            
            # Add weighted contribution to overall tally
            b_over_a_weighted_sum += b_prop * exh_count
            a_over_b_weighted_sum += a_prop * exh_count
            total_weighted += exh_count
    
    # Calculate final probability
    if total_weighted > 0:
        # Calculate observed percentage
        observed_b_pref_pct = 100 * b_over_a_weighted_sum / (b_over_a_weighted_sum + a_over_b_weighted_sum)
        
        # Create Beta distribution for final probability calculation
        # We use pseudocounts for regularization based on data amount
        n = total_weighted
        pseudocount_factor = 2 / (n + 2)  # Ensures influence scales with sample size
        
        alpha = b_over_a_weighted_sum + pseudocount_factor
        beta = a_over_b_weighted_sum + pseudocount_factor
        
        # Convert required preference to proportion
        req_proportion = required_preference_pct / 100
        
        # Calculate probability using Beta CDF
        raw_probability = 1 - stats.beta.cdf(req_proportion, alpha, beta)
        
        # Apply smoothing toward 0.5 for smaller differences
        z_score = (observed_b_pref_pct - required_preference_pct) / 10  # Using fixed 10% std as baseline
        smoothing_factor = 1 - math.exp(-abs(z_score) / 1.5)
        probability = 0.5 + (raw_probability - 0.5) * smoothing_factor
    else:
        # Default to neutral when no evidence
        probability = 0.5
        observed_b_pref_pct = 50
    
    print(f"[Similarity] b_over_a={b_over_a_weighted_sum:.2f}, a_over_b={a_over_b_weighted_sum:.2f}, " +
          f"obs_pref={observed_b_pref_pct:.2f}%, req_pref={required_preference_pct:.2f}%, prob={probability:.4f}")
    
    return probability

def direct_posterior_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using only observed data, without theoretical priors or regularization.
    
    This model uses direct evidence from observed ballot patterns to estimate 
    the probability that exhausted ballots would prefer candidate B over A.
    """
    # Extract first preferences from exhausted ballots
    first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        # No exhausted ballots to analyze
        return 0.5  # Return neutral probability when no data
    
    # For each exhausted ballot type, find exact preference patterns in complete ballots
    weighted_b_over_a = 0
    weighted_a_over_b = 0
    total_weighted = 0
    missing_data = 0
    
    # Process each first preference group in exhausted ballots
    for first_pref, count in first_prefs.items():
        if first_pref in ['A', 'B']:
            continue
        
        # Calculate weight as proportion of this preference among exhausted ballots
        weight = count / total_exhausted
        
        # Get preference distribution for this first preference group
        b_over_a, a_over_b, total = calculate_preference_distributions(ballot_counts, candidates, first_pref)
        
        if total > 0:
            # Direct calculation of preferences without smoothing
            b_over_a_prop = b_over_a / total
            a_over_b_prop = a_over_b / total
            
            # Weighted contribution based on exhausted ballot proportion
            weighted_b_over_a += weight * b_over_a_prop * total_exhausted
            weighted_a_over_b += weight * a_over_b_prop * total_exhausted
            total_weighted += weight * total_exhausted
        else:
            # Track missing data for reporting
            missing_data += weight * total_exhausted
    
    # Calculate the raw probability directly from data
    if total_weighted > 0:
        # Calculate observed B>A proportion
        observed_b_pref_pct = 100 * weighted_b_over_a / (weighted_b_over_a + weighted_a_over_b)
        raw_prob = observed_b_pref_pct / 100
        
        # Compare with required preference
        if required_preference_pct > observed_b_pref_pct:
            # B cannot win based on observed data
            probability = 0
        else:
            # B would win based on observed data
            probability = 1
    else:
        # No matching preference data found
        raw_prob = 0.5  # Default value when no data
        probability = 0.5  # Neutral when no evidence
    
    # Print diagnostics
    print(f"[Direct Posterior] b_over_a={weighted_b_over_a:.2f}, a_over_b={weighted_a_over_b:.2f}, raw_prob={raw_prob:.4f}, req_pref={required_preference_pct:.2f}, missing_data={missing_data:.2f}")
    
    return probability

def direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using a Beta distribution fitted to observed data.
    
    This model uses direct evidence from observed ballot patterns but applies a Beta
    distribution to model uncertainty, producing a continuous probability.
    """
    # Extract first preferences from exhausted ballots
    first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        # No exhausted ballots to analyze
        return 0.5  # Return neutral probability when no data
    
    # For each exhausted ballot type, find exact preference patterns in complete ballots
    weighted_b_over_a = 0
    weighted_a_over_b = 0
    total_weighted = 0
    missing_data = 0
    
    # Process each first preference group in exhausted ballots
    for first_pref, count in first_prefs.items():
        if first_pref in ['A', 'B']:
            continue
        
        # Calculate weight as proportion of this preference among exhausted ballots
        weight = count / total_exhausted
        
        # Get preference distribution for this first preference group
        b_over_a, a_over_b, total = calculate_preference_distributions(ballot_counts, candidates, first_pref)
        
        if total > 0:
            # Direct calculation of preferences without smoothing
            b_over_a_prop = b_over_a / total
            a_over_b_prop = a_over_b / total
            
            # Weighted contribution based on exhausted ballot proportion
            weighted_b_over_a += weight * b_over_a_prop * total_exhausted
            weighted_a_over_b += weight * a_over_b_prop * total_exhausted
            total_weighted += weight * total_exhausted
        else:
            # Track missing data for reporting
            missing_data += weight * total_exhausted
    
    # Calculate probability using Beta distribution
    if total_weighted > 0:
        # Use observed counts as alpha and beta parameters for Beta distribution
        # Add small pseudocounts (1) to avoid zero probabilities
        alpha = weighted_b_over_a + 1
        beta = weighted_a_over_b + 1
        
        # Convert required preference to proportion
        req_proportion = required_preference_pct / 100
        
        # Calculate probability using Beta CDF
        probability = 1 - stats.beta.cdf(req_proportion, alpha, beta)
        
        # For reporting
        observed_b_pref_pct = 100 * weighted_b_over_a / (weighted_b_over_a + weighted_a_over_b)
    else:
        # No matching preference data found
        probability = 0.5  # Neutral when no evidence
        observed_b_pref_pct = 50
    
    # Print diagnostics
    print(f"[Direct Posterior Beta] b_over_a={weighted_b_over_a:.2f}, a_over_b={weighted_a_over_b:.2f}, " +
          f"obs_pref={observed_b_pref_pct:.2f}%, req_pref={required_preference_pct:.2f}%, " +
          f"alpha={alpha:.2f}, beta={beta:.2f}, prob={probability:.4f}")
    
    return probability

def direct_posterior_normal(required_preference_pct, ballot_counts, candidates, exhausted_ballots):
    """Calculate probability using a Normal distribution fitted to observed data with proper uncertainty.
    
    This model uses direct evidence from observed ballot patterns but applies a properly
    regularized Normal distribution to model uncertainty, producing a continuous probability
    that reflects both statistical and domain-specific uncertainty.
    """
    # Get gap size for theoretical parameters
    gap_to_win_pct = abs(required_preference_pct - 50) * 2
    
    # Calculate theoretical parameters based on gap size
    theoretical_mean, theoretical_std = normal_parameters(gap_to_win_pct)
    
    # Extract first preferences from exhausted ballots
    first_prefs = extract_first_preferences(exhausted_ballots, candidates)
    total_exhausted = sum(count for pref, count in first_prefs.items() if pref not in ['A', 'B'])
    
    if total_exhausted == 0:
        # No exhausted ballots to analyze
        return 0.5  # Return neutral probability when no data
    
    # For each exhausted ballot type, find exact preference patterns in complete ballots
    weighted_b_over_a = 0
    weighted_a_over_b = 0
    total_weighted = 0
    missing_data = 0
    
    # Process each first preference group in exhausted ballots
    for first_pref, count in first_prefs.items():
        if first_pref in ['A', 'B']:
            continue
            
        # Calculate weight as proportion of this preference among exhausted ballots
        weight = count / total_exhausted
        
        # Get preference distribution for this first preference group
        b_over_a, a_over_b, total = calculate_preference_distributions(ballot_counts, candidates, first_pref)
        
        if total > 0:
            # Direct calculation of preferences without smoothing
            b_over_a_prop = b_over_a / total
            a_over_b_prop = a_over_b / total
            
            # Weighted contribution based on exhausted ballot proportion
            weighted_b_over_a += weight * b_over_a_prop * total_exhausted
            weighted_a_over_b += weight * a_over_b_prop * total_exhausted
            total_weighted += weight * total_exhausted
        else:
            # Track missing data for reporting
            missing_data += weight * total_exhausted
    
    # Calculate probability using Normal distribution with proper uncertainty
    if total_weighted > 0:
        # Calculate observed percentage
        observed_b_pref_pct = 100 * weighted_b_over_a / (weighted_b_over_a + weighted_a_over_b)
        
        # Calculate empirical standard error
        p = observed_b_pref_pct / 100  # Convert to proportion
        effective_sample_size = weighted_b_over_a + weighted_a_over_b
        
        # Basic binomial standard error (as percentage)
        binomial_std_error = 100 * math.sqrt((p * (1 - p)) / effective_sample_size)
        
        # Combine empirical standard error with theoretical model standard deviation
        # Weight based on effective sample size - more data means more weight on empirical
        data_weight = min(0.8, effective_sample_size / 1000)
        combined_std = (1 - data_weight) * theoretical_std + data_weight * binomial_std_error
        
        # Add uncertainty floor - even with infinite samples, preferences have inherent uncertainty
        min_uncertainty = 5.0  # Minimum 5% uncertainty due to inherent preference variability
        
        # Additional uncertainty based on missing data proportion
        missing_data_factor = math.sqrt(1 + (missing_data / total_exhausted))
        
        # Add uncertainty for extreme probabilities (very high or very low)
        extreme_prob_factor = 1 + 0.5 * math.exp(-10 * abs(p - 0.5))
        
        # Final standard error with all factors
        std_error = max(
            combined_std * missing_data_factor * extreme_prob_factor,
            min_uncertainty
        )
        
        # Combine observed mean with theoretical mean
        # More weight on observed for larger sample sizes
        combined_mean = (1 - data_weight) * theoretical_mean + data_weight * observed_b_pref_pct
        
        # Calculate probability using Normal CDF with regularized uncertainty
        z_score = (combined_mean - required_preference_pct) / std_error
        
        # Apply a smoothing function to the probability near the threshold
        # This prevents unrealistic certainty for small differences
        raw_probability = stats.norm.cdf(z_score)
        
        # Smooth probability closer to 0.5 when z-score is small
        # Stronger smoothing effect for large gaps (less certainty)
        smoothing_strength = 1.5 + gap_to_win_pct / 20  # Ranges from 1.5 to ~2.5
        smoothing_factor = 1 - math.exp(-abs(z_score) / smoothing_strength)
        probability = 0.5 + (raw_probability - 0.5) * smoothing_factor
        
    else:
        # No matching preference data found, use theoretical model
        z_score = (theoretical_mean - required_preference_pct) / theoretical_std
        raw_probability = stats.norm.cdf(z_score)
        
        # Strong smoothing toward 0.5 since we have no data
        smoothing_factor = 0.5
        probability = 0.5 + (raw_probability - 0.5) * smoothing_factor
        
        # For reporting only
        observed_b_pref_pct = theoretical_mean
        std_error = theoretical_std
    
    # Print diagnostics
    print(f"[Direct Posterior Normal] b_over_a={weighted_b_over_a:.2f}, a_over_b={weighted_a_over_b:.2f}, " +
          f"obs_pref={observed_b_pref_pct:.2f}%, req_pref={required_preference_pct:.2f}%, " +
          f"std_error={std_error:.2f}, prob={probability:.4f}")
    
    return probability

def process_election_data():
    """Process NYC and Alaska election data, filtering for exhaust > strategy only"""
    print("Loading and processing election data...")
    
    # Create output directory
    os.makedirs('figures_focused', exist_ok=True)
    
    try:
        # Load NYC data
        nyc_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_nyc_final.xlsx"
        nyc_df = pd.read_excel(nyc_path)
        nyc_df = nyc_df[nyc_df['file_name'].str.contains("DEM", na=False)].copy()
        print(f"Loaded NYC data: {len(nyc_df)} rows")
        
        # Load NYC ballot data
        nyc_ballot_data = {}
        nyc_folder = "Case_Studies/NewYork_City_RCV/nyc_files"
        for input_file in os.listdir(nyc_folder):
            if input_file.endswith('.csv'):
                full_path = os.path.join(nyc_folder, input_file)
                df, processed_file = process_single_file(full_path)
                candidates_mapping = create_candidate_mapping(processed_file)
                ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
                nyc_ballot_data[input_file] = {
                    'ballot_counts': ballot_counts,
                    'candidates': list(candidates_mapping.values()),
                    'df': df,
                    'candidates_mapping': candidates_mapping
                }
    except Exception as e:
        print(f"Error loading NYC data: {e}")
        nyc_df = pd.DataFrame()
        nyc_ballot_data = {}
    
    try:
        # Load Alaska data
        alaska_path = "/Users/saeesbox/Desktop/Social_Choice_Work/codes/EC_codes/Optimal_Strategies_in_RCV/summary_table_alska_lite.xlsx"
        alaska_df = pd.read_excel(alaska_path)
        print(f"Loaded Alaska data: {len(alaska_df)} rows")
        
        # Load Alaska ballot data
        alaska_ballot_data = {}
        alaska_folder = "Case_Studies/Alaska_RCV/alaska_files"
        for input_file in os.listdir(alaska_folder):
            if input_file.endswith('.csv'):
                full_path = os.path.join(alaska_folder, input_file)
                df, processed_file = process_single_file(full_path)
                candidates_mapping = create_candidate_mapping(processed_file)
                ballot_counts = case_study_helpers.get_ballot_counts_df(candidates_mapping, df)
                alaska_ballot_data[input_file] = {
                    'ballot_counts': ballot_counts,
                    'candidates': list(candidates_mapping.values()),
                    'df': df,
                    'candidates_mapping': candidates_mapping
                }
    except Exception as e:
        print(f"Error loading Alaska data: {e}")
        alaska_df = pd.DataFrame()
        alaska_ballot_data = {}
    
    # Process NYC data - ONLY where exhaust > strategy
    nyc_data = []
    for idx, row in nyc_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                print(f"Skipping NYC row {idx}: Missing strategy or exhaust data")
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    
                    # ONLY include cases where exhaust > strategy
                    if exhaust_val > strategy_val:
                        diff = exhaust_val - strategy_val
                        
                        # Get ballot data for this election
                        ballot_data = nyc_ballot_data.get(row['file_name'], {})
                        ballot_counts = ballot_data.get('ballot_counts', {})
                        candidates = ballot_data.get('candidates', [])
                        df = ballot_data.get('df', None)
                        candidates_mapping = ballot_data.get('candidates_mapping', {})
                        
                        if df is None or not candidates_mapping:
                            print(f"Missing data for NYC file: {row['file_name']}")
                            continue
                        
                        # Create final mapping and updated ballot counts
                        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, 1, sum(ballot_counts.values())/(1+1))
                        results, subresults = utils.return_main_sub(rt)
                        reverse_mapping = {code: name for name, code in candidates_mapping.items()}
                        ordered_candidate_names = [reverse_mapping[code] for code in results]
                        final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
                        updated_ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
                        
                        # Extract exhausted ballots
                        exhausted_ballots = {ballot: count for ballot, count in updated_ballot_counts.items() if 'A' not in ballot and 'B' not in ballot}
                        
                        nyc_data.append({
                            'letter': letter,
                            'diff': diff,
                            'strategy': strategy_val,
                            'exhaust': exhaust_val,
                            'region': 'NYC',
                            'election_id': row.get('file_name', f"NYC_{idx}"),
                            'ballot_counts': updated_ballot_counts,
                            'candidates': list(final_mapping.values()),
                            'exhausted_ballots': exhausted_ballots
                        })
                        print(f"Processed NYC row {idx}, letter {letter}: exhaust={exhaust_val}, strategy={strategy_val}, diff={diff}")
        except Exception as e:
            print(f"Error processing NYC row {idx}: {e}")
    
    # Process Alaska data - ONLY where exhaust > strategy
    alaska_data = []
    for idx, row in alaska_df.iterrows():
        try:
            strategy_dict = extract_strategy_dict(row['Strategies'])
            exhaust_dict = extract_exhaust_dict(row['exhaust_percents'])
            
            # Skip if no data
            if not strategy_dict or not exhaust_dict:
                print(f"Skipping Alaska row {idx}: Missing strategy or exhaust data")
                continue
            
            # Process each letter (excluding A)
            for letter, strategy_val in strategy_dict.items():
                if letter == 'A':
                    continue
                
                if letter in exhaust_dict:
                    exhaust_val = exhaust_dict[letter]
                    
                    # ONLY include cases where exhaust > strategy
                    if exhaust_val > strategy_val:
                        diff = exhaust_val - strategy_val
                        
                        # Get ballot data for this election
                        ballot_data = alaska_ballot_data.get(row['file_name'], {})
                        ballot_counts = ballot_data.get('ballot_counts', {})
                        candidates = ballot_data.get('candidates', [])
                        df = ballot_data.get('df', None)
                        candidates_mapping = ballot_data.get('candidates_mapping', {})
                        
                        if df is None or not candidates_mapping:
                            print(f"Missing data for Alaska file: {row['file_name']}")
                            continue
                        
                        # Create final mapping and updated ballot counts
                        rt, dt, collection = STV_optimal_result_simple(candidates, ballot_counts, 1, sum(ballot_counts.values())/(1+1))
                        results, subresults = utils.return_main_sub(rt)
                        reverse_mapping = {code: name for name, code in candidates_mapping.items()}
                        ordered_candidate_names = [reverse_mapping[code] for code in results]
                        final_mapping = {candidate: ascii_uppercase[i] for i, candidate in enumerate(ordered_candidate_names)}
                        updated_ballot_counts = case_study_helpers.get_ballot_counts_df(final_mapping, df)
                        
                        # Extract exhausted ballots
                        exhausted_ballots = {ballot: count for ballot, count in updated_ballot_counts.items() if 'A' not in ballot and 'B' not in ballot}
                        
                        alaska_data.append({
                            'letter': letter,
                            'diff': diff,
                            'strategy': strategy_val,
                            'exhaust': exhaust_val,
                            'region': 'Alaska',
                            'election_id': row.get('file_name', f"Alaska_{idx}"),
                            'ballot_counts': updated_ballot_counts,
                            'candidates': list(final_mapping.values()),
                            'exhausted_ballots': exhausted_ballots
                        })
                        print(f"Processed Alaska row {idx}, letter {letter}: exhaust={exhaust_val}, strategy={strategy_val}, diff={diff}")
        except Exception as e:
            print(f"Error processing Alaska row {idx}: {e}")
    
    # Create DataFrames
    nyc_df_analysis = pd.DataFrame(nyc_data)
    alaska_df_analysis = pd.DataFrame(alaska_data)
    
    print(f"NYC analysis DataFrame (exhaust > strategy only): {len(nyc_df_analysis)} rows")
    print(f"Alaska analysis DataFrame (exhaust > strategy only): {len(alaska_df_analysis)} rows")
    
    # Save preprocessed data
    nyc_df_analysis.to_csv('nyc_focused_analysis.csv', index=False)
    alaska_df_analysis.to_csv('alaska_focused_analysis.csv', index=False)
    
    return nyc_df_analysis, alaska_df_analysis

def calculate_probabilities(nyc_df, alaska_df):
    """Calculate probabilities for each candidate using various models"""
    print("Calculating probabilities...")
    
    probability_results = []
    
    # Process NYC data
    if not nyc_df.empty:
        for idx, row in nyc_df.iterrows():
            letter = row['letter']
            gap_to_win_pct = row['strategy']
            exhaust_pct = row['exhaust']
            ballot_counts = row['ballot_counts']
            candidates = row['candidates']
            exhausted_ballots = row['exhausted_ballots']
            
            # Skip if exhaust is too small
            if exhaust_pct <= 0.01:
                continue
                
            # Calculate required net advantage
            required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
            
            # Calculate required preference percentage
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            # Calculate model probabilities
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            uniform_prob = uniform_probability(required_preference_pct, gap_to_win_pct)
            
            # Calculate Bayesian probabilities
            bayesian_beta_prob = bayesian_beta_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            bayesian_normal_prob = bayesian_normal_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Calculate Similarity Bayesian probability
            similarity_bayesian_prob = similarity_bayesian_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Calculate Direct Posterior probabilities
            direct_posterior_prob = direct_posterior_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_posterior_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_posterior_normal_prob = direct_posterior_normal(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Combined probability - weighted average of all models
            # Give slightly higher weight to models that leverage ballot data
            weights = [1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
            weight_sum = sum(weights)
            combined_prob = (
                weights[0] * beta_prob + 
                weights[1] * normal_prob + 
                weights[2] * uniform_prob + 
                weights[3] * bayesian_beta_prob + 
                weights[4] * bayesian_normal_prob + 
                weights[5] * similarity_bayesian_prob +
                weights[6] * direct_posterior_prob +
                weights[7] * direct_posterior_beta_prob +
                weights[8] * direct_posterior_normal_prob
            ) / weight_sum
            
            # Calculate strategy/exhaust ratio
            strategy_exhaust_ratio = gap_to_win_pct / exhaust_pct
            
            # Record results
            probability_results.append({
                'region': 'NYC',
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_preference_pct': required_preference_pct,
                'strategy_exhaust_ratio': strategy_exhaust_ratio,
                'beta_probability': beta_prob,
                'normal_probability': normal_prob,
                'uniform_probability': uniform_prob,
                'bayesian_beta_probability': bayesian_beta_prob,
                'bayesian_normal_probability': bayesian_normal_prob,
                'similarity_bayesian_probability': similarity_bayesian_prob,
                'direct_posterior_probability': direct_posterior_prob,
                'direct_posterior_beta_probability': direct_posterior_beta_prob,
                'direct_posterior_normal_probability': direct_posterior_normal_prob,
                'combined_probability': combined_prob
            })
    
    # Process Alaska data
    if not alaska_df.empty:
        for idx, row in alaska_df.iterrows():
            letter = row['letter']
            gap_to_win_pct = row['strategy']
            exhaust_pct = row['exhaust']
            ballot_counts = row['ballot_counts']
            candidates = row['candidates']
            exhausted_ballots = row['exhausted_ballots']
            
            # Skip if exhaust is too small
            if exhaust_pct <= 0.01:
                continue
                
            # Calculate required net advantage
            required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
            
            # Calculate required preference percentage
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            # Calculate model probabilities
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            uniform_prob = uniform_probability(required_preference_pct, gap_to_win_pct)
            
            # Calculate Bayesian probabilities
            bayesian_beta_prob = bayesian_beta_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            bayesian_normal_prob = bayesian_normal_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Calculate Similarity Bayesian probability
            similarity_bayesian_prob = similarity_bayesian_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Calculate Direct Posterior probabilities
            direct_posterior_prob = direct_posterior_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_posterior_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_posterior_normal_prob = direct_posterior_normal(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Combined probability - weighted average of all models
            # Give slightly higher weight to models that leverage ballot data
            weights = [1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
            weight_sum = sum(weights)
            combined_prob = (
                weights[0] * beta_prob + 
                weights[1] * normal_prob + 
                weights[2] * uniform_prob + 
                weights[3] * bayesian_beta_prob + 
                weights[4] * bayesian_normal_prob + 
                weights[5] * similarity_bayesian_prob +
                weights[6] * direct_posterior_prob +
                weights[7] * direct_posterior_beta_prob +
                weights[8] * direct_posterior_normal_prob
            ) / weight_sum
            
            # Calculate strategy/exhaust ratio
            strategy_exhaust_ratio = gap_to_win_pct / exhaust_pct
            
            # Record results
            probability_results.append({
                'region': 'Alaska',
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_preference_pct': required_preference_pct,
                'strategy_exhaust_ratio': strategy_exhaust_ratio,
                'beta_probability': beta_prob,
                'normal_probability': normal_prob,
                'uniform_probability': uniform_prob,
                'bayesian_beta_probability': bayesian_beta_prob,
                'bayesian_normal_probability': bayesian_normal_prob,
                'similarity_bayesian_probability': similarity_bayesian_prob,
                'direct_posterior_probability': direct_posterior_prob,
                'direct_posterior_beta_probability': direct_posterior_beta_prob,
                'direct_posterior_normal_probability': direct_posterior_normal_prob,
                'combined_probability': combined_prob
            })
    
    probability_df = pd.DataFrame(probability_results)
    probability_df.to_csv('probability_results_focused.csv', index=False)
    
    return probability_df

def create_visualizations(nyc_df, alaska_df, probability_df):
    """Create visualizations for the analysis"""
    print("Creating visualizations...")

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 12})
    
    # Create output directory
    os.makedirs('figures_focused', exist_ok=True)
    
    # Set up models, labels, and colors for consistent visualization
    models = ['beta_probability', 'normal_probability', 'uniform_probability', 
             'bayesian_beta_probability', 'bayesian_normal_probability', 
             'similarity_bayesian_probability', 'direct_posterior_probability', 'combined_probability']
    model_labels = ['Beta', 'Normal', 'Neutral', 'Bayesian Beta', 'Bayesian Normal', 
                   'Similarity Bayesian', 'Direct Posterior', 'Combined']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'magenta', 'cyan', 'gray']
    
    # 1. Frequency distribution of gap-to-win
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(nyc_df['strategy'], kde=True, bins=15)
    plt.title('NYC: Distribution of Gap-to-Win %', fontsize=14)
    plt.xlabel('Gap to Win %', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.histplot(alaska_df['strategy'], kde=True, bins=15, color='orange')
    plt.title('Alaska: Distribution of Gap-to-Win %', fontsize=14)
    plt.xlabel('Gap to Win %', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures_focused/gap_distribution.png', dpi=300)
    plt.close()
    
    # 2. Boxplot of exhaust-strategy difference by letter
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='letter', y='diff', data=nyc_df)
    plt.title('NYC: Exhaust - Strategy Difference by Letter', fontsize=14)
    plt.xlabel('Candidate Letter', fontsize=12)
    plt.ylabel('Difference (Exhaust % - Strategy %)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='letter', y='diff', data=alaska_df)
    plt.title('Alaska: Exhaust - Strategy Difference by Letter', fontsize=14) 
    plt.xlabel('Candidate Letter', fontsize=12)
    plt.ylabel('Difference (Exhaust % - Strategy %)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_focused/diff_boxplot.png', dpi=300)
    plt.close()
    
    # 3. Scatter plot of exhaust vs strategy
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='strategy', y='exhaust', hue='letter', data=nyc_df, alpha=0.7, s=80)
    plt.plot([0, 50], [0, 50], 'r--', alpha=0.5)  # Identity line
    plt.title('NYC: Exhaust % vs Strategy %', fontsize=14)
    plt.xlabel('Strategy %', fontsize=12)
    plt.ylabel('Exhaust %', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='strategy', y='exhaust', hue='letter', data=alaska_df, alpha=0.7, s=80)
    plt.plot([0, 50], [0, 50], 'r--', alpha=0.5)  # Identity line
    plt.title('Alaska: Exhaust % vs Strategy %', fontsize=14)
    plt.xlabel('Strategy %', fontsize=12)
    plt.ylabel('Exhaust %', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_focused/exhaust_vs_strategy.png', dpi=300)
    plt.close()
    
    if probability_df.empty:
        return
    
    # 4. Create probability distribution curves
    plt.figure(figsize=(15, 10))
    
    # Beta distribution
    plt.subplot(2, 2, 1)
    x = np.linspace(0, 1, 1000)
    for gap in [1, 5, 10, 20]:
        a, b = beta_parameters(gap)
        plt.plot(x, stats.beta.pdf(x, a, b), 
                label=f'Gap {gap}%, Beta({a:.1f}, {b:.1f})')
    plt.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    plt.title('Beta Distribution by Gap Size', fontsize=14)
    plt.xlabel('Proportion Preferring B over A', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Normal distribution
    plt.subplot(2, 2, 2)
    x = np.linspace(0, 100, 1000)
    for gap in [1, 5, 10, 20]:
        mean, std_dev = normal_parameters(gap)
        plt.plot(x, stats.norm.pdf(x, mean, std_dev), 
                label=f'Gap {gap}%, Normal({mean:.1f}, {std_dev:.1f})')
    plt.axvline(50, color='k', linestyle='--', alpha=0.5)
    plt.title('Normal Distribution by Gap Size', fontsize=14)
    plt.xlabel('Percentage Preferring B over A', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uniform distribution
    plt.subplot(2, 2, 3)
    x = np.linspace(0, 100, 1000)
    for gap in [1, 5, 10, 20]:
        # Calculate uniform normal for each gap
        std_dev = max(5, 10 - gap * 0.3)
        plt.plot(x, stats.norm.pdf(x, 50, std_dev), 
                label=f'Gap {gap}%, Normal(50, {std_dev:.1f})')
    plt.axvline(50, color='k', linestyle='--', alpha=0.5)
    plt.title('Neutral Distribution by Gap Size', fontsize=14)
    plt.xlabel('Percentage Preferring B over A', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Required preference vs probability
    plt.subplot(2, 2, 4)
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    if not nyc_prob.empty:
        plt.scatter(nyc_prob['required_preference_pct'], nyc_prob['combined_probability'], 
                  label='NYC', alpha=0.7, color='blue', s=60)
    
    if not alaska_prob.empty:
        plt.scatter(alaska_prob['required_preference_pct'], alaska_prob['combined_probability'], 
                   label='Alaska', alpha=0.7, color='red', s=60)
    
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    plt.axvline(50, color='k', linestyle='--', alpha=0.5)
    plt.title('Required Preference vs Probability', fontsize=14)
    plt.xlabel('Required Preference %', fontsize=12)
    plt.ylabel('Combined Probability', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_focused/probability_distributions.png', dpi=300)
    plt.close()
    
    # 6. All models vs. required preference - NYC
    if not nyc_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by required preference percentage for smoother lines
        nyc_sorted = nyc_prob.sort_values('required_preference_pct')
        
        for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            plt.plot(nyc_sorted['required_preference_pct'], nyc_sorted[model], 
                   'o-', color=color, label=label, markersize=8, linewidth=2, alpha=0.7)
            
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.axvline(50, color='k', linestyle='--', alpha=0.5)
        plt.title('NYC: Different Probability Models vs Required Preference %', fontsize=16)
        plt.xlabel('Required % of Exhausted Ballots Preferring Candidate B over A', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_probability_models_by_preference.png', dpi=300)
        plt.close()
    
    # 7. All models vs. required preference - Alaska
    if not alaska_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by required preference percentage for smoother lines
        alaska_sorted = alaska_prob.sort_values('required_preference_pct')
        
        for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            plt.plot(alaska_sorted['required_preference_pct'], alaska_sorted[model], 
                   'o-', color=color, label=label, markersize=8, linewidth=2, alpha=0.7)
            
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.axvline(50, color='k', linestyle='--', alpha=0.5)
        plt.title('Alaska: Different Probability Models vs Required Preference %', fontsize=16)
        plt.xlabel('Required % of Exhausted Ballots Preferring Candidate B over A', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_probability_models_by_preference.png', dpi=300)
        plt.close()
    
    # 8. Comparison of probability models for each region (without fitting)
    # NYC - Strategy/Exhaust ratio comparison
    if not nyc_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by ratio for smoother lines  
        nyc_sorted = nyc_prob.sort_values('strategy_exhaust_ratio')
        
        for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            plt.plot(nyc_sorted['strategy_exhaust_ratio'], nyc_sorted[model], 
                   'o-', color=color, label=label, markersize=8, linewidth=2, alpha=0.7)
            
        plt.title('NYC: Different Probability Models vs Strategy/Exhaust Ratio', fontsize=16)
        plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_probability_models_comparison.png', dpi=300)
        plt.close()
    
    # Alaska - Strategy/Exhaust ratio comparison  
    if not alaska_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by ratio for smoother lines
        alaska_sorted = alaska_prob.sort_values('strategy_exhaust_ratio')
        
        for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            plt.plot(alaska_sorted['strategy_exhaust_ratio'], alaska_sorted[model], 
                   'o-', color=color, label=label, markersize=8, linewidth=2, alpha=0.7)
            
        plt.title('Alaska: Different Probability Models vs Strategy/Exhaust Ratio', fontsize=16)
        plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_probability_models_comparison.png', dpi=300)
        plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    
    nyc_summary = nyc_prob[models + ['required_preference_pct', 'strategy_exhaust_ratio']].describe()
    print("\nNYC Summary:")
    print(nyc_summary)
    
    alaska_summary = alaska_prob[models + ['required_preference_pct', 'strategy_exhaust_ratio']].describe()
    print("\nAlaska Summary:")
    print(alaska_summary)
    
    # Save summary statistics
    nyc_summary.to_csv('figures_focused/nyc_probability_summary.csv')
    alaska_summary.to_csv('figures_focused/alaska_probability_summary.csv')

def create_probability_heatmaps(probability_df):
    """Create heatmaps of average combined probability binned by gap and exhaust percentages"""
    print("Creating probability heatmaps...")
    
    # Split data by region
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    # Define bins for gap to win and exhaust percentages
    gap_bins = [0, 1, 2.5, 5, 10, 100]
    exhaust_bins = [0, 5, 10, 15, 25, 100]
    
    # Create bin labels
    gap_labels = [f"{gap_bins[i]}-{gap_bins[i+1]}%" for i in range(len(gap_bins)-1)]
    exhaust_labels = [f"{exhaust_bins[i]}-{exhaust_bins[i+1]}%" for i in range(len(exhaust_bins)-1)]
    
    # Process NYC data
    if not nyc_prob.empty:
        # Create bins
        nyc_prob['gap_bin'] = pd.cut(nyc_prob['gap_to_win_pct'], bins=gap_bins, labels=gap_labels)
        nyc_prob['exhaust_bin'] = pd.cut(nyc_prob['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels)
        
        # Create pivot table for average combined probability
        nyc_pivot = nyc_prob.pivot_table(values='combined_probability', 
                                        index='gap_bin', 
                                        columns='exhaust_bin', 
                                        aggfunc='mean')
        
        # Create pivot table for counts
        nyc_counts = nyc_prob.pivot_table(values='combined_probability', 
                                         index='gap_bin', 
                                         columns='exhaust_bin', 
                                         aggfunc='count')
        
        # Fill NaN values with 0
        nyc_pivot = nyc_pivot.fillna(0)
        nyc_counts = nyc_counts.fillna(0).astype(int)
        
        # Create annotation with both probability and count
        nyc_annot = pd.DataFrame(index=nyc_pivot.index, columns=nyc_pivot.columns)
        for idx in nyc_pivot.index:
            for col in nyc_pivot.columns:
                if pd.notnull(nyc_pivot.loc[idx, col]):
                    prob = nyc_pivot.loc[idx, col]
                    count = nyc_counts.loc[idx, col]
                    nyc_annot.loc[idx, col] = f"{prob:.2%}\n(n={count})"
                else:
                    nyc_annot.loc[idx, col] = ""
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(nyc_pivot, annot=nyc_annot, fmt="", cmap='YlOrRd', 
                   vmin=0, vmax=0.5, cbar_kws={'label': 'Probability of Candidate B Winning'})
        
        # Adjust font size for annotations
        for t in ax.texts:
            t.set_fontsize(9)
            
        plt.title('NYC: Average Combined Probability by Gap and Exhaust Bins', fontsize=16)
        plt.xlabel('Exhausted Ballot Percentage Bin', fontsize=14)
        plt.ylabel('Gap Needed for Candidate B to Win (Percentage Bin)', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_probability_heatmap.png', dpi=300)
        plt.close()
    
    # Process Alaska data
    if not alaska_prob.empty:
        # Create bins
        alaska_prob['gap_bin'] = pd.cut(alaska_prob['gap_to_win_pct'], bins=gap_bins, labels=gap_labels)
        alaska_prob['exhaust_bin'] = pd.cut(alaska_prob['exhaust_pct'], bins=exhaust_bins, labels=exhaust_labels)
        
        # Create pivot table for average combined probability
        alaska_pivot = alaska_prob.pivot_table(values='combined_probability', 
                                             index='gap_bin', 
                                             columns='exhaust_bin', 
                                             aggfunc='mean')
        
        # Create pivot table for counts
        alaska_counts = alaska_prob.pivot_table(values='combined_probability', 
                                              index='gap_bin', 
                                              columns='exhaust_bin', 
                                              aggfunc='count')
        
        # Fill NaN values with 0
        alaska_pivot = alaska_pivot.fillna(0)
        alaska_counts = alaska_counts.fillna(0).astype(int)
        
        # Create annotation with both probability and count
        alaska_annot = pd.DataFrame(index=alaska_pivot.index, columns=alaska_pivot.columns)
        for idx in alaska_pivot.index:
            for col in alaska_pivot.columns:
                if pd.notnull(alaska_pivot.loc[idx, col]):
                    prob = alaska_pivot.loc[idx, col]
                    count = alaska_counts.loc[idx, col]
                    alaska_annot.loc[idx, col] = f"{prob:.2%}\n(n={count})"
                else:
                    alaska_annot.loc[idx, col] = ""
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(alaska_pivot, annot=alaska_annot, fmt="", cmap='YlOrRd', 
                   vmin=0, vmax=0.5, cbar_kws={'label': 'Probability of Candidate B Winning'})
        
        # Adjust font size for annotations
        for t in ax.texts:
            t.set_fontsize(9)
            
        plt.title('Alaska: Average Combined Probability by Gap and Exhaust Bins', fontsize=16)
        plt.xlabel('Exhausted Ballot Percentage Bin', fontsize=14)
        plt.ylabel('Gap Needed for Candidate B to Win (Percentage Bin)', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_probability_heatmap.png', dpi=300)
        plt.close()
    
    # Create side-by-side heatmaps for both regions
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # NYC heatmap
    if not nyc_prob.empty:
        ax = sns.heatmap(nyc_pivot, annot=nyc_annot, fmt="", cmap='YlOrRd', 
                   vmin=0, vmax=0.5, cbar_kws={'label': 'Probability of Candidate B Winning'}, ax=axes[0])
        
        # Adjust font size for annotations
        for t in ax.texts:
            t.set_fontsize(8)
            
        axes[0].set_title('NYC: Average Combined Probability by Gap and Exhaust Bins', fontsize=14)
        axes[0].set_xlabel('Exhausted Ballot Percentage Bin', fontsize=12)
        axes[0].set_ylabel('Gap Needed for Candidate B to Win (Percentage Bin)', fontsize=12)
    
    # Alaska heatmap
    if not alaska_prob.empty:
        ax = sns.heatmap(alaska_pivot, annot=alaska_annot, fmt="", cmap='YlOrRd', 
                   vmin=0, vmax=0.5, cbar_kws={'label': 'Probability of Candidate B Winning'}, ax=axes[1])
        
        # Adjust font size for annotations
        for t in ax.texts:
            t.set_fontsize(8)
            
        axes[1].set_title('Alaska: Average Combined Probability by Gap and Exhaust Bins', fontsize=14)
        axes[1].set_xlabel('Exhausted Ballot Percentage Bin', fontsize=12)
        axes[1].set_ylabel('Gap Needed for Candidate B to Win (Percentage Bin)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures_focused/combined_probability_heatmap.png', dpi=300)
    plt.close()

def create_scatter_heatmap(probability_df):
    """Create scatter plot heatmap where each point represents an individual election case"""
    print("Creating scatter plot heatmap...")
    
    # Split data by region
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    # Create a figure with two subplots for the combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define a colormap where darker colors represent higher probabilities
    cmap = plt.cm.YlOrRd  # Standard YlOrRd - darker red = higher probability
    
    # NYC scatter heatmap (in combined figure)
    if not nyc_prob.empty:
        scatter = axes[0].scatter(
            nyc_prob['exhaust_pct'], 
            nyc_prob['gap_to_win_pct'], 
            c=nyc_prob['combined_probability'],
            s=120,  # Larger size of points
            cmap=cmap,
            vmin=0, 
            vmax=0.5,
            alpha=0.9,
            edgecolors='black',  # Black boundaries
            linewidths=1        # Width of boundaries
        )
        
        # Add probability values as annotations for ALL points
        for idx, row in nyc_prob.iterrows():
            axes[0].annotate(
                f"{row['combined_probability']:.1%}",  # One decimal place
                (row['exhaust_pct'], row['gap_to_win_pct']),
                xytext=(4, 0),
                textcoords='offset points',
                fontsize=9,  # Slightly smaller font since all points will have labels
                fontweight='bold',
                color='black'
            )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axes[0])
        cbar.set_label('Probability of Candidate B Winning', fontsize=12)
        
        # Set titles and labels
        axes[0].set_title('NYC: Probability by Gap and Exhaust (Exact Points)', fontsize=14)
        axes[0].set_xlabel('Exhausted Ballot Percentage', fontsize=12)
        axes[0].set_ylabel('Gap Needed for Candidate B to Win (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_ylim(bottom=0)  # Start y-axis at 0
        
        # Set background color to very light gray to improve contrast
        axes[0].set_facecolor('#f8f8f8')
    
    # Alaska scatter heatmap (in combined figure)
    if not alaska_prob.empty:
        scatter = axes[1].scatter(
            alaska_prob['exhaust_pct'], 
            alaska_prob['gap_to_win_pct'], 
            c=alaska_prob['combined_probability'],
            s=120,  # Larger size of points
            cmap=cmap,
            vmin=0, 
            vmax=0.5,
            alpha=0.9,
            edgecolors='black',  # Black boundaries
            linewidths=1        # Width of boundaries
        )
        
        # Add probability values as annotations for ALL points
        for idx, row in alaska_prob.iterrows():
            axes[1].annotate(
                f"{row['combined_probability']:.1%}",  # One decimal place
                (row['exhaust_pct'], row['gap_to_win_pct']),
                xytext=(4, 0),
                textcoords='offset points',
                fontsize=9,  # Slightly smaller font since all points will have labels
                fontweight='bold',
                color='black'
            )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axes[1])
        cbar.set_label('Probability of Candidate B Winning', fontsize=12)
        
        # Set titles and labels
        axes[1].set_title('Alaska: Probability by Gap and Exhaust (Exact Points)', fontsize=14)
        axes[1].set_xlabel('Exhausted Ballot Percentage', fontsize=12)
        axes[1].set_ylabel('Gap Needed for Candidate B to Win (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_ylim(bottom=0)  # Start y-axis at 0
        
        # Set background color to very light gray to improve contrast
        axes[1].set_facecolor('#f8f8f8')
    
    # Save the combined figure
    plt.tight_layout()
    plt.savefig('figures_focused/scatter_heatmap.png', dpi=300)
    plt.close()
    
    # Create separate figures for NYC and Alaska
    
    # NYC separate figure
    if not nyc_prob.empty:
        fig_nyc = plt.figure(figsize=(10, 8))
        ax_nyc = fig_nyc.add_subplot(111)
        
        scatter_nyc = ax_nyc.scatter(
            nyc_prob['exhaust_pct'], 
            nyc_prob['gap_to_win_pct'], 
            c=nyc_prob['combined_probability'],
            s=130,  # Slightly larger for standalone plot
            cmap=cmap,
            vmin=0, 
            vmax=0.5,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )
        
        # Add probability values
        for idx, row in nyc_prob.iterrows():
            ax_nyc.annotate(
                f"{row['combined_probability']:.1%}",
                (row['exhaust_pct'], row['gap_to_win_pct']),
                xytext=(4, 0),
                textcoords='offset points',
                fontsize=10,  # Slightly larger font for standalone plot
                fontweight='bold',
                color='black'
            )
        
        # Add colorbar
        cbar_nyc = fig_nyc.colorbar(scatter_nyc)
        cbar_nyc.set_label('Probability of Candidate B Winning', fontsize=12)
        
        # Set titles and labels
        ax_nyc.set_title('NYC: Probability by Gap and Exhaust (Exact Points)', fontsize=16)
        ax_nyc.set_xlabel('Exhausted Ballot Percentage', fontsize=14)
        ax_nyc.set_ylabel('Gap Needed for Candidate B to Win (%)', fontsize=14)
        ax_nyc.grid(True, alpha=0.3, linestyle='--')
        ax_nyc.set_ylim(bottom=0)
        ax_nyc.set_facecolor('#f8f8f8')
        
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_scatter_heatmap.png', dpi=300)
        plt.close()
    
    # Alaska separate figure
    if not alaska_prob.empty:
        fig_alaska = plt.figure(figsize=(10, 8))
        ax_alaska = fig_alaska.add_subplot(111)
        
        scatter_alaska = ax_alaska.scatter(
            alaska_prob['exhaust_pct'], 
            alaska_prob['gap_to_win_pct'], 
            c=alaska_prob['combined_probability'],
            s=130,  # Slightly larger for standalone plot
            cmap=cmap,
            vmin=0, 
            vmax=0.5,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )
        
        # Add probability values
        for idx, row in alaska_prob.iterrows():
            ax_alaska.annotate(
                f"{row['combined_probability']:.1%}",
                (row['exhaust_pct'], row['gap_to_win_pct']),
                xytext=(4, 0),
                textcoords='offset points',
                fontsize=10,  # Slightly larger font for standalone plot
                fontweight='bold',
                color='black'
            )
        
        # Add colorbar
        cbar_alaska = fig_alaska.colorbar(scatter_alaska)
        cbar_alaska.set_label('Probability of Candidate B Winning', fontsize=12)
        
        # Set titles and labels
        ax_alaska.set_title('Alaska: Probability by Gap and Exhaust (Exact Points)', fontsize=16)
        ax_alaska.set_xlabel('Exhausted Ballot Percentage', fontsize=14)
        ax_alaska.set_ylabel('Gap Needed for Candidate B to Win (%)', fontsize=14)
        ax_alaska.grid(True, alpha=0.3, linestyle='--')
        ax_alaska.set_ylim(bottom=0)
        ax_alaska.set_facecolor('#f8f8f8')
        
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_scatter_heatmap.png', dpi=300)
        plt.close()

def create_bayesian_models_plot(probability_df):
    """Create plots showing only the Bayesian models for comparison"""
    print("Creating Bayesian models comparison plot...")
    
    # Split data by region
    nyc_prob = probability_df[probability_df['region'] == 'NYC']
    alaska_prob = probability_df[probability_df['region'] == 'Alaska']
    
    # Set up Bayesian models, labels, and colors
    bayesian_models = ['bayesian_beta_probability', 'bayesian_normal_probability', 
                     'direct_posterior_probability', 'direct_posterior_beta_probability', 
                     'direct_posterior_normal_probability']
    model_labels = ['Bayesian Beta', 'Bayesian Normal', 'Direct Posterior (Binary)',
                   'Direct Posterior Beta', 'Direct Posterior Normal']
    colors = ['purple', 'orange', 'cyan', 'blue', 'green']
    line_styles = ['-', '--', ':', '-', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    
    # NYC Bayesian models
    if not nyc_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by required preference percentage for smoother lines
        nyc_sorted = nyc_prob.sort_values('required_preference_pct')
        
        for i, (model, label, color, ls, marker) in enumerate(zip(bayesian_models, model_labels, colors, line_styles, markers)):
            plt.plot(nyc_sorted['required_preference_pct'], nyc_sorted[model], 
                   marker=marker, linestyle=ls, color=color, label=label, 
                   markersize=8, linewidth=2.5, alpha=0.8)
            
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.axvline(50, color='k', linestyle='--', alpha=0.5)
        plt.title('NYC: Bayesian Probability Models vs Required Preference %', fontsize=16)
        plt.xlabel('Required % of Exhausted Ballots Preferring Candidate B over A', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.05, 1.05)  # Set y-axis limits to show full probability range
        
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_bayesian_models_only.png', dpi=300)
        plt.close()
    
    # Alaska Bayesian models
    if not alaska_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by required preference percentage for smoother lines
        alaska_sorted = alaska_prob.sort_values('required_preference_pct')
        
        for i, (model, label, color, ls, marker) in enumerate(zip(bayesian_models, model_labels, colors, line_styles, markers)):
            plt.plot(alaska_sorted['required_preference_pct'], alaska_sorted[model], 
                   marker=marker, linestyle=ls, color=color, label=label, 
                   markersize=8, linewidth=2.5, alpha=0.8)
            
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.axvline(50, color='k', linestyle='--', alpha=0.5)
        plt.title('Alaska: Bayesian Probability Models vs Required Preference %', fontsize=16)
        plt.xlabel('Required % of Exhausted Ballots Preferring Candidate B over A', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.05, 1.05)  # Set y-axis limits to show full probability range
        
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_bayesian_models_only.png', dpi=300)
        plt.close()
    
    # Alternative plot comparing Bayesian models by strategy/exhaust ratio
    if not nyc_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by ratio for smoother lines  
        nyc_sorted = nyc_prob.sort_values('strategy_exhaust_ratio')
        
        for i, (model, label, color, ls, marker) in enumerate(zip(bayesian_models, model_labels, colors, line_styles, markers)):
            plt.plot(nyc_sorted['strategy_exhaust_ratio'], nyc_sorted[model], 
                   marker=marker, linestyle=ls, color=color, label=label, 
                   markersize=8, linewidth=2.5, alpha=0.8)
            
        plt.title('NYC: Bayesian Probability Models vs Strategy/Exhaust Ratio', fontsize=16)
        plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.05, 1.05)  # Set y-axis limits to show full probability range
        
        plt.tight_layout()
        plt.savefig('figures_focused/nyc_bayesian_models_ratio.png', dpi=300)
        plt.close()
    
    if not alaska_prob.empty:
        plt.figure(figsize=(14, 9))
        
        # Sort by ratio for smoother lines  
        alaska_sorted = alaska_prob.sort_values('strategy_exhaust_ratio')
        
        for i, (model, label, color, ls, marker) in enumerate(zip(bayesian_models, model_labels, colors, line_styles, markers)):
            plt.plot(alaska_sorted['strategy_exhaust_ratio'], alaska_sorted[model], 
                   marker=marker, linestyle=ls, color=color, label=label, 
                   markersize=8, linewidth=2.5, alpha=0.8)
            
        plt.title('Alaska: Bayesian Probability Models vs Strategy/Exhaust Ratio', fontsize=16)
        plt.xlabel('Strategy/Exhaust Ratio (Gap to Win % ÷ Exhaust %)', fontsize=14)
        plt.ylabel('Probability of Candidate B Winning', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.ylim(-0.05, 1.05)  # Set y-axis limits to show full probability range
        
        plt.tight_layout()
        plt.savefig('figures_focused/alaska_bayesian_models_ratio.png', dpi=300)
        plt.close()

def corrected_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=100):
    """
    Correctly implemented bootstrap that properly simulates the candidate trailing by the gap percentage.
    """
    print(f"Running corrected bootstrap with {n_bootstrap} iterations...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Extract complete ballots (those ranking both A and B)
    complete_ballots = {ballot: count for ballot, count in ballot_counts.items() 
                       if 'A' in ballot and 'B' in ballot}
    
    total_complete = sum(complete_ballots.values())
    total_exhausted = sum(exhausted_ballots.values())
    
    print(f"Total complete ballots: {total_complete}")
    print(f"Total exhausted ballots: {total_exhausted}")
    
    # Check original A and B first preferences
    original_a_votes = sum(count for ballot, count in ballot_counts.items() 
                         if ballot and ballot[0] == 'A')
    original_b_votes = sum(count for ballot, count in ballot_counts.items() 
                         if ballot and ballot[0] == 'B')
    
    print(f"Original first preferences - A: {original_a_votes}, B: {original_b_votes}")
    print(f"Original difference (B-A): {original_b_votes - original_a_votes}")
    
    # Calculate votes needed for B to win based on gap percentage
    # Need to properly simulate B trailing by the gap_to_win_pct
    total_ab_votes = original_a_votes + original_b_votes
    
    # Calculate what the vote counts should be to create the desired gap
    # where B is trailing A by the specified gap percentage
    ideal_gap_votes = int(gap_to_win_pct * total_ballots / 100)
    
    # Create adjusted scenario where B is trailing by gap_to_win_pct
    # Keep total votes the same, but redistribute between A and B
    adjusted_b_votes = (total_ab_votes - ideal_gap_votes) / 2
    adjusted_a_votes = total_ab_votes - adjusted_b_votes
    
    print(f"Adjusted vote distribution to create {gap_to_win_pct:.2f}% gap:")
    print(f"Adjusted A votes: {adjusted_a_votes}, B votes: {adjusted_b_votes}")
    print(f"Adjusted gap (A-B): {adjusted_a_votes - adjusted_b_votes} ({100*(adjusted_a_votes - adjusted_b_votes)/total_ballots:.2f}%)")
    
    # Group complete ballots by first preference for sampling
    grouped_complete = {}
    for ballot, count in complete_ballots.items():
        if ballot:  # Non-empty ballot
            first_pref = ballot[0]
            if first_pref not in grouped_complete:
                grouped_complete[first_pref] = {}
            grouped_complete[first_pref][ballot] = count
    
    # Print how many ballots have B over A vs A over B in complete ballots
    b_over_a = 0
    a_over_b = 0
    for ballot, count in complete_ballots.items():
        if 'A' in ballot and 'B' in ballot:
            if ballot.index('B') < ballot.index('A'):
                b_over_a += count
            else:
                a_over_b += count
    
    print(f"In complete ballots: B>A: {b_over_a}, A>B: {a_over_b}, B>A percentage: {100*b_over_a/(b_over_a+a_over_b):.2f}%")
    
    # Extract first preferences from exhausted ballots
    exh_first_prefs = {}
    for ballot, count in exhausted_ballots.items():
        if ballot:  # Non-empty ballot
            first_pref = ballot[0]
            if first_pref not in exh_first_prefs:
                exh_first_prefs[first_pref] = 0
            exh_first_prefs[first_pref] += count
    
    print("First preferences in exhausted ballots:")
    for pref, count in sorted(exh_first_prefs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_exhausted:.2f}%)")
    
    # Results container
    bootstrap_results = []
    b_win_counts = 0
    
    # Run bootstrap iterations
    for i in range(n_bootstrap):
        if i % 25 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
            
        # Start with adjusted votes where B is trailing by gap_to_win_pct
        a_votes = int(adjusted_a_votes)
        b_votes = int(adjusted_b_votes)
        
        # Track how many ballots were completed
        completed_count = 0
        b_over_a_count = 0
        a_over_b_count = 0
        
        # Process each exhausted ballot
        for exh_ballot, exh_count in exhausted_ballots.items():
            if not exh_ballot:  # Empty ballot
                continue
                
            first_pref = exh_ballot[0] if exh_ballot else None
            
            # If we have complete ballots with same first preference
            if first_pref in grouped_complete and grouped_complete[first_pref]:
                # Get options for sampling
                completion_options = list(grouped_complete[first_pref].keys())
                weights = [grouped_complete[first_pref][ballot] for ballot in completion_options]
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    
                    # Sample completions for each ballot in this group
                    sampled_completions = np.random.choice(
                        completion_options, 
                        size=exh_count, 
                        replace=True, 
                        p=weights
                    )
                    
                    # Track preferences in completed ballots
                    for completion in sampled_completions:
                        completed_count += 1
                        
                        # Find which candidate is preferred
                        if 'A' in completion and 'B' in completion:
                            if completion.index('B') < completion.index('A'):
                                b_over_a_count += 1
                                # Add to B's votes if B is first choice
                                if completion[0] == 'B':
                                    b_votes += 1
                                # Add to B's transfers if B is second choice after an eliminated candidate
                                elif completion[0] not in ['A', 'B']:
                                    if 'B' in completion and ('A' not in completion or completion.index('B') < completion.index('A')):
                                        b_votes += 1
                            else:
                                a_over_b_count += 1
                                # Add to A's votes if A is first choice
                                if completion[0] == 'A':
                                    a_votes += 1
                                # Add to A's transfers if A is second choice after an eliminated candidate
                                elif completion[0] not in ['A', 'B']:
                                    if 'A' in completion and ('B' not in completion or completion.index('A') < completion.index('B')):
                                        a_votes += 1
            
            # If no matching first preference, use all complete ballots
            else:
                # Use all complete ballots with a simple approach
                all_complete = list(complete_ballots.keys())
                all_weights = [complete_ballots[ballot] for ballot in all_complete]
                
                # Normalize weights
                total_weight = sum(all_weights)
                if total_weight > 0:
                    all_weights = [w/total_weight for w in all_weights]
                    
                    # Sample completions
                    sampled_completions = np.random.choice(
                        all_complete, 
                        size=exh_count, 
                        replace=True, 
                        p=all_weights
                    )
                    
                    # Track preferences in completed ballots
                    for completion in sampled_completions:
                        completed_count += 1
                        
                        # Find which candidate is preferred
                        if 'A' in completion and 'B' in completion:
                            if completion.index('B') < completion.index('A'):
                                b_over_a_count += 1
                                # Add to B's votes if B is first choice
                                if completion[0] == 'B':
                                    b_votes += 1
                                # Add to B's transfers if B is second choice after an eliminated candidate
                                elif completion[0] not in ['A', 'B']:
                                    if 'B' in completion and ('A' not in completion or completion.index('B') < completion.index('A')):
                                        b_votes += 1
                            else:
                                a_over_b_count += 1
                                # Add to A's votes if A is first choice
                                if completion[0] == 'A':
                                    a_votes += 1
                                # Add to A's transfers if A is second choice after an eliminated candidate
                                elif completion[0] not in ['A', 'B']:
                                    if 'A' in completion and ('B' not in completion or completion.index('A') < completion.index('B')):
                                        a_votes += 1
        
        # Check if B wins after accounting for exhausted ballots
        b_wins = b_votes > a_votes
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            completed_pct = 100 * b_over_a_count / completed_count if completed_count > 0 else 0
            print(f"\nIteration {i} details:")
            print(f"  Completed {completed_count} ballots")
            print(f"  B>A: {b_over_a_count}, A>B: {a_over_b_count}, B>A%: {completed_pct:.2f}%")
            print(f"  Final votes - A: {a_votes}, B: {b_votes}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print("\nCorrected Bootstrap Results:")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results

def category_based_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=1000):
    """
    Category-based bootstrap simulation following the user's specifications:
    1. Find total votes and using gap, find number of additional votes for B to win
    2. Find exhausted ballots and categorize by first choice
    3. For each category, find A>B and B>A preferences in non-exhausted ballots
    4. Use sampling to fill in preferences for exhausted ballots
    5. Check if completed preferences give B enough votes to win
    6. Repeat 1000 times and report results
    """
    print(f"Running category-based bootstrap with {n_bootstrap} iterations...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    print(f"Gap in votes (A leads B by): {gap_votes}")
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    print(f"B needs {votes_needed_for_b} additional votes to win")
    
    # Get total exhausted ballots
    total_exhausted = sum(exhausted_ballots.values())
    print(f"Total exhausted ballots: {total_exhausted}")
    
    # Categorize exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Print first preference distribution in exhausted ballots
    print("\nExhausted ballots by first preference:")
    for pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_exhausted:.2f}%)")
    
    # Identify non-exhausted ballots that have both A and B
    complete_by_first_pref = {}
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
        if 'A' in ballot and 'B' in ballot:
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'ballots': {}, 'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            
            complete_by_first_pref[first_pref]['ballots'][ballot] = count
            complete_by_first_pref[first_pref]['total'] += count
            
            # Record A>B vs B>A preference
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count
    
    # Calculate A>B and B>A percentages for each first preference category
    print("\nA>B vs B>A preferences by first preference category:")
    for pref, data in sorted(complete_by_first_pref.items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] > 0:
            b_over_a_pct = 100 * data['b_over_a'] / data['total']
            print(f"  {pref}: Total={data['total']}, B>A={data['b_over_a']} ({b_over_a_pct:.2f}%), " +
                  f"A>B={data['a_over_b']} ({100-b_over_a_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    b_win_counts = 0
    
    # Results for each iteration
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Track net votes gained by B (B>A minus A>B)
        net_votes_for_b = 0
        
        # Track completions
        b_over_a_completions = 0
        a_over_b_completions = 0
        
        # Process each category of exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            # Check if we have data on preferences for this first preference
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Get preference distribution for this category
                category_data = complete_by_first_pref[first_pref]
                prob_b_over_a = category_data['b_over_a'] / category_data['total']
                
                # Sample B>A vs A>B preferences for this category
                b_completions = np.random.binomial(count, prob_b_over_a)
                a_completions = count - b_completions
                
                # Add to totals
                b_over_a_completions += b_completions
                a_over_b_completions += a_completions
                
                # Update net votes for B
                net_votes_for_b += (b_completions - a_completions)
            else:
                # If no data for this category, use the overall distribution
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob_b_over_a = total_b_over_a / total_complete
                    
                    # Sample using overall distribution
                    b_completions = np.random.binomial(count, overall_prob_b_over_a)
                    a_completions = count - b_completions
                    
                    # Add to totals
                    b_over_a_completions += b_completions
                    a_over_b_completions += a_completions
                    
                    # Update net votes for B
                    net_votes_for_b += (b_completions - a_completions)
        
        # Check if B wins: net_votes_for_b must exceed the gap
        b_wins = net_votes_for_b >= votes_needed_for_b
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            total_completions = b_over_a_completions + a_over_b_completions
            b_over_a_pct = 100 * b_over_a_completions / total_completions if total_completions > 0 else 0
            
            print(f"\nIteration {i} details:")
            print(f"  Completed {total_completions} ballots")
            print(f"  B>A: {b_over_a_completions}, A>B: {a_over_b_completions}, B>A%: {b_over_a_pct:.2f}%")
            print(f"  Net votes gained by B: {net_votes_for_b}")
            print(f"  Votes needed for B to win: {votes_needed_for_b}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print("\nCategory-Based Bootstrap Results:")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results

def limited_ranking_bootstrap(ballot_counts, candidates, exhausted_ballots, gap_to_win_pct, exhaust_pct, required_preference_pct=None, n_bootstrap=1000, max_rankings=6):
    """
    Bootstrap simulation that respects NYC's 6-candidate ranking limit.
    
    This function only considers partially filled exhausted ballots (those with fewer than 6 rankings)
    and excludes ballots that already have the maximum number of rankings.
    
    Parameters:
    - ballot_counts: Dictionary of ballots and their counts
    - candidates: List of candidates
    - exhausted_ballots: Dictionary of exhausted ballots and their counts
    - gap_to_win_pct: Gap percentage needed for B to win
    - exhaust_pct: Percentage of exhausted ballots
    - required_preference_pct: Required preference percentage for B to win
    - n_bootstrap: Number of bootstrap iterations
    - max_rankings: Maximum number of candidates that can be ranked (NYC limit is 6)
    """
    print(f"Running limited ranking bootstrap with {n_bootstrap} iterations (max rankings: {max_rankings})...")
    
    # Get total ballots
    total_ballots = sum(ballot_counts.values())
    print(f"Total ballots in election: {total_ballots}")
    
    # Calculate the gap in votes directly from the gap percentage
    gap_votes = int(gap_to_win_pct * total_ballots / 100)
    print(f"Gap in votes (A leads B by): {gap_votes}")
    
    # Votes needed for B to win: gap + 1
    votes_needed_for_b = gap_votes + 1
    print(f"B needs {votes_needed_for_b} additional votes to win")
    
    # Filter exhausted ballots to only include those with fewer than max_rankings
    partial_exhausted_ballots = {ballot: count for ballot, count in exhausted_ballots.items() 
                               if ballot and len(ballot) < max_rankings}
    
    # Count all exhausted ballots and partial ones
    total_exhausted = sum(exhausted_ballots.values())
    total_partial_exhausted = sum(partial_exhausted_ballots.values())
    
    print(f"Total exhausted ballots: {total_exhausted}")
    print(f"Total partially filled exhausted ballots (< {max_rankings} rankings): {total_partial_exhausted} ({100*total_partial_exhausted/total_exhausted:.2f}%)")
    
    # Categorize partially filled exhausted ballots by first preference
    exh_by_first_pref = {}
    for ballot, count in partial_exhausted_ballots.items():
        if not ballot:  # Skip empty ballots
            continue
        first_pref = ballot[0]
        if first_pref not in exh_by_first_pref:
            exh_by_first_pref[first_pref] = 0
        exh_by_first_pref[first_pref] += count
    
    # Print first preference distribution in partially filled exhausted ballots
    print(f"\nPartially filled exhausted ballots (< {max_rankings} rankings) by first preference:")
    for pref, count in sorted(exh_by_first_pref.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pref}: {count} ({100*count/total_partial_exhausted:.2f}%)")
    
    # Identify non-exhausted ballots that have both A and B
    complete_by_first_pref = {}
    for ballot, count in ballot_counts.items():
        if not ballot:
            continue
        if 'A' in ballot and 'B' in ballot:
            first_pref = ballot[0]
            if first_pref not in complete_by_first_pref:
                complete_by_first_pref[first_pref] = {'ballots': {}, 'b_over_a': 0, 'a_over_b': 0, 'total': 0}
            
            complete_by_first_pref[first_pref]['ballots'][ballot] = count
            complete_by_first_pref[first_pref]['total'] += count
            
            # Record A>B vs B>A preference
            if ballot.index('B') < ballot.index('A'):
                complete_by_first_pref[first_pref]['b_over_a'] += count
            else:
                complete_by_first_pref[first_pref]['a_over_b'] += count
    
    # Calculate A>B and B>A percentages for each first preference category
    print("\nA>B vs B>A preferences by first preference category:")
    for pref, data in sorted(complete_by_first_pref.items(), key=lambda x: x[1]['total'], reverse=True):
        if data['total'] > 0:
            b_over_a_pct = 100 * data['b_over_a'] / data['total']
            print(f"  {pref}: Total={data['total']}, B>A={data['b_over_a']} ({b_over_a_pct:.2f}%), " +
                  f"A>B={data['a_over_b']} ({100-b_over_a_pct:.2f}%)")
    
    # Run bootstrap iterations
    bootstrap_results = []
    b_win_counts = 0
    
    # Results for each iteration
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
        
        # Track net votes gained by B (B>A minus A>B)
        net_votes_for_b = 0
        
        # Track completions
        b_over_a_completions = 0
        a_over_b_completions = 0
        
        # Process each category of partially filled exhausted ballots
        for first_pref, count in exh_by_first_pref.items():
            # Check if we have data on preferences for this first preference
            if first_pref in complete_by_first_pref and complete_by_first_pref[first_pref]['total'] > 0:
                # Get preference distribution for this category
                category_data = complete_by_first_pref[first_pref]
                prob_b_over_a = category_data['b_over_a'] / category_data['total']
                
                # Sample B>A vs A>B preferences for this category
                b_completions = np.random.binomial(count, prob_b_over_a)
                a_completions = count - b_completions
                
                # Add to totals
                b_over_a_completions += b_completions
                a_over_b_completions += a_completions
                
                # Update net votes for B
                net_votes_for_b += (b_completions - a_completions)
            else:
                # If no data for this category, use the overall distribution
                total_complete = sum(data['total'] for data in complete_by_first_pref.values())
                total_b_over_a = sum(data['b_over_a'] for data in complete_by_first_pref.values())
                
                if total_complete > 0:
                    overall_prob_b_over_a = total_b_over_a / total_complete
                    
                    # Sample using overall distribution
                    b_completions = np.random.binomial(count, overall_prob_b_over_a)
                    a_completions = count - b_completions
                    
                    # Add to totals
                    b_over_a_completions += b_completions
                    a_over_b_completions += a_completions
                    
                    # Update net votes for B
                    net_votes_for_b += (b_completions - a_completions)
        
        # Check if B wins: net_votes_for_b must exceed the gap
        b_wins = net_votes_for_b >= votes_needed_for_b
        
        # Record result
        bootstrap_results.append(b_wins)
        if b_wins:
            b_win_counts += 1
        
        # Record detailed stats for first few iterations
        if i < 5:
            total_completions = b_over_a_completions + a_over_b_completions
            b_over_a_pct = 100 * b_over_a_completions / total_completions if total_completions > 0 else 0
            
            print(f"\nIteration {i} details:")
            print(f"  Completed {total_completions} partially filled ballots")
            print(f"  B>A: {b_over_a_completions}, A>B: {a_over_b_completions}, B>A%: {b_over_a_pct:.2f}%")
            print(f"  Net votes gained by B: {net_votes_for_b}")
            print(f"  Votes needed for B to win: {votes_needed_for_b}")
            print(f"  B wins: {b_wins}")
    
    # Calculate probability and confidence interval
    b_win_probability = b_win_counts / n_bootstrap
    
    # 95% confidence interval
    alpha = 0.05
    se = np.sqrt((b_win_probability * (1 - b_win_probability)) / n_bootstrap)
    ci_lower = max(0, b_win_probability - 1.96 * se)
    ci_upper = min(1, b_win_probability + 1.96 * se)
    
    # Report results
    print(f"\nLimited Ranking Bootstrap Results (max {max_rankings} rankings):")
    print(f"Probability B wins: {b_win_probability:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"B won in {b_win_counts} out of {n_bootstrap} iterations")
    if required_preference_pct is not None:
        print(f"Required preference: {required_preference_pct:.2f}%")
    
    return b_win_probability, (ci_lower, ci_upper), bootstrap_results

def run_corrected_bootstrap(nyc_df=None, alaska_df=None):
    """Run corrected bootstrap analysis on a competitive election from NYC"""
    # If dataframes not provided, load them
    if nyc_df is None or alaska_df is None:
        try:
            nyc_df = pd.read_csv('nyc_focused_analysis.csv')
            alaska_df = pd.read_csv('alaska_focused_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for df in [nyc_df, alaska_df]:
                for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        except Exception as e:
            print(f"Error loading data files: {e}")
            return
    
    # Find competitive NYC elections (close gap_to_win_pct)
    if not nyc_df.empty:
        # Sort by gap_to_win_pct to find closest elections
        nyc_competitive = nyc_df.sort_values('strategy')
        
        # Get the most competitive election
        if len(nyc_competitive) > 0:
            election = nyc_competitive.iloc[0]
            
            print(f"\nCorrected analysis of competitive NYC election: {election['election_id']}")
            print(f"Candidate: {election['letter']}")
            print(f"Gap to win: {election['strategy']:.2f}%")
            print(f"Exhaust percent: {election['exhaust']:.2f}%")
            
            # Calculate required preference percentage
            required_net_advantage = (election['strategy'] / election['exhaust']) * 100
            required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
            
            print(f"Required preference percentage: {required_preference_pct:.2f}%")
            
            # Get ballot data
            ballot_counts = election['ballot_counts']
            candidates = election['candidates']
            exhausted_ballots = election['exhausted_ballots']
            
            # Run standard corrected bootstrap analysis
            bootstrap_prob, bootstrap_ci, bootstrap_results = corrected_bootstrap(
                ballot_counts, candidates, exhausted_ballots, 
                gap_to_win_pct=election['strategy'],
                exhaust_pct=election['exhaust'],
                required_preference_pct=required_preference_pct,
                n_bootstrap=1000  # Use 1000 iterations for better statistics
            )
            
            # Run category-based bootstrap analysis
            cat_bootstrap_prob, cat_bootstrap_ci, cat_bootstrap_results = category_based_bootstrap(
                ballot_counts, candidates, exhausted_ballots, 
                gap_to_win_pct=election['strategy'],
                exhaust_pct=election['exhaust'],
                required_preference_pct=required_preference_pct,
                n_bootstrap=1000  # Use 1000 iterations for better statistics
            )
            
            # Run limited ranking bootstrap analysis (NYC 6-candidate limit)
            limited_bootstrap_prob, limited_bootstrap_ci, limited_bootstrap_results = limited_ranking_bootstrap(
                ballot_counts, candidates, exhausted_ballots, 
                gap_to_win_pct=election['strategy'],
                exhaust_pct=election['exhaust'],
                required_preference_pct=required_preference_pct,
                n_bootstrap=1000,  # Use 1000 iterations for better statistics
                max_rankings=6     # NYC maximum allowed rankings
            )
            
            # For comparison, get probabilities from other models
            beta_prob = beta_probability(required_preference_pct, election['strategy'])
            normal_prob = normal_probability(required_preference_pct, election['strategy'])
            uniform_prob = uniform_probability(required_preference_pct, election['strategy'])
            bayesian_beta_prob = bayesian_beta_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            bayesian_normal_prob = bayesian_normal_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            similarity_prob = similarity_bayesian_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_beta_prob = direct_posterior_beta(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            direct_normal_prob = direct_posterior_normal(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Print comparison
            print("\nModel Probability Comparison:")
            print(f"Standard Bootstrap:  {bootstrap_prob:.4f} ({bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f})")
            print(f"Category Bootstrap:  {cat_bootstrap_prob:.4f} ({cat_bootstrap_ci[0]:.4f}, {cat_bootstrap_ci[1]:.4f})")
            print(f"Limited Ranking:     {limited_bootstrap_prob:.4f} ({limited_bootstrap_ci[0]:.4f}, {limited_bootstrap_ci[1]:.4f})")
            print(f"Beta:                {beta_prob:.4f}")
            print(f"Normal:              {normal_prob:.4f}")
            print(f"Uniform:             {uniform_prob:.4f}")
            print(f"Bayesian Beta:       {bayesian_beta_prob:.4f}")
            print(f"Bayesian Normal:     {bayesian_normal_prob:.4f}")
            print(f"Similarity:          {similarity_prob:.4f}")
            print(f"Direct Beta:         {direct_beta_prob:.4f}")
            print(f"Direct Normal:       {direct_normal_prob:.4f}")
            
            # Create visualization comparing models
            plt.figure(figsize=(14, 7))
            models = ['Standard\nBootstrap', 'Category\nBootstrap', 'Limited\nRanking', 'Beta', 'Normal', 'Uniform', 
                     'Bayesian\nBeta', 'Bayesian\nNormal', 'Similarity', 'Direct\nBeta', 'Direct\nNormal']
            probs = [bootstrap_prob, cat_bootstrap_prob, limited_bootstrap_prob, beta_prob, normal_prob, uniform_prob, 
                    bayesian_beta_prob, bayesian_normal_prob, similarity_prob, direct_beta_prob, direct_normal_prob]
            
            # Bar chart of probabilities
            bars = plt.bar(models, probs, color='skyblue', alpha=0.7)
            bars[0].set_color('darkred')    # Highlight standard bootstrap
            bars[1].set_color('darkgreen')  # Highlight category bootstrap
            bars[2].set_color('purple')     # Highlight limited ranking bootstrap
            
            # Add confidence intervals for bootstrap methods
            plt.errorbar(models[0], bootstrap_prob, 
                       yerr=[[bootstrap_prob-bootstrap_ci[0]], [bootstrap_ci[1]-bootstrap_prob]],
                       fmt='o', color='black', capsize=10)
            
            plt.errorbar(models[1], cat_bootstrap_prob, 
                       yerr=[[cat_bootstrap_prob-cat_bootstrap_ci[0]], [cat_bootstrap_ci[1]-cat_bootstrap_prob]],
                       fmt='o', color='black', capsize=10)
                       
            plt.errorbar(models[2], limited_bootstrap_prob, 
                       yerr=[[limited_bootstrap_prob-limited_bootstrap_ci[0]], [limited_bootstrap_ci[1]-limited_bootstrap_prob]],
                       fmt='o', color='black', capsize=10)
            
            plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
            plt.title(f'Probability Models for NYC Election {election["election_id"]}', fontsize=14)
            plt.ylabel('Probability of Candidate B Winning', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('bootstrap_comparison.png', dpi=300)
            plt.close()
            
            print("\nVisualization saved as 'bootstrap_comparison.png'")
            
            return election, bootstrap_prob, bootstrap_ci, cat_bootstrap_prob, cat_bootstrap_ci, limited_bootstrap_prob, limited_bootstrap_ci, probs
        else:
            print("No competitive NYC elections found.")
    else:
        print("NYC data is empty.")

def analyze_nyc_elections_with_limited_ranking(nyc_df=None):
    """
    Run the limited_ranking_bootstrap on all NYC elections.
    NYC has a rule that only 6 candidates can be ranked.
    
    This function analyzes all NYC elections to see how the results change
    when only partially filled exhausted ballots are considered.
    """
    print("\n=== ANALYZING NYC ELECTIONS WITH LIMITED RANKING BOOTSTRAP ===\n")
    
    # If dataframe not provided, load it
    if nyc_df is None:
        try:
            nyc_df = pd.read_csv('nyc_focused_analysis.csv')
            
            # Convert string representations of dictionaries back to actual dictionaries
            for col in ['ballot_counts', 'candidates', 'exhausted_ballots']:
                if col in nyc_df.columns:
                    nyc_df[col] = nyc_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    
            print(f"Loaded {len(nyc_df)} NYC elections from CSV file")
        except Exception as e:
            print(f"Error loading data file: {e}")
            return
    
    if nyc_df.empty:
        print("No NYC election data available")
        return
    
    # Sort elections by gap_to_win_pct (strategy)
    nyc_df_sorted = nyc_df.sort_values('strategy')
    
    # Store results for visualization
    results_data = []
    
    # Create output directory for results
    os.makedirs('nyc_limited_ranking_results', exist_ok=True)
    
    # Analyze each NYC election
    for idx, election in nyc_df_sorted.iterrows():
        election_id = election['election_id']
        letter = election['letter']
        gap_to_win_pct = election['strategy']
        exhaust_pct = election['exhaust']
        
        # Skip if no exhausted ballots
        if exhaust_pct <= 0.01:
            print(f"Skipping {election_id}, letter {letter}: No exhausted ballots")
            continue
        
        # Calculate required preference percentage
        required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
        required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
        
        print(f"\nAnalyzing NYC election: {election_id}")
        print(f"Candidate: {letter}")
        print(f"Gap to win: {gap_to_win_pct:.2f}%")
        print(f"Exhaust percent: {exhaust_pct:.2f}%")
        print(f"Required preference percentage: {required_preference_pct:.2f}%")
        
        # Get ballot data
        ballot_counts = election['ballot_counts']
        candidates = election['candidates']
        exhausted_ballots = election['exhausted_ballots']
        
        total_ballots = sum(ballot_counts.values())
        
        # Only run bootstrap if there are exhausted ballots
        total_exhausted = sum(exhausted_ballots.values())
        if total_exhausted == 0:
            print(f"Skipping {election_id}: No exhausted ballots")
            continue
        
        # Filter for partially filled exhausted ballots
        partial_exhausted_ballots = {ballot: count for ballot, count in exhausted_ballots.items() 
                                  if ballot and len(ballot) < 6}
        total_partial_exhausted = sum(partial_exhausted_ballots.values())
        
        print(f"Total exhausted ballots: {total_exhausted}")
        print(f"Partially filled exhausted ballots (< 6 rankings): {total_partial_exhausted} ({100*total_partial_exhausted/total_exhausted:.2f}%)")
        
        # Only run bootstrap if there are partially filled exhausted ballots
        if total_partial_exhausted == 0:
            print(f"Skipping {election_id}: No partially filled exhausted ballots")
            continue
        
        # Run limited ranking bootstrap with 100 iterations for speed
        try:
            limited_bootstrap_prob, limited_bootstrap_ci, _ = limited_ranking_bootstrap(
                ballot_counts, candidates, exhausted_ballots, 
                gap_to_win_pct=gap_to_win_pct,
                exhaust_pct=exhaust_pct,
                required_preference_pct=required_preference_pct,
                n_bootstrap=100,  # Use fewer iterations for speed
                max_rankings=6    # NYC maximum allowed rankings
            )
            
            # Calculate other model probabilities for comparison
            beta_prob = beta_probability(required_preference_pct, gap_to_win_pct)
            normal_prob = normal_probability(required_preference_pct, gap_to_win_pct)
            bayesian_beta_prob = bayesian_beta_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            bayesian_normal_prob = bayesian_normal_probability(required_preference_pct, ballot_counts, candidates, exhausted_ballots)
            
            # Store results
            results_data.append({
                'election_id': election_id,
                'letter': letter,
                'gap_to_win_pct': gap_to_win_pct,
                'exhaust_pct': exhaust_pct,
                'required_preference_pct': required_preference_pct,
                'limited_bootstrap_prob': limited_bootstrap_prob,
                'limited_bootstrap_ci_lower': limited_bootstrap_ci[0],
                'limited_bootstrap_ci_upper': limited_bootstrap_ci[1],
                'beta_prob': beta_prob,
                'normal_prob': normal_prob,
                'bayesian_beta_prob': bayesian_beta_prob,
                'bayesian_normal_prob': bayesian_normal_prob,
                'total_ballots': total_ballots,
                'total_exhausted': total_exhausted,
                'total_partial_exhausted': total_partial_exhausted,
                'percent_partial': 100 * total_partial_exhausted / total_exhausted
            })
            
            # Create individual election visualization
            plt.figure(figsize=(10, 6))
            models = ['Limited\nRanking', 'Beta', 'Normal', 'Bayesian\nBeta', 'Bayesian\nNormal']
            probs = [limited_bootstrap_prob, beta_prob, normal_prob, bayesian_beta_prob, bayesian_normal_prob]
            
            # Bar chart of probabilities
            bars = plt.bar(models, probs, color='skyblue', alpha=0.7)
            bars[0].set_color('purple')  # Highlight limited ranking bootstrap
            
            # Add confidence interval for limited ranking bootstrap
            plt.errorbar(models[0], limited_bootstrap_prob, 
                       yerr=[[limited_bootstrap_prob-limited_bootstrap_ci[0]], [limited_bootstrap_ci[1]-limited_bootstrap_prob]],
                       fmt='o', color='black', capsize=10)
            
            plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
            plt.title(f'NYC Election {election_id}, Candidate {letter}', fontsize=14)
            plt.ylabel('Probability of Candidate B Winning', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Create a clean filename
            clean_id = election_id.replace("/", "_").replace("\\", "_")
            plt.savefig(f'nyc_limited_ranking_results/election_{clean_id}_candidate_{letter}.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing {election_id}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_data)
    
    if not results_df.empty:
        # Save results to CSV
        results_df.to_csv('nyc_limited_ranking_results.csv', index=False)
        print(f"\nSaved results for {len(results_df)} NYC elections to 'nyc_limited_ranking_results.csv'")
        
        # Create summary visualizations
        plt.figure(figsize=(12, 8))
        plt.scatter(results_df['gap_to_win_pct'], results_df['limited_bootstrap_prob'], 
                  s=80, alpha=0.7, c=results_df['exhaust_pct'], cmap='viridis')
        plt.colorbar(label='Exhaust Percentage')
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.title('Limited Ranking Bootstrap Probability vs Gap to Win', fontsize=14)
        plt.xlabel('Gap to Win (%)', fontsize=12)
        plt.ylabel('Probability of B Winning', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('nyc_limited_ranking_results/gap_vs_probability.png', dpi=300)
        plt.close()
        
        # Create comparison with other models
        plt.figure(figsize=(14, 8))
        
        # Sort by limited_bootstrap_prob for better visualization
        results_sorted = results_df.sort_values('limited_bootstrap_prob')
        
        plt.plot(range(len(results_sorted)), results_sorted['limited_bootstrap_prob'], 'o-', 
               label='Limited Ranking Bootstrap', markersize=8, linewidth=2, color='purple')
        plt.plot(range(len(results_sorted)), results_sorted['bayesian_beta_prob'], 'o-', 
               label='Bayesian Beta', markersize=6, linewidth=1.5, alpha=0.7)
        plt.plot(range(len(results_sorted)), results_sorted['bayesian_normal_prob'], 'o-', 
               label='Bayesian Normal', markersize=6, linewidth=1.5, alpha=0.7)
        
        plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        plt.title('Limited Ranking Bootstrap vs Other Models for NYC Elections', fontsize=14)
        plt.xlabel('Election Index (sorted by Limited Ranking probability)', fontsize=12)
        plt.ylabel('Probability of B Winning', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('nyc_limited_ranking_results/model_comparison.png', dpi=300)
        plt.close()
        
        # Print summary statistics
        print("\nSummary Statistics for Limited Ranking Bootstrap:")
        summary = results_df[['gap_to_win_pct', 'exhaust_pct', 'required_preference_pct', 
                             'limited_bootstrap_prob', 'percent_partial']].describe()
        print(summary)
        
        # Count elections where bootstrap probability > 0.5
        win_count = len(results_df[results_df['limited_bootstrap_prob'] > 0.5])
        print(f"\nElections where B is likely to win (prob > 0.5): {win_count} out of {len(results_df)} ({100*win_count/len(results_df):.1f}%)")
        
        return results_df
    else:
        print("No results to analyze")
        return None

def main():
    # Process election data (filtered for exhaust > strategy only)
    nyc_df, alaska_df = process_election_data()
    
    # Calculate probabilities
    probability_df = calculate_probabilities(nyc_df, alaska_df)
    
    # Create visualizations
    create_visualizations(nyc_df, alaska_df, probability_df)
    
    # Create probability heatmaps
    create_probability_heatmaps(probability_df)
    
    # Create scatter heatmap (individual points without binning)
    create_scatter_heatmap(probability_df)
    
    # Create plot with only Bayesian models
    create_bayesian_models_plot(probability_df)
    
    # Run corrected bootstrap analysis on a competitive NYC election
    run_corrected_bootstrap(nyc_df, alaska_df)
    
    print("\nAnalysis complete. Visualizations saved to 'figures_focused' directory.")

if __name__ == "__main__":
    main() 