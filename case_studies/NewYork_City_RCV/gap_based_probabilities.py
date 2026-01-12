import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_gap_based_probabilities():
    """
    Calculate the probability that a candidate wins if all exhausted ballots are completed,
    using distributions that reflect the strategy gap (how much the candidate needs to close to win).
    
    Key concepts:
    - "strategy_pct" represents the gap that candidate B needs to close to win
    - "exhaust_pct" represents the percentage of ballots that were exhausted
    - We calculate the net advantage B needs among exhausted ballots to close the gap
    - We then estimate the probability of this happening using various distribution models
      that are calibrated directly to the strategy gap
    """
    # Load the data
    df = pd.read_csv('exhaust_vs_strategy_analysis.csv')
    
    # Filter for cases where exhaust > strategy and exclude Letter A (the winner)
    vulnerable = df[(df['exhaust_greater']) & (df['letter'] != 'A')]
    
    # Calculate empirical parameters from the data
    all_strategy_values = df['strategy'].values
    all_exhaust_values = df['exhaust'].values
    
    # Calculate Beta parameters from the data
    beta_params = calculate_beta_parameters_from_data(all_strategy_values)
    
    # Calculate Normal parameters from the data
    normal_params = calculate_normal_params_from_data(all_strategy_values)
    
    # Group data by gap size ranges for more specific models
    gap_ranges = [(0, 1), (1, 3), (3, 10), (10, float('inf'))]
    gap_models = {}
    
    for min_gap, max_gap in gap_ranges:
        gap_data = df[(df['strategy'] >= min_gap) & (df['strategy'] < max_gap)]
        if len(gap_data) >= 5:  # Only create model if we have enough data
            gap_strategies = gap_data['strategy'].values
            gap_models[f"{min_gap}-{max_gap}"] = {
                'beta': calculate_beta_parameters_from_data(gap_strategies),
                'normal': calculate_normal_params_from_data(gap_strategies),
                'count': len(gap_data)
            }
    
    # Print the empirically derived parameters
    print("\nEmpirical Parameters Derived from Data:")
    print(f"Overall Beta parameters: Beta({beta_params[0]:.2f}, {beta_params[1]:.2f})")
    print(f"Overall Normal parameters: mean={normal_params[0]:.2f}, std={normal_params[1]:.2f}")
    
    print("\nGap-Specific Parameters:")
    for gap_range, params in gap_models.items():
        print(f"Gap range {gap_range}% (n={params['count']}):")
        print(f"  Beta({params['beta'][0]:.2f}, {params['beta'][1]:.2f})")
        print(f"  Normal(μ={params['normal'][0]:.2f}, σ={params['normal'][1]:.2f})")
    
    # Calculate key metrics for each candidate
    results = []
    
    for idx, row in vulnerable.iterrows():
        region = row['region']
        election_id = row['election_id']
        letter = row['letter']
        gap_to_win_pct = row['strategy']  # Gap B needs to close to win
        exhaust_pct = row['exhaust']  # Available exhausted ballots
        
        # Step 1: Calculate the required net advantage among exhausted ballots
        required_net_advantage = (gap_to_win_pct / exhaust_pct) * 100
        
        # Step 2: Calculate the required percentage of exhausted ballots ranking B higher
        required_preference_pct = (1 + required_net_advantage/100) / 2 * 100
        
        # Step 3: Calculate probabilities using empirical distribution models
        
        # Find the appropriate gap model for this election
        selected_gap_model = None
        for (min_gap, max_gap), gap_range in zip(gap_ranges, gap_models.keys()):
            if min_gap <= gap_to_win_pct < max_gap and gap_range in gap_models:
                selected_gap_model = gap_models[gap_range]
                break
        
        # If we don't have a specific model for this gap range, use the overall model
        if selected_gap_model is None:
            gap_beta_params = beta_params
            gap_normal_params = normal_params
        else:
            gap_beta_params = selected_gap_model['beta']
            gap_normal_params = selected_gap_model['normal']
        
        # Model 1: Empirical Beta distribution
        beta_prob = empirical_beta_probability(required_preference_pct, gap_beta_params)
        
        # Model 2: Empirical Normal distribution
        normal_prob = empirical_normal_probability(required_preference_pct, gap_normal_params)
        
        # Model 3: Competitive model based on empirical data
        competitive_prob = empirical_competitive_probability(required_preference_pct, gap_to_win_pct, df)
        
        # Model 4: Uniform distribution (baseline comparison)
        uniform_prob = uniform_distribution_probability(required_preference_pct)
        
        # Model 5: RCV patterns model based on empirical data
        rcv_prob = empirical_rcv_patterns_probability(required_preference_pct, gap_to_win_pct, df)
        
        # Model 6: Combined model - weighted average of the most realistic models
        combined_prob = empirical_combined_model_probability(
            beta_prob, normal_prob, competitive_prob, rcv_prob
        )
        
        # Model 7: Bayesian model with empirical priors
        bayesian_prob = empirical_bayesian_probability(
            required_preference_pct, 
            gap_to_win_pct,
            df
        )
        
        # Store results
        results.append({
            'region': region,
            'election_id': election_id,
            'letter': letter,
            'gap_to_win_pct': gap_to_win_pct,
            'exhaust_pct': exhaust_pct,
            'required_net_advantage': required_net_advantage,
            'required_preference_pct': required_preference_pct,
            'beta_probability': beta_prob,
            'normal_probability': normal_prob,
            'competitive_probability': competitive_prob,
            'uniform_probability': uniform_prob,
            'rcv_probability': rcv_prob,
            'bayesian_probability': bayesian_prob,
            'combined_probability': combined_prob,
            'beta_params': gap_beta_params,
            'normal_params': gap_normal_params
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nEmpirical Gap-Based Analysis: Winning probabilities with data-derived distributions")
    
    print("\nBeta Distribution Model (empirically calibrated):")
    for region in ['NYC', 'Alaska']:
        region_df = results_df[results_df['region'] == region]
        for letter in sorted(region_df['letter'].unique()):
            letter_df = region_df[region_df['letter'] == letter]
            print(f"{region} Letter {letter} (n={len(letter_df)}):")
            print(f"  Mean probability: {letter_df['beta_probability'].mean():.2%}")
            print(f"  Median probability: {letter_df['beta_probability'].median():.2%}")
            high_prob = letter_df[letter_df['beta_probability'] > 0.5]
            print(f"  Elections with >50% chance: {len(high_prob)}/{len(letter_df)} ({len(high_prob)/len(letter_df):.1%})")
    
    # Print similar statistics for other models
    for model_name, column_name in [
        ('Normal Distribution Model', 'normal_probability'),
        ('Competitive Model', 'competitive_probability'),
        ('Uniform Distribution Model', 'uniform_probability'),
        ('RCV Patterns Model', 'rcv_probability'),
        ('Bayesian Model', 'bayesian_probability'),
        ('Combined Model', 'combined_probability')
    ]:
        print(f"\n{model_name} (empirically calibrated):")
        for region in ['NYC', 'Alaska']:
            region_df = results_df[results_df['region'] == region]
            for letter in sorted(region_df['letter'].unique()):
                letter_df = region_df[region_df['letter'] == letter]
                print(f"{region} Letter {letter} (n={len(letter_df)}):")
                print(f"  Mean probability: {letter_df[column_name].mean():.2%}")
                print(f"  Median probability: {letter_df[column_name].median():.2%}")
                high_prob = letter_df[letter_df[column_name] > 0.5]
                print(f"  Elections with >50% chance: {len(high_prob)}/{len(letter_df)} ({len(high_prob)/len(letter_df):.1%})")
    
    # Find elections with highest probability
    high_prob_elections = results_df.sort_values('combined_probability', ascending=False)
    
    # Print example of one competitive election in detail
    # Find a close election for example
    close_elections = results_df[results_df['gap_to_win_pct'] < 1].sort_values('gap_to_win_pct')
    
    if not close_elections.empty:
        example = close_elections.iloc[0]
        print("\nDETAILED EXAMPLE: Understanding the probability calculation")
        print(f"Election: {example['region']} - {example['election_id']}, Letter {example['letter']}")
        print(f"Gap to win: {example['gap_to_win_pct']:.2f}% (how much more B needs to win)")
        print(f"Exhausted ballots: {example['exhaust_pct']:.2f}%")
        print(f"Using Beta({example['beta_params'][0]:.2f}, {example['beta_params'][1]:.2f})")
        print(f"Using Normal(μ={example['normal_params'][0]:.2f}, σ={example['normal_params'][1]:.2f})")
        print("\nTo win, candidate B needs:")
        print(f"  Net advantage among exhausted: {example['required_net_advantage']:.1f}%")
        print(f"  Preference percentage required: {example['required_preference_pct']:.1f}% of exhausted ballots")
        print("\nProbability of this happening under different models:")
        print(f"  Beta model: {example['beta_probability']:.2%}")
        print(f"  Normal model: {example['normal_probability']:.2%}")
        print(f"  Competitive model: {example['competitive_probability']:.2%}")
        print(f"  Uniform model: {example['uniform_probability']:.2%}")
        print(f"  RCV patterns model: {example['rcv_probability']:.2%}")
        print(f"  Bayesian model: {example['bayesian_probability']:.2%}")
        print(f"  Combined model: {example['combined_probability']:.2%}")
        print("\nInterpretation: Even in this close election, the combined probability")
        print(f"that the exhausted ballots would change the outcome is {example['combined_probability']:.2%}.")
        
    print("\nElections with highest probability of outcome change if exhausted ballots were completed:")
    for i, (_, row) in enumerate(high_prob_elections.iterrows(), 1):
        if i > 5:
            break
        print(f"{i}. {row['region']} - {row['election_id']}, Letter {row['letter']}")
        print(f"   Gap to win: {row['gap_to_win_pct']:.2f}%, Exhaust: {row['exhaust_pct']:.2f}%")
        print(f"   Required net advantage: {row['required_net_advantage']:.1f}% among exhausted ballots")
        print(f"   Required preference: {row['required_preference_pct']:.1f}% of exhausted ballots must prefer B")
        print(f"   COMBINED PROBABILITY: {row['combined_probability']:.2%}")
        print(f"   Individual models: Beta {row['beta_probability']:.2%}, "
              f"Normal {row['normal_probability']:.2%}, "
              f"Competitive {row['competitive_probability']:.2%}, "
              f"Uniform {row['uniform_probability']:.2%}, "
              f"RCV {row['rcv_probability']:.2%}, "
              f"Bayesian {row['bayesian_probability']:.2%}")
    
    # Visualize the results
    plt.figure(figsize=(16, 12))
    
    # 1. Visualize empirical Beta distributions for different gap ranges
    plt.subplot(2, 2, 1)
    x = np.linspace(0, 1, 1000)
    
    for gap_range, params in gap_models.items():
        a, b = params['beta']
        plt.plot(x, stats.beta.pdf(x, a, b), 
                 label=f'Gap: {gap_range}%, Beta({a:.2f},{b:.2f}), μ={(a/(a+b)):.2f}')
    
    plt.axvline(0.5, color='k', linestyle='--')
    plt.title('Empirical Beta Distributions for Different Strategy Gap Levels')
    plt.xlabel('Proportion Preferring B over A')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Visualize empirical Normal distributions for different gap ranges
    plt.subplot(2, 2, 2)
    x = np.linspace(0, 100, 1000)
    
    for gap_range, params in gap_models.items():
        mean, std = params['normal']
        plt.plot(x, stats.norm.pdf(x, mean, std), 
                 label=f'Gap: {gap_range}%, μ={mean:.1f}, σ={std:.1f}')
    
    plt.axvline(50, color='k', linestyle='--')
    plt.title('Empirical Normal Distributions for Different Strategy Gap Levels')
    plt.xlabel('Percentage Preferring B over A')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Scatterplot of required preference vs. probability
    plt.subplot(2, 2, 3)
    for model in ['beta_probability', 'normal_probability', 
                 'competitive_probability', 'rcv_probability', 'bayesian_probability', 'combined_probability']:
        plt.scatter(results_df['required_preference_pct'], results_df[model], 
                   label=model.split('_')[0].title(), alpha=0.5)
    
    plt.axhline(0.5, color='r', linestyle='--')
    plt.axvline(50, color='r', linestyle='--')
    plt.xlabel('Required % of Exhausted Ballots Preferring B')
    plt.ylabel('Winning Probability')
    plt.title('Required Preference vs. Probability for Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Relationship between gap and probability
    plt.subplot(2, 2, 4)
    for model in ['beta_probability', 'normal_probability', 
                 'competitive_probability', 'bayesian_probability', 'combined_probability']:
        plt.scatter(results_df['gap_to_win_pct'], results_df[model], 
                   label=model.split('_')[0].title(), alpha=0.5)
    
    plt.axhline(0.5, color='r', linestyle='--')
    plt.xlabel('Gap to Win Percentage')
    plt.ylabel('Winning Probability')
    plt.title('Gap to Win vs. Probability for Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('empirical_gap_based_probabilities.png', dpi=300)
    
    # Save results to CSV
    results_df.to_csv('empirical_gap_based_probabilities.csv', index=False)
    
    return results_df

def calculate_beta_parameters_from_data(strategy_values):
    """
    Calculate Beta distribution parameters directly from observed strategy values.
    
    Args:
        strategy_values: Array of strategy percentages
        
    Returns:
        tuple: (a, b) parameters for Beta distribution
    """
    # Convert percentages to proportions
    proportions = strategy_values / 100
    
    # Calculate mean and variance
    mean = np.mean(proportions)
    variance = np.var(proportions)
    
    # Handle edge cases
    if variance == 0 or mean == 0 or mean == 1:
        # Fallback parameters based on overall mean
        if mean < 0.5:
            return (3.5, 6.5)  # Right-skewed
        else:
            return (6.5, 3.5)  # Left-skewed
    
    # Calculate Beta parameters from mean and variance
    # For Beta(a,b): mean = a/(a+b) and variance = ab/((a+b)²(a+b+1))
    t = mean * (1 - mean) / variance - 1
    a = mean * t
    b = (1 - mean) * t
    
    # Ensure parameters are reasonable
    a = max(0.5, min(a, 20))  # Cap between 0.5 and 20
    b = max(0.5, min(b, 20))  # Cap between 0.5 and 20
    
    return (a, b)

def calculate_normal_params_from_data(strategy_values):
    """
    Calculate Normal distribution parameters directly from strategy values.
    
    Args:
        strategy_values: Array of strategy percentages
        
    Returns:
        tuple: (mean, std_dev) parameters for Normal distribution
    """
    # Calculate mean and standard deviation
    mean = np.mean(strategy_values)
    std_dev = np.std(strategy_values)
    
    # Ensure std_dev is reasonable
    std_dev = max(2.0, std_dev)  # Minimum std_dev of 2.0 to avoid overconfidence
    
    return (mean, std_dev)

def empirical_beta_probability(required_preference_pct, beta_params):
    """
    Calculate probability using a Beta distribution with empirically derived parameters.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        beta_params: Tuple of (a, b) parameters for Beta distribution
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Convert from percentage to proportion
    required_proportion = required_preference_pct / 100
    
    # Unpack Beta parameters
    a, b = beta_params
    
    # Probability is P(X ≥ required_proportion) where X ~ Beta(a,b)
    # Using survival function (1 - CDF)
    return 1 - stats.beta.cdf(required_proportion, a, b)

def empirical_normal_probability(required_preference_pct, normal_params):
    """
    Calculate probability using a Normal distribution with empirically derived parameters.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        normal_params: Tuple of (mean, std_dev) parameters for Normal distribution
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Unpack Normal parameters
    mean, std_dev = normal_params
    
    # Probability is P(X ≥ required_preference_pct) where X ~ N(mean, std_dev)
    # Using survival function (1 - CDF)
    return 1 - stats.norm.cdf(required_preference_pct, mean, std_dev)

def empirical_competitive_probability(required_preference_pct, gap_to_win_pct, df):
    """
    Calculate probability using a model based on observed relationship between
    gap size and completion patterns in the data.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        gap_to_win_pct: Gap that candidate B needs to close to win
        df: DataFrame with election data
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Find similar elections with similar gap sizes
    similar_gap_elections = df[(df['strategy'] > gap_to_win_pct*0.8) & 
                              (df['strategy'] < gap_to_win_pct*1.2)]
    
    if len(similar_gap_elections) >= 5:
        # Calculate average voter preference based on similar elections
        estimated_preference = 50 - similar_gap_elections['strategy'].mean() * 0.7
    else:
        # Fallback if not enough similar elections
        estimated_preference = 50 - gap_to_win_pct * 0.7
    
    # Cap the estimated preference
    estimated_preference = max(30, min(estimated_preference, 50))
    
    # Calculate probability based on how far required_preference_pct is from estimated_preference
    if required_preference_pct <= estimated_preference:
        # If required is less than estimated, high probability
        normalized = required_preference_pct / estimated_preference
        return max(0, 1 - 0.5 * normalized)
    else:
        # If required is more than estimated, low probability that declines exponentially
        normalized = (required_preference_pct - estimated_preference) / (100 - estimated_preference)
        return 0.5 * np.exp(-3 * normalized)

def uniform_distribution_probability(required_preference_pct):
    """
    Calculate probability assuming uniform distribution of preferences.
    
    This model assumes that each exhausted ballot, if completed, would have
    an equal chance of ranking any candidate first. Included as a baseline.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Under uniform distribution, each pairwise comparison is 50/50
    # So probability declines linearly from 1 at 0% to 0 at 100%
    if required_preference_pct <= 50:
        return 1.0  # Certain if required is less than 50%
    else:
        # Linear decline from 1 at 50% to 0 at 100%
        return 2 * (1 - required_preference_pct / 100)

def empirical_rcv_patterns_probability(required_preference_pct, gap_to_win_pct, df):
    """
    Calculate probability based on observed patterns in RCV elections.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        gap_to_win_pct: Gap that candidate B needs to close to win
        df: DataFrame with election data
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Group by letter to see how different candidates perform
    letter_stats = df.groupby('letter')['strategy'].agg(['mean', 'count'])
    
    # Calculate average expected preference based on empirical data
    # Filter for non-winners with sufficient data points
    viable_letters = letter_stats[(letter_stats['count'] >= 5) & 
                                 (letter_stats.index != 'A')]
    
    if not viable_letters.empty:
        # Use the mean strategy gap to estimate expected preference
        avg_gap = viable_letters['mean'].mean()
        expected_preference = 50 - avg_gap * 0.5
    else:
        # Fallback if not enough data
        expected_preference = 48 - gap_to_win_pct * 0.8
    
    # Cap the expected preference
    expected_preference = max(35, min(expected_preference, 50))
    
    if required_preference_pct <= expected_preference:
        # Linear from 1.0 at 0% to 0.5 at expected_preference
        normalized = required_preference_pct / expected_preference
        return 1 - 0.5 * normalized
    else:
        # Exponential decay from 0.5 at expected_preference to 0 at 100%
        normalized = (required_preference_pct - expected_preference) / (100 - expected_preference)
        return 0.5 * np.exp(-3 * normalized)

def empirical_combined_model_probability(beta_prob, normal_prob, competitive_prob, rcv_prob):
    """
    Calculate a weighted average of the most realistic probability models.
    
    Args:
        beta_prob: Probability from Beta distribution model
        normal_prob: Probability from Normal distribution model
        competitive_prob: Probability from competitive model
        rcv_prob: Probability from RCV patterns model
        
    Returns:
        float: Combined probability
    """
    # Weights for each model (sum to 1)
    # These weights reflect confidence in each model's ability to accurately
    # represent exhausted ballot preferences based on empirical data
    weights = {
        'beta': 0.30,        # Theoretically sound and adaptable
        'normal': 0.20,      # Good for central tendency modeling
        'competitive': 0.25, # Reflects electoral competitiveness
        'rcv': 0.25          # Based on empirical RCV patterns
    }
    
    # Calculate weighted average
    return (weights['beta'] * beta_prob + 
            weights['normal'] * normal_prob + 
            weights['competitive'] * competitive_prob + 
            weights['rcv'] * rcv_prob)

def empirical_bayesian_probability(required_preference_pct, gap_to_win_pct, df):
    """
    Calculate probability using a Bayesian approach with empirical priors.
    
    Args:
        required_preference_pct: Percentage of exhausted ballots that must prefer B
        gap_to_win_pct: Gap that candidate B needs to close to win
        df: DataFrame with election data
        
    Returns:
        float: Probability that enough exhausted ballots prefer B
    """
    # Get baseline Beta parameters from the data
    a, b = calculate_beta_parameters_from_data(df['strategy'].values)
    
    # Find similar elections to use as prior evidence
    similar_elections = df[(df['strategy'] > gap_to_win_pct*0.5) & 
                          (df['strategy'] < gap_to_win_pct*1.5)]
    
    if len(similar_elections) >= 5:
        # Calculate average preference for B based on similar elections
        similar_strategies = similar_elections['strategy'].values
        previous_for_b = 0.5 - np.mean(similar_strategies) / 200  # Convert to proportion
        
        # Update our Beta parameters with this evidence
        n_samples = min(100, len(similar_elections))  # Cap the sample size
        
        # Discount factor - less weight for larger gaps
        discount = max(0.3, 0.8 - gap_to_win_pct * 0.05)
        
        # Update parameters
        previous_for_a = 1 - previous_for_b
        a += discount * n_samples * previous_for_b
        b += discount * n_samples * previous_for_a
    
    # Convert from percentage to proportion
    required_proportion = required_preference_pct / 100
    
    # Calculate probability using the posterior Beta distribution
    return 1 - stats.beta.cdf(required_proportion, a, b)

if __name__ == "__main__":
    analyze_gap_based_probabilities() 