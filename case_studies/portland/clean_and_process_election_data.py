import pandas as pd

# Function to extract the top 6 ranked choices for each voter with policies for incorrectly marked ballots
def extract_top_choices(row, choice_columns):
    choices = {}

    # Iterate through the columns to collect rankings and check for issues
    for col in choice_columns:
        if row[col] > 0:  # A valid vote
            parts = col.split(":")
            candidate = parts[-2]  # Extract candidate name
            try:
                rank = int(parts[-4])  # Extract rank

                # Check for repeat rankings: Keep only the highest rank for a candidate
                if candidate in choices:
                    if rank < choices[candidate]:
                        choices[candidate] = rank  # Update to the higher rank
                else:
                    choices[candidate] = rank
            except ValueError:
                continue

    # Handle overvotes by skipping ranks with multiple candidates
    rank_to_candidates = {}
    for candidate, rank in choices.items():
        if rank not in rank_to_candidates:
            rank_to_candidates[rank] = []
        rank_to_candidates[rank].append(candidate)

    # Resolve overvotes by keeping only the first candidate in rank order and skipping others
    resolved_choices = []
    for rank in sorted(rank_to_candidates.keys()):
        candidates_at_rank = rank_to_candidates[rank]
        if len(candidates_at_rank) == 1:
            resolved_choices.append((rank, candidates_at_rank[0]))
        # If there's an overvote (multiple candidates), skip this rank

    # Sort choices by rank and return the top 6, padded with None if fewer than 6
    sorted_choices = sorted(resolved_choices, key=lambda x: x[0])[:6]

    return [candidate for _, candidate in sorted_choices] + [None] * (6 - len(sorted_choices))


# Load the dataset
file_path = 'dis4long.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Identify choice columns
choice_columns = [col for col in df.columns if "Choice_" in col]

# Apply the extraction function
simplified_df = df.apply(lambda row: extract_top_choices(row, choice_columns), axis=1, result_type='expand')
simplified_df.columns = [f"Choice_{i+1}" for i in range(6)]

# Add RowNumber to identify ballots
simplified_df.insert(0, "RowNumber", df["RowNumber"])

# Save to a new CSV file
output_file_path = 'dis4.csv'
simplified_df.to_csv(output_file_path, index=False)

# Display summary statistics
average_choices = simplified_df.iloc[:, 1:].notna().sum(axis=1).mean()
first_choice_counts = simplified_df["Choice_1"].value_counts()

print(f"Average choices per person: {average_choices}")
print("First choice counts:")
print(first_choice_counts)

