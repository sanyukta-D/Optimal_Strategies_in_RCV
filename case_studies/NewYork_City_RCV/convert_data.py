import os
import csv
import json
import pandas as pd
from string import ascii_uppercase

import pandas as pd

def clean_rcv_data(input_file):
    """
    Reads a CSV file and processes ranking columns in a general way.
    
    The function will:
      - Detect columns that start with 'rank' (e.g. 'rank1', 'rank2', ...)
      - Replace entries 'skipped', 'overvote', and 'writein' with an empty string in those columns
      - Remove rows where all ranking columns are empty after replacement
      - Rename ranking columns to Choice_1, Choice_2, etc.
      - Optionally rename other columns (e.g. 'Precinct' to 'RowNumber' and 'source_file' to 'Election')
      - Select only the required columns (RowNumber if available, plus all Choice_* columns)
      - Reset the index
      
    :param input_file: Path to the CSV file.
    :return: Cleaned DataFrame.
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file, low_memory=False)
    
    # Dynamically detect ranking columns (case insensitive)
    ranking_columns = [col for col in df.columns if col.lower().startswith("rank")]
    
    # If ranking columns have numbers, sort them by the numerical part (assumes format like 'rank1', 'rank2', ...)
    try:
        ranking_columns = sorted(ranking_columns, key=lambda x: int(''.join(filter(str.isdigit, x))))
    except ValueError:
        # If conversion fails, keep original order
        pass
    
    # Replace irrelevant words in the detected ranking columns
    irrelevant = ['skipped', 'overvote', 'writein']
    df[ranking_columns] = df[ranking_columns].replace(irrelevant, '')
    
    # Remove rows where all ranking columns are empty after replacement
    df = df.loc[~(df[ranking_columns] == '').all(axis=1)]
    
    # Prepare a column mapping for renaming columns
    column_mapping = {}
    if 'Precinct' in df.columns:
        column_mapping['Precinct'] = 'RowNumber'
    if 'source_file' in df.columns:
        column_mapping['source_file'] = 'Election'
    
    # Rename ranking columns to Choice_1, Choice_2, etc.
    for i, col in enumerate(ranking_columns, 1):
        column_mapping[col] = f'Choice_{i}'
    
    df = df.rename(columns=column_mapping)
    
    # Build list of required columns
    required_columns = []
    if 'RowNumber' in df.columns:
        required_columns.append('RowNumber')
    required_columns += [f'Choice_{i}' for i in range(1, len(ranking_columns)+1)]
    df = df[required_columns]
    
    # Reset the index
    df = df.reset_index(drop=True)
    
    return df


def generate_simplified_filename(original_filename):
    """
    Convert long filenames to simplified format.
    Example: NewYorkCity_06222021_DEMCouncilMember13thCouncilDistrict.csv
    -> NYC_DEM_13_Council_District.csv
    """
    # Remove file extension
    name_without_ext = original_filename.split('.')[0]
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # Create simplified name
    simplified_name = 'NYC'
    
    # Add party (DEM)
    if any('DEM' in part for part in parts):
        simplified_name += '_DEM'
    
    # Extract council district number (fallback to '13' if not found)
    council_district = next((part for part in parts if 'Council' in part and part.replace('th','').isdigit()), '13')
    simplified_name += f'_{council_district}_Council_District.csv'
    
    return simplified_name

def process_single_file(input_file):
    """
    Processes a single CSV file by cleaning its contents,
    saving the cleaned file in a "cleaned_files" directory, and returning its path.
    """
    # Ensure cleaned_files directory exists
    cleaned_dir = 'cleaned_files'
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Clean the data
    cleaned_df = clean_rcv_data(input_file)
    
    # Generate new filename
    original_filename = os.path.basename(input_file)
    new_filename = generate_simplified_filename(original_filename)
    output_path = os.path.join(cleaned_dir, new_filename)
    
    # Save cleaned file
    cleaned_df.to_csv(output_path, index=False)
    #print(f"Processed: {original_filename} -> {new_filename}")
    
    return cleaned_df, output_path

def create_candidate_mapping(csv_filename):
    """
    Reads the given CSV file, finds all unique candidate names
    (ignoring empty or 'skipped' entries) from dynamically detected "Choice_" columns,
    and assigns each a letter code (A, B, C, ...).

    :param csv_filename: Path to the CSV file.
    :return: Dictionary of {candidate_name: letter_code}.
    """
    candidates = set()

    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Dynamically detect columns that start with "Choice_"
        choice_columns = [col for col in reader.fieldnames if col.startswith("Choice_")]
        # Optionally sort the columns by their numeric suffix if possible
        try:
            choice_columns = sorted(choice_columns, key=lambda x: int(x.split('_')[1]))
        except (ValueError, IndexError):
            pass

        # Collect candidate names from all detected choice columns
        for row in reader:
            for col in choice_columns:
                name = row[col].strip()
                if name and name.lower() != 'skipped':
                    candidates.add(name)

    # Sort candidates alphabetically (or any desired custom order)
    sorted_candidates = sorted(candidates)

    # Assign a letter code to each candidate
    mapping = {}
    for i, candidate_name in enumerate(sorted_candidates):
        if i < len(ascii_uppercase):
            mapping[candidate_name] = ascii_uppercase[i]
        else:
            # For more than 26 candidates, create extra codes
            mapping[candidate_name] = f"Extra_{i}"

    return mapping


def save_candidate_mapping(cleaned_csv_file):
    """
    Creates a candidate mapping from the cleaned CSV file and saves it as a JSON file.
    The JSON file is saved in the same directory as the cleaned CSV with a '_mapping.json' suffix.
    """
    mapping = create_candidate_mapping(cleaned_csv_file)
    mapping_filename = cleaned_csv_file.replace(".csv", "_mapping.json")
    with open(mapping_filename, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    print(f"Candidate mapping saved to: {mapping_filename}")
    return mapping_filename

if __name__ == "__main__":
    # Example usage with your file path
    file_path = 'nyc_files/NewYorkCity_06222021_DEMCouncilMember13thCouncilDistrict.csv'
    cleaned_df, processed_file = process_single_file(file_path)
    mapping_file = save_candidate_mapping(processed_file)
    
    # Optionally, load the saved mapping to verify
    with open(mapping_file, "r", encoding="utf-8") as f:
        candidate_mapping = json.load(f)
    print(candidate_mapping)
