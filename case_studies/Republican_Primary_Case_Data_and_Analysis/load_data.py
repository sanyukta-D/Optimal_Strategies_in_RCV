import os
import pandas as pd

# Base directory for data
BASE_DIR = "Case_Studies/Republican_Primary_Case_Data_and_Analysis"

# Create a mapping from candidate names to variables
candidates_mapping = {
    "Trump": "A",
    "Haley": "B",
    "Ramaswamy": "C",
    "DeSantis": "D",
    "Christie":"E",
    "Pence": "F",
    "Scott": "G",
    "Hurd": "H",
    "Hutchinson": "I",
    "Youngkin": "J",
    "Burgum": "K",
    "Elder": "L",
    "Suarez": "M",
}


file_path = os.path.join(BASE_DIR, "poll_data.csv")
df = pd.read_csv(file_path)

bootstrap_dir = os.path.join(BASE_DIR,'Republican_primary_bootstrap_files')

# List all files in the directory
bootstrap_files = os.listdir(bootstrap_dir)

# Store DataFrame and bootstrap files in dictionary
election_data = {
    'df': df,
    'bootstrap_dir': bootstrap_dir,
    'bootstrap_files': bootstrap_files,
    'candidates_mapping': candidates_mapping
}