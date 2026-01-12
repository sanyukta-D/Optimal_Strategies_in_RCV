import pandas as pd
import os

# Base directory for data
BASE_DIR = "Case_Studies/Portland_City_Council_Data_and_Analysis"

# Dictionary to store all data
district_data = {}

# Load data for each district
for district_num in range(1, 5):  # Districts 1-4
    # Load DataFrame
    file_path = os.path.join(BASE_DIR, f"Dis_{district_num}", f"Election_results_dis{district_num}.csv")
    df = pd.read_csv(file_path)
    df = df.drop(columns=['RowNumber'])
    
    # Get bootstrap samples directory
    bootstrap_dir = os.path.join(BASE_DIR, f"Dis_{district_num}", f"bootstrap_samples_dis{district_num}")
    
    # Get list of bootstrap files if directory exists
    bootstrap_files = []
    if os.path.exists(bootstrap_dir):
        bootstrap_files = os.listdir(bootstrap_dir)
    
    # Store DataFrame and bootstrap files in dictionary
    district_data[district_num] = {
        'df': df,
        'bootstrap_dir': bootstrap_dir,
        'bootstrap_files': bootstrap_files
    }
    
# Define candidate mappings
district_data[1]['candidates_mapping'] = {
    "Candace Avalos": "A",
    "Jamie Dunphy": "B",
    "Loretta Smith": "C",
    "Noah Ernst": "E",
    "Terrence Hayes":"D",
    "Steph Routh": "F",
    "Timur Ender": "G",
    "Doug Clove": "H",
    "Peggy Sue Owens": "I",
    "David Linn": "J",
    "Joe Allen": "K",
    "Michael (Mike) Sands": "L",
    "Deian Salazar": "M",
    "Cayle Tern": "N",
    "Thomas Shervey": "O",
    "Joe Furi": "P"
}

district_data[2]['candidates_mapping'] = candidates_mapping_2 = {
    "Sameer Kanal": "A",
    "Dan Ryan": "B",
    "Elana Pirtle-Guiney": "C",
    "Tiffani Penson": "D",
    "Michelle DePass": "E",
    "Nat West": "F",
    "Marnie Glickman": "G",
    "Jonathan Tasini": "H",
    "Bob Simril": "I",
    "Mariah Hudson": "J",
    "Michael (Mike) Marshall": "K",
    "James Armstrong": "L",
    "Chris Olson": "M",
    "Debbie Kitchin": "N",
    "Jennifer Park": "O",
    "Nabil Zaghloul": "P",
    "Will Mespelt": "Q",
    "Laura Streib": "R",
    "Reuben Berlin": "S",
    "Liz Taylor": "T",
    "Sam Sachs": "U",
    "Antonio Jamal PettyJohnBlue": "V"
}

district_data[3]['candidates_mapping'] = candidates_mapping_3 = {
    "Steve Novick": "A",
    "Angelita Morillo": "B",
    "Tiffany Koyama Lane": "C",
    "Kezia Wanner": "D",
    "Rex Burkholder": "E",
    "Jesse Cornett": "F",
    "Harrison Kass": "G",
    "Daniel DeMelo": "H",
    "Philippe Knab": "I",
    "Sandeep Bali": "J",
    "Cristal Azul Otero": "K",
    "Jonathan (Jon) Walker": "L",
    "Chris Flanary": "M",
    "Melodie Beirwagen": "N",
    "Matthew (Matt) Anderson": "O",
    "Ahlam K Osman": "P",
    "Luke Zak": "Q",
    "Heart Free Pham": "R",
    "Brian Conley": "S",
    "Terry Parker": "T",
    "Dan Gilk": "U",
    "Christopher Brummer": "V",
    "John Sweeney": "W",
    "Kelly Janes (KJ)": "X",
    "Theo Hathaway Saner": "Y",
    "Jaclyn Smith-Moore": "Z",
    "Patrick Hilton": "a",
    "David O'Connor": "b",
    "Kenneth (Kent) R Landgraver III": "c",
    "Clifford Higgins": "d"
}


district_data[4]['candidates_mapping'] = candidates_mapping = {
    "Olivia Clark": "A",
    "Mitch Green": "B",
    "Eric Zimmerman": "C",
    "Eli Arnold": "D",
    "Chad Lykins": "E",
    "Sarah Silkie": "F",
    "Bob Weinstein": "G",
    "Lisa Freeman": "H",
    "Tony Morse": "I",
    "Ben Hufford": "J",
    "Kevin Goldsmith": "K",
    "Andra Vltavin": "L",
    "Stanley Penkin": "M",
    "John Toran": "N",
    "Chloe Mason": "O",
    "Bob Callahan": "P",
    "Moses Ross": "Q",
    "Ciatta R Thompson": "R",
    "Raquel Coyote": "S",
    "Mike DiNapoli": "T",
    "John J Goldsmith": "U",
    "Chris Henry": "V",
    "Joseph (Joe) Alfone": "W",
    "Michael Trimble": "X",
    "Kelly Doyle": "Y",
    "Brandon Farley": "Z",
    "Patrick Cashman": "a",
    "Tony Schwartz": "b",
    "Lee Odell": "c",
    "L Christopher Regis": "d"
}


# Now you can access any district's data like this:
# district_data[2]['df'] - DataFrame for district 2
# district_data[2]['candidates_mapping'] - Mapping for district 2