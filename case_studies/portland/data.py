import pandas as pd
from case_studies.portland.load_district_data import district_data

def create_portland_strategy_data():
    """
    Create strategy data for Portland districts in the same format as NYC/Alaska data.
    All strategies are selfish (candidates helping themselves only).
    Only include non-winning candidates with actual strategy data.
    
    Returns:
        dict: Strategy data for each district with exhaust_strategy analysis format
    """
    
    # Strategy data based on provided information
    # District 1: From figure - mapping candidate names to their strategy percentages
    # EXCLUDE WINNERS: Candace Avalos (A), Loretta Smith (C), Jamie Dumphy (B)
    district1_strategy_raw = {
        "Terrence Hayes": 1.60,      # Hayes self-support (1.60%)
        "Steph Routh": 1.93,         # Routh self-support (1.93%)
        "Noah Ernst": 2.81,          # Ernst self-support (2.81%)
        "Timur Ender": 4.21          # Ender self-support (4.21%)
    }
    
    # District 2: Tiffani Penson: 5.69% to themself
    # EXCLUDE WINNERS: Sameer Kanal (A), Dan Ryan (B), Elana Pirtle-Guiney (C)
    district2_strategy_raw = {
        "Tiffani Penson": 5.69
    }
    
    # District 4: Eli Arnold: 1.12% to themself
    # EXCLUDE WINNERS: Olivia Clark (A), Mitch Green (B), Eric Zimmerman (C)
    district4_strategy_raw = {
        "Eli Arnold": 1.12
    }
    
    # Convert to letter-based format using candidate mappings
    portland_strategy_data = {}
    
    # District 1
    candidates_mapping_1 = district_data[1]['candidates_mapping']
    district1_strategies = {}
    for candidate_name, percentage in district1_strategy_raw.items():
        if candidate_name in candidates_mapping_1:
            letter = candidates_mapping_1[candidate_name]
            # Format as list with single value (matching NYC/Alaska format)
            district1_strategies[letter] = [percentage]
    
    portland_strategy_data[1] = {
        'district': 1,
        'strategies': district1_strategies,
        'raw_data': district1_strategy_raw,
        'candidates_mapping': candidates_mapping_1
    }
    
    # District 2  
    candidates_mapping_2 = district_data[2]['candidates_mapping']
    district2_strategies = {}
    for candidate_name, percentage in district2_strategy_raw.items():
        if candidate_name in candidates_mapping_2:
            letter = candidates_mapping_2[candidate_name]
            district2_strategies[letter] = [percentage]
    
    portland_strategy_data[2] = {
        'district': 2,
        'strategies': district2_strategies,
        'raw_data': district2_strategy_raw,
        'candidates_mapping': candidates_mapping_2
    }
    
    # District 4
    candidates_mapping_4 = district_data[4]['candidates_mapping']
    district4_strategies = {}
    for candidate_name, percentage in district4_strategy_raw.items():
        if candidate_name in candidates_mapping_4:
            letter = candidates_mapping_4[candidate_name]
            district4_strategies[letter] = [percentage]
    
    portland_strategy_data[4] = {
        'district': 4,
        'strategies': district4_strategies,
        'raw_data': district4_strategy_raw,
        'candidates_mapping': candidates_mapping_4
    }
    
    return portland_strategy_data

def get_portland_exhaust_data():
    """
    Get exhaustion data for Portland districts to match with strategy data.
    Extract round-by-round exhaustion data correctly using existing portland_exhaustion code.
    
    Returns:
        dict: Exhaust data for each district
    """
    import sys
    from io import StringIO
    from contextlib import redirect_stdout
    from portland_exhaustion import print_exhaustion_analysis
    
    portland_exhaust_data = {}
    
    for district in [1, 2, 4]:
        try:
            # Get candidate mapping to convert names to letters
            candidates_mapping = district_data[district]['candidates_mapping']
            # Create name lookup (handle partial names)
            name_to_letter = {}
            for full_name, letter in candidates_mapping.items():
                # Add full name
                name_to_letter[full_name] = letter
                # Add first name only
                first_name = full_name.split()[0]
                name_to_letter[first_name] = letter
                # Handle special cases like "Michael (Mike) Sands"
                if '(' in full_name and ')' in full_name:
                    nickname = full_name[full_name.find('(')+1:full_name.find(')')]
                    name_to_letter[nickname] = letter
            
            # Capture the output from print_exhaustion_analysis
            captured_output = StringIO()
            with redirect_stdout(captured_output):
                print_exhaustion_analysis(district)
            
            output = captured_output.getvalue()
            
            # Parse the table to extract round data
            lines = output.split('\n')
            table_started = False
            round_data = []
            
            for line in lines:
                if 'Round  Exhausted %  Used %' in line:
                    table_started = True
                    continue
                elif table_started and line.strip().startswith('='):
                    break
                elif table_started and line.strip() and not line.strip().startswith('-'):
                    # Parse table row
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        round_num = int(parts[0])
                        exhausted_pct = float(parts[1])
                        used_pct = float(parts[2])
                        candidate_type = parts[3]
                        # Candidate name is the rest - extract the letter from parentheses
                        candidate_info = ' '.join(parts[4:])
                        
                        # Extract candidate letter from parentheses: "(Name)" 
                        if '(' in candidate_info and ')' in candidate_info:
                            start = candidate_info.find('(') + 1
                            end = candidate_info.find(')', start)
                            if end > start:
                                candidate_name = candidate_info[start:end]
                                
                                # Map name to letter
                                candidate_letter = None
                                if candidate_name in name_to_letter:
                                    candidate_letter = name_to_letter[candidate_name]
                                else:
                                    # Try to find partial match
                                    for name, letter in name_to_letter.items():
                                        if candidate_name in name or name in candidate_name:
                                            candidate_letter = letter
                                            break
                                
                                if candidate_letter:
                                    round_data.append({
                                        'Round': round_num,
                                        'Exhausted_pct': exhausted_pct,
                                        'Used_pct': used_pct,
                                        'Type': candidate_type,
                                        'Candidate': candidate_letter,
                                        'Name': candidate_name
                                    })
            
            # Now extract exhaust percentages for each candidate from PREVIOUS round
            exhausted_ballots_dict_percent = {}
            
            for i, round_info in enumerate(round_data):
                candidate = round_info['Candidate']
                
                # For each candidate, use the exhaustion percentage from the PREVIOUS round
                if i == 0:
                    # First round, no previous round, so use 0
                    prev_exhausted_pct = 0.0
                else:
                    # Use exhaustion percentage from previous round
                    prev_exhausted_pct = round_data[i-1]['Exhausted_pct']
                
                exhausted_ballots_dict_percent[candidate] = round(prev_exhausted_pct, 3)
            
            # Get total ballots count
            df = district_data[district]['df']
            total_ballots = len(df)
            
            print(f"\nDistrict {district} Extracted exhaustion data:")
            print(f"{'Round':<6} {'Letter':<6} {'Name':<15} {'Type':<10} {'Prev Exhaust %':<15} {'Current Exhaust %':<15}")
            print("-" * 85)
            for i, round_info in enumerate(round_data):
                prev_pct = 0.0 if i == 0 else round_data[i-1]['Exhausted_pct']
                current_pct = round_info['Exhausted_pct']
                print(f"{round_info['Round']:<6} {round_info['Candidate']:<6} {round_info['Name']:<15} {round_info['Type']:<10} {prev_pct:<15.2f} {current_pct:<15.2f}")
            
            print(f"\nFinal exhaust percentages for District {district}:")
            for candidate, pct in exhausted_ballots_dict_percent.items():
                # Get candidate name for display
                candidate_name = "Unknown"
                for name, letter in candidates_mapping.items():
                    if letter == candidate:
                        candidate_name = name
                        break
                print(f"  {candidate} ({candidate_name}): {pct}%")
            
            portland_exhaust_data[district] = {
                'district': district,
                'exhaust_percents': exhausted_ballots_dict_percent,
                'total_ballots': total_ballots,
                'round_data': round_data
            }
            
        except Exception as e:
            print(f"Error processing district {district}: {e}")
            import traceback
            traceback.print_exc()
            portland_exhaust_data[district] = None
    
    return portland_exhaust_data

def create_portland_summary_dataframe():
    """
    Create a summary dataframe in the same format as NYC/Alaska data for analysis.
    
    Returns:
        pd.DataFrame: Summary dataframe with strategy and exhaust data
    """
    strategy_data = create_portland_strategy_data()
    exhaust_data = get_portland_exhaust_data()
    
    summary_rows = []
    
    for district in [1, 2, 4]:
        if district in strategy_data and district in exhaust_data and exhaust_data[district] is not None:
            # Create row in format similar to NYC/Alaska data
            row = {
                'file_name': f'Portland_District_{district}',
                'district': district,
                'Strategies': str(strategy_data[district]['strategies']),
                'exhaust_percents': str(exhaust_data[district]['exhaust_percents']),
                'total_ballots': exhaust_data[district]['total_ballots'],
                'region': 'Portland'
            }
            summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)

def print_portland_strategy_summary():
    """
    Print a formatted summary of Portland strategy data.
    """
    strategy_data = create_portland_strategy_data()
    
    print("Portland Strategy Data Summary")
    print("=" * 60)
    
    for district, data in strategy_data.items():
        print(f"\nDistrict {district}:")
        print("-" * 20)
        
        candidates_mapping = data['candidates_mapping']
        strategies = data['strategies']
        raw_data = data['raw_data']
        
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in candidates_mapping.items()}
        
        print("Candidate strategies:")
        for letter, percentage_list in strategies.items():
            candidate_name = reverse_mapping[letter]
            percentage = percentage_list[0]
            print(f"  {letter} ({candidate_name}): {percentage}%")
        
        print(f"\nTotal candidates with strategies: {len(strategies)}")
        print(f"Raw data candidates: {list(raw_data.keys())}")

if __name__ == "__main__":
    # Test the functions
    print_portland_strategy_summary()
    
    # Create summary dataframe
    print("\n" + "="*60)
    print("Creating summary dataframe...")
    
    try:
        df = create_portland_summary_dataframe()
        print(f"Summary dataframe created with {len(df)} rows")
        print("\nDataframe preview:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error creating summary dataframe: {e}") 