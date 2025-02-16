import pandas as pd
import unicodedata

def clean_name(name):
    # Name mapping for special cases
    name_mapping = {
        'Ha-Seong Kim': 'Ha-seong Kim',
        'Jonny DeLuca': 'Jonny Deluca',
        'Joshua Palacios': 'Josh Palacios',
        'Victor Scott II': 'Victor Scott',
        'Carlos D Rodriguez': 'Carlos Rodriguez',
        'Robert Hassell III': 'Robert Hassell',
        'Michael Siani': 'Mike Siani',
        'Ji Hwan Bae': 'Ji-Hwan Bae',
        'Benjamin Cowles': 'Ben Cowles',
        'Nick Ahmed': 'Nicholas Ahmed'
    }
    
    # Check if name is in our mapping
    if str(name) in name_mapping:
        name = name_mapping[str(name)]
    
    # Normalize unicode characters and remove accents
    cleaned = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    # Remove periods
    cleaned = cleaned.replace('.', '')
    # Remove Jr and Jr (with possible spaces)
    cleaned = cleaned.replace(' Jr', '').replace('Jr', '')
    return cleaned.strip()

# Read the CSV files
baseball_stats = pd.read_csv('baseball_stats_atbat.csv')
fantrax = pd.read_csv('fantrax-jan212025.csv')

# Clean names in both dataframes
baseball_stats['Name'] = baseball_stats['Name'].apply(clean_name)
fantrax['Player'] = fantrax['Player'].apply(clean_name)

# Remove duplicates from baseball_stats
baseball_stats = baseball_stats.drop_duplicates(subset=['Name'])

# Remove duplicates from fantrax, keeping the row with highest Score
fantrax = fantrax.sort_values('Score', ascending=False).drop_duplicates(subset=['Player'])

# Rename the column in fantrax
fantrax = fantrax.rename(columns={"Player": "Name"})

# Perform an outer join on Name column
merged_df = pd.merge(baseball_stats, fantrax, on='Name', how='left')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_baseball_stats.csv', index=False)

# Find unmatched players (where fantrax columns are null)
unmatched_players = merged_df[merged_df.iloc[:, baseball_stats.shape[1]:].isnull().all(axis=1)]['Name'].tolist()

# Print unmatched players
if unmatched_players:
    print("\nPlayers from baseball_stats_atbat.csv that weren't found in fantrax-jan182025.csv:")
    for player in unmatched_players:
        print(f"- {player}")
else:
    print("\nAll players were successfully matched!")