import pandas as pd

# Read the CSV file
df = pd.read_csv('baseball_stats.csv')

# Calculate per at-bat statistics
# Assuming 'AB' is the column name for at-bats
at_bat_columns = ['H', 'HR', 'RBI', 'R', 'BB', 'SO', 'WAR']  # Add or remove stats as needed

# Create new columns with per at-bat calculations
for col in at_bat_columns:
    if col in df.columns:
        df[f'{col}_per_PA'] = df[col] / df['PA']

# Sort by WAR_per_PA in descending order and add rank
df = df.sort_values('WAR_per_PA', ascending=False)
df.insert(0, 'PA_RANK', range(1, len(df) + 1))

# Save the updated dataframe to a new CSV file
df.to_csv('baseball_stats_atbat.csv', index=False)
print("Per at-bat statistics have been calculated and saved to baseball_stats_atbat.csv")
