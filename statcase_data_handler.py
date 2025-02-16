import pybaseball as pb
import pandas as pd

# Define the date range for the 2024 season
start_date = '2024-03-01'  #  MLB Regular Season typically starts in late March/early April, but March 1st is a safe starting point to capture Spring Training if needed and early season games. Adjust if necessary.
end_date = '2024-12-31'  #  End of the regular season and potentially playoffs. You can adjust this to a more current date as the season progresses if you only need data up to a certain point.  'today' or a specific recent date is also valid.

try:
    # Fetch Statcast pitch data for the 2024 season
    data_2024 = pb.statcast(start_dt=start_date, end_dt=end_date)

    # Print the first few rows to verify the data
    print(data_2024.head())

    # Print the shape of the dataframe to see how much data you pulled
    print(f"\nShape of 2024 Statcast data: {data_2024.shape}")

except Exception as e:
    print(f"Error fetching Statcast data: {e}")
    print("Please ensure you have pybaseball installed (`pip install pybaseball`) and your internet connection is working.")
    print("Also, data might not be available yet if it's very early in the 2024 season or if there are issues with the data source.")