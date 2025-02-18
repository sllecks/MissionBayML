import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
import warnings

# Filter out the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='errors=\'ignore\' is deprecated')

def get_pitch_data_for_year(year):
    """
    Fetches all pitch data for a specified year and saves it to a CSV file.
    
    Args:
        year (int): The year to fetch pitch data for
    """
    # Create start and end dates for the specified year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # Initialize empty DataFrame to store all pitch data
    all_pitch_data = pd.DataFrame()
    
    # Break the year into smaller chunks to handle rate limiting
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_datetime:
        # Get end date for current chunk (2 weeks later)
        chunk_end = min(current_date + timedelta(days=14), end_datetime)
        
        print(f"Fetching data from {current_date.date()} to {chunk_end.date()}")
        
        try:
            # Fetch pitch data for current chunk
            chunk_data = statcast(
                start_dt=current_date.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d")
            )
            
            if chunk_data is not None and not chunk_data.empty:
                all_pitch_data = pd.concat([all_pitch_data, chunk_data], ignore_index=True)
            
        except Exception as e:
            print(f"Error fetching data for period {current_date.date()} to {chunk_end.date()}: {str(e)}")
        
        # Move to next chunk
        current_date = chunk_end + timedelta(days=1)
    
    # Save the complete dataset to CSV
    if not all_pitch_data.empty:
        output_file = f"pitch_data_{year}.csv"
        all_pitch_data.to_csv(output_file, index=False)
        print(f"Successfully saved pitch data to {output_file}")
        print(f"Total pitches collected: {len(all_pitch_data)}")
    else:
        print("No data was collected")

if __name__ == "__main__":
    # Example usage
    year = 2023  # Change this to the desired year
    get_pitch_data_for_year(year)
