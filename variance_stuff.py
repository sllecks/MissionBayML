# notes

# stuff+ does a really good job of predicting the raw talent of pitches and players
# however, it does not do a good job of predicting the variance of pitches and players
# stuff- tries to predict the variance of pitches and players
# if we can create a model that can predict the variance of pitches and players, we can use it to create a more accurate model of a pitchers performance
# thinking that variance of a pitch or variance of a player can explain a lot to the success of a pitcher

# So the goal here is to create a robust stuff model that predicts the stuff of an individual pitch, then variance can be calculated from that.
# https://pbs.twimg.com/media/GjCdWh0bIAQnOdu?format=jpg&name=medium
# That's the high level model of the stuff model.
# First it starts with a swing model, then a ump model, then a contact model, then batted ball specs model, then a batted ball outcome model.  

# Other ideas:
# 1. Pair it with a strength of schedule model
# 2. Pair it with a field impact model (e.g. how much does the field impact the variance of a pitch)
# 3. Pair it with a weather model (e.g. how much does the weather impact the variance of a pitch)
# 4. Establish benchmarks for pitchers, like high variance pitches, low variance pitches, etc.
# 5. How does variance change from start to finish of a season? fatigue? 
# 6. Create a fatigue model?
# 7. Game context model? Inning, base runners, outs, etc.
# 8. Break it down by pitch type
# 9. Percentiles of variance for a pitchers outings, and then cut across other factors like time, fatigue, etc. 
# 10. Is a pitcher's pitch-to-pitch variance (within-outing SD) larger or smaller than their outing-to-outing variance (SD of outing averages)?
# 11. Large Within-Outing, Small Outing-to-Outing: Might suggest a pitcher who is consistently at a certain performance level per game, but within each game, their pitch quality varies quite a bit. Could be due to strategy (saving best pitches), fatigue, or simply natural pitch variation.
# 12. Small Within-Outing, Large Outing-to-Outing: Might suggest a pitcher whose individual pitches are quite consistent within a game, but their overall game performance fluctuates significantly from one outing to the next. Could be due to opponent matchups, physical condition on different days, mental game factors affecting whole outings.
# 13. Both Large: Highly variable at both pitch and outing levels. Potentially inconsistent or unpredictable performance.
# 14. Both Small: Very consistent and reliable pitcher, both within and across outings.


import pybaseball as pb
import pandas as pd
import numpy as np

# Placeholder functions for models -  REPLACE WITH ACTUAL ML MODELS for real application

def swing_model(pitch_data):
    """
    Simplified swing model: Predicts probability of a swing based on pitch location.
    In a real model, this would be a trained ML model (e.g., logistic regression, NN)
    using features like pitch location (plate_x, plate_z), pitch type, count, etc.
    """
    plate_x = pitch_data['plate_x']
    plate_z = pitch_data['plate_z']

    # Simplified rule: Pitches closer to the center of the plate are more likely to be swung at
    swing_probability = 0.8 - abs(plate_x) * 0.1 - abs(plate_z) * 0.1  # Example formula
    swing_probability = max(0.1, min(0.95, swing_probability)) # Clamp probabilities

    return np.random.rand() < swing_probability # Simulate probabilistic outcome

def ump_model(pitch_data):
    """
    Simplified umpire model: Determines ball, strike, or hit by pitch if no swing.
    In reality, this is based on pitch location relative to the strike zone and umpire tendencies.
    Could be rule-based (as simplified here) or a model trained on called pitches.
    """
    plate_x = pitch_data['plate_x']
    plate_z = pitch_data['plate_z']
    sz_top = pitch_data['sz_top']
    sz_bot = pitch_data['sz_bot']
    p_throws = pitch_data['p_throws'] # pitcher throws

    # Simplified rule-based strike zone (ignoring umpire variability for simplicity)
    if plate_z >= sz_bot and plate_z <= sz_top and plate_x >= -0.83 and plate_x <= 0.83: # Rough plate width est.
        return "strike_called"
    else:
        return "ball" # Default to ball outside strike zone (can be more nuanced)

def contact_model(pitch_data):
    """
    Simplified contact model: Predicts whiff, foul, or batted ball if swing occurs.
    Real model would use pitch characteristics (velocity, spin), batter characteristics, etc.
    """
    # In a real model, use features to predict contact type probabilities.
    # Here, we'll use simple probabilities based on a general assumption.

    whiff_prob = 0.25 # Example: 25% chance of whiff on a swing
    foul_prob = 0.35  # Example: 35% chance of foul ball
    # Remaining probability is for batted ball

    rand_val = np.random.rand()
    if rand_val < whiff_prob:
        return "whiff"
    elif rand_val < whiff_prob + foul_prob:
        return "foul"
    else:
        return "batted_ball"

def batted_ball_specs_model(pitch_data):
    """
    Simplified batted ball specs model: Predicts exit velocity and launch angle.
    Real model would use pitch characteristics, contact location, batter attributes.
    This is a regression problem (predicting continuous values).
    """
    # Placeholder - replace with a model predicting exit_velocity and launch_angle
    # based on relevant features.

    release_speed = pitch_data['release_speed']

    # Very basic, placeholder relationships (adjust based on real data/models)
    predicted_exit_velocity = release_speed * 2 + np.random.normal(0, 5) # Example linear relationship + noise
    predicted_launch_angle = np.random.normal(15, 15) # Example normal distribution around 15 degrees

    predicted_exit_velocity = max(60, min(120, predicted_exit_velocity)) # Clamp to reasonable ranges
    predicted_launch_angle = max(-20, min(60, predicted_launch_angle)) # Clamp launch angle


    return {"exit_velocity": predicted_exit_velocity, "launch_angle": predicted_launch_angle}

def batted_ball_outcome_model(pitch_data, batted_ball_specs):
    """
    Simplified batted ball outcome model: Predicts batted ball result (out, hit).
    Real model would use exit velocity, launch angle, spray angle, fielder positioning, etc.
    This is a classification problem (predicting categories).
    """
    exit_velocity = batted_ball_specs["exit_velocity"]
    launch_angle = batted_ball_specs["launch_angle"]

    # Very simplified outcome logic based on exit velocity and launch angle ranges
    if launch_angle < 10:
        if exit_velocity < 95:
            return "ground_out"
        else:
            return "low_line_drive_out" # Or could be single if lucky
    elif 10 <= launch_angle <= 25:
        if exit_velocity < 90:
            return "line_drive_out" # or single
        elif exit_velocity < 100:
            return "single" # or double
        else:
            return "double" # or triple
    elif 25 < launch_angle <= 40:
        if exit_velocity < 85:
            return "popup"
        elif exit_velocity < 100:
            return "single" # or double
        else:
            return "double" # or homerun
    elif launch_angle > 40:
        if exit_velocity < 90:
            return "popup"
        elif exit_velocity < 105:
            return "flyout"
        else:
            return "home_run" # Likely home run at higher exit velo

    return "out_other" # Catch-all for other outcomes



def predict_pitch_outcome(pitch_data):
    """
    Main function to predict the full pitch outcome using the series of models.
    """
    if swing_model(pitch_data):
        contact_result = contact_model(pitch_data)
        if contact_result == "whiff":
            return "strike_swinging" # Strikeout swinging
        elif contact_result == "foul":
            return "foul_ball" # Foul ball (game continues)
        elif contact_result == "batted_ball":
            batted_ball_specs = batted_ball_specs_model(pitch_data)
            outcome = batted_ball_outcome_model(pitch_data, batted_ball_specs)
            return outcome
    else:
        ump_call = ump_model(pitch_data)
        return ump_call


def get_sample_pitch_data():
    """
    Fetches sample pitch data for demonstration. You might need to adjust player IDs and date.
    """
    # Example: Get last game pitches for Jacob deGrom (player_id=594798 - you might need to look up a current player)
    # You can find player IDs using pybaseball.playerid_lookup('last_name', 'first_name')

    # You'll likely need to adjust this to get a *single* pitch row for a *recent* pitch
    # and ensure it has the necessary columns for our simplified models.

    # Example using a hardcoded pitch (replace with actual data retrieval)
    sample_pitch = {
        'plate_x': 0.1,  # Example location (feet from center plate, horizontal)
        'plate_z': 2.5,  # Example location (feet above ground, vertical)
        'sz_top': 3.5,  # Strike zone top (from pybaseball data, example value)
        'sz_bot': 1.5,  # Strike zone bottom (from pybaseball data, example value)
        'release_speed': 95, # Example release speed (MPH)
        'p_throws': 'R', # Pitcher throws right (or 'L' for left)
        'pitch_type': 'FF', # Fastball (example pitch type)
        # Add other relevant Statcast columns as needed by your real models
    }
    return pd.Series(sample_pitch) # Returning as a Pandas Series to mimic DataFrame row


if __name__ == '__main__':
    sample_pitch_data = get_sample_pitch_data()

    predicted_outcome = predict_pitch_outcome(sample_pitch_data)
    print(f"Predicted Pitch Outcome: {predicted_outcome}")

    # Example of getting and using pybaseball data (commented out as it might require setup)
    # player_info = pb.playerid_lookup('kershaw', 'clayton')
    # kershaw_id = player_info.iloc[0]['playerid']
    # recent_pitches = pb.pitching_stats_bref(2023, kershaw_id) # Replace year and player ID
    # if not recent_pitches.empty:
    #     pitch_row = recent_pitches.iloc[0] # Take the first pitch row as example
    #     predicted_outcome_from_data = predict_pitch_outcome(pitch_row)
    #     print(f"Predicted Outcome from Pybaseball Data: {predicted_outcome_from_data}")
    # else:
    #     print("Could not fetch recent pitches for the player (check player ID and year)")