import pretty_midi
import pandas as pd
import numpy as np


def midi_to_df(location):
    """
    Converts a MIDI file into a DataFrame with minute, start_time, end_time, and pitch columns.
    """
    midi_data = pretty_midi.PrettyMIDI(location)
    data = []

    # Calculate the maximum time and total seconds
    time = max(note.end for instrument in midi_data.instruments for note in instrument.notes)
    intervals = int(np.ceil(time / 60))
    total_seconds = intervals * 60

    # Loop through instruments and notes
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Ensure note's end time does not exceed total_seconds
            start_time = note.start
            end_time = min(note.end, total_seconds)
            pitch = note.pitch

            # Only include notes within the total_seconds limit
            if start_time <= total_seconds:
                start_segment = int(start_time / 60)
                end_segment = int(end_time / 60)
                if end_time - (60 * start_segment) >= 60:
                    next_end = end_time - (60 * (start_segment + 1))
                    next_start = 0.00
                    end_time = 59.99

                    data.append({
                        "minute": start_segment,
                        "start_time": round(start_time - (60 * start_segment), 3),
                        "end_time": round(end_time, 3),
                        "pitch": pitch,
                    })

                    if intervals != start_segment + 1:
                        data.append({
                            "minute": start_segment + 1,
                            "start_time": round(next_start, 3),
                            "end_time": round(next_end, 3),
                            "pitch": pitch,
                        })
                else:
                    end_time = end_time - (60 * start_segment)

                    data.append({
                        "minute": start_segment,
                        "start_time": round(start_time - (60 * start_segment), 3),
                        "end_time": round(end_time, 3),
                        "pitch": pitch,
                    })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["minute", "start_time"], ascending=True).reset_index(drop=True)
    return df

def generate_time_df(max_minute, time_increment):
    """
    Generates a DataFrame with minute and seconds columns, covering the entire duration of the song.
    """
    time_dfs = []
    for minute in range(max_minute + 1):
        # Create 'seconds' from 0 up to 60 seconds (not including 60)
        seconds = np.arange(0, 60, time_increment)
        # Ensure 'seconds' does not exceed 59.99 seconds
        seconds = seconds[seconds < 59.99]
        # Create a DataFrame for this minute
        time_df = pd.DataFrame({
            'minute': minute,
            'seconds': seconds
        })
        time_dfs.append(time_df)
    # Combine all time DataFrames
    time_df = pd.concat(time_dfs, ignore_index=True)
    # Reset index to ensure continuous indexing
    time_df.reset_index(drop=True, inplace=True)
    return time_df

def create_piano_roll(df, time_df):
    """
    Creates a piano roll (binary matrix) indicating active pitches at each time interval.
    """
    num_times = len(time_df)
    num_pitches = 88  # 88 piano keys (MIDI notes 21 to 108)
    piano_roll = np.zeros((num_times, num_pitches), dtype=int)

    # Iterate over each note in the MIDI dataframe
    for index, row in df.iterrows():
        minute = row['minute']
        start_seconds = row['start_time']
        end_seconds = row['end_time']
        pitch = row['pitch']
        pitch_index = int(pitch) - 21  # MIDI note 21 corresponds to index 0

        if 0 <= pitch_index < num_pitches:
            # Find indices where the note is active
            # Create masks for minute and seconds
            minute_mask = time_df['minute'] == minute
            seconds_mask = (time_df['seconds'] >= start_seconds) & (time_df['seconds'] < end_seconds)
            # Combine masks
            active_indices = time_df[minute_mask & seconds_mask].index

            # Set the corresponding entries to 1
            piano_roll[active_indices, pitch_index] = 1
        else:
            print(f"Pitch {pitch} is out of piano range.")
    return piano_roll

def create_piano_roll_df(piano_roll, time_df):
    """
    Combines the piano roll matrix with time_df to create the final DataFrame.
    """
    # Create column names for each pitch
    pitch_columns = [f'pitch_{pitch}' for pitch in range(21, 109)]  # MIDI notes 21 to 108

    # Create a DataFrame from the piano_roll matrix
    piano_roll_df = pd.DataFrame(piano_roll, columns=pitch_columns)

    # Insert the 'minute' and 'seconds' columns
    piano_roll_df.insert(0, 'seconds', time_df['seconds'])
    piano_roll_df.insert(0, 'minute', time_df['minute'])

    return piano_roll_df

def process_midi_file(location):
    """
    Main function to process a MIDI file and return a DataFrame indicating active pitches over time.
    """
    df = midi_to_df(location)
    max_minute = df['minute'].max()
    time_increment = 0.0116099071207  # Approximately 11 milliseconds
    time_df = generate_time_df(max_minute, time_increment)
    piano_roll = create_piano_roll(df, time_df)
    piano_roll_df = create_piano_roll_df(piano_roll, time_df)
    return piano_roll_df