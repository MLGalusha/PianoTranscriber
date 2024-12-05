import os
import pandas as pd
import numpy as np
import pretty_midi
import librosa
import torch

def midi_to_df(location):
    """
    Converts a MIDI file into a DataFrame with start_time, end_time, and pitch columns.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(f"{location}midi")
    except FileNotFoundError:
        return None

    data = []

    # Loop through instruments and notes
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            data.append({
                "start_time": round(note.start, 4),
                "end_time": round(note.end, 4),
                "pitch": note.pitch,
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by="start_time", ascending=True).reset_index(drop=True)
    return df

def generate_spectrogram(file_path):

    if os.path.exists(f"{file_path}wav"):
        file_path = f"{file_path}wav"
    elif os.path.exists(f"{file_path}mp3"):
        file_path = f"{file_path}mp3"

    # Load audio
    y, sr = librosa.load(file_path, sr=44100)

    duration = librosa.get_duration(y=y, sr=sr)

    # Generate spectrogram
    n_fft = 1024
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(S)
    return duration, spectrogram

def generate_time_df(duration, step):
    """
    Generates a DataFrame with seconds column, covering the entire duration of the song.
    """
    # Generate time steps using numpy.arange
    times = np.arange(0, duration, step)

    time_df = pd.DataFrame({'seconds': times})
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
        start_seconds = row['start_time']
        end_seconds = row['end_time']
        pitch = row['pitch']
        pitch_index = int(pitch) - 21  # MIDI note 21 corresponds to index 0

        if 0 <= pitch_index < num_pitches:
            # Find indices where the note is active
            seconds_mask = (time_df['seconds'] >= start_seconds) & (time_df['seconds'] < end_seconds)
            active_indices = time_df[seconds_mask].index

            # Set the corresponding entries to 1
            piano_roll[active_indices, pitch_index] = 1
        else:
            print(f"Pitch {pitch} is out of piano range.")
    return piano_roll

def create_piano_roll_df(piano_roll):
    """
    Creates a DataFrame from the piano roll matrix with appropriate column names.
    """
    # Create column names for each pitch
    pitch_columns = [f'pitch_{pitch}' for pitch in range(21, 109)]  # MIDI notes 21 to 108

    # Create a DataFrame from the piano_roll matrix
    piano_roll_df = pd.DataFrame(piano_roll, columns=pitch_columns)

    return piano_roll_df

def process_files(location):
    """
    Main function to process a MIDI file and return a DataFrame indicating active pitches over time.
    """
    # Generate MIDI DataFrame
    df = midi_to_df(location)

    # Generate spectrogram and duration
    duration, spectrogram = generate_spectrogram(location)

    # Generate time DataFrame
    time_df = generate_time_df(duration, duration / len(spectrogram[0]))

    # Create DataFrames from spectrogram and piano roll
    spectrogram_df = pd.DataFrame(spectrogram.T)  # Transpose to align time dimension

    if not df.empty:
        # Create piano roll matrix
        piano_roll = create_piano_roll(df, time_df)

        piano_roll_df = create_piano_roll_df(piano_roll)

        # Combine spectrogram and piano roll DataFrames
        final_df = pd.concat([spectrogram_df, piano_roll_df], axis=1)

        # Identify label columns (the last 88 columns)
        label_columns = piano_roll_df.columns.tolist()

        # Remove rows where any label is not 0 or 1
        # Check if all labels in each row are either 0 or 1
        condition = final_df[label_columns].isin([0, 1]).all(axis=1)

        # Filter the DataFrame based on the condition
        final_df = final_df[condition].reset_index(drop=True)
        final_df.columns = final_df.columns.astype(str)

        # Optionally, you can print how many rows were removed
        num_removed = len(spectrogram_df) - len(final_df)
        print(f"Removed {num_removed} rows with invalid label values.")

        # Check for NaN or Inf values in the final_df
        is_bad_data = final_df.isnull().any(axis=1) | np.isinf(final_df).any(axis=1)
        bad_data = final_df[is_bad_data]
        final_df = final_df[~is_bad_data].reset_index(drop=True)

        # Fill in data for song to be divisible by 80
        zeros_df = pd.DataFrame(0, index=range(80 - (final_df.shape[0] % 80)), columns=final_df.columns)
        final_df = pd.concat([final_df, zeros_df], ignore_index=True)



        # Optionally, handle bad data (e.g., save to a file or log)
        if not bad_data.empty:
            print(f"Removed {len(bad_data)} rows containing NaN or Inf values.")
            # You can collect bad_data for further analysis if needed
            # For example, save bad data to a CSV or Parquet file
            # bad_data.to_parquet('bad_data.parquet', index=False)
        else:
            print("No NaN or Inf values found in the data.")
    else:
        final_df = spectrogram_df

    # Return the cleaned DataFrame
    return final_df

# Example usage
if __name__ == "__main__":
    # Specify the location of your MIDI and WAV files (without the extension)
    file_location = 'path/to/your/file'  # Replace with your actual file path

    # Process the files
    cleaned_df = process_files(file_location)

    # Optionally, save the cleaned DataFrame to a file
    # cleaned_df.to_parquet('cleaned_data.parquet', index=False)
