import pandas as pd
import pretty_midi
import librosa
import numpy as np


def midi_to_df(location):
    """
    Converts a MIDI file into a DataFrame with start_time, end_time, and pitch columns.
    """
    midi_data = pretty_midi.PrettyMIDI(f"{location}midi")
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
    # Load audio
    y, sr = librosa.load(f"{file_path}wav", sr=44100)

    duration = librosa.get_duration(y=y, sr=sr)

    # Generate spectrogram
    n_fft = 1024
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(S)
    return [duration, spectrogram]

def generate_time_df(duration, step):
    """
    Generates a DataFrame with seconds column, covering the entire duration of the song.
    """
    time_list = []

    # Generate the list using numpy.arange
    numbers = np.arange(0, duration, step)  # Add step to include the endpoint if possible
    for number in numbers:
        time_list.append({"seconds": number})

    time_df = pd.DataFrame(time_list)
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
            # Create masks for seconds
            seconds_mask = (time_df['seconds'] >= start_seconds) & (time_df['seconds'] < end_seconds)
            active_indices = time_df[seconds_mask].index

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

    return piano_roll_df

def process_files(location):
    """
    Main function to process a MIDI file and return a DataFrame indicating active pitches over time.
    """
    df = midi_to_df(location)
    duration, spectrogram = generate_spectrogram(location)
    time_df = generate_time_df(duration, duration/len(spectrogram[0]))
    piano_roll = create_piano_roll(df, time_df)
    piano_roll_df = create_piano_roll_df(piano_roll, time_df)
    spectrogram_df = pd.DataFrame(spectrogram.T)
    final_df = pd.concat([spectrogram_df, piano_roll_df], axis=1)
    return final_df