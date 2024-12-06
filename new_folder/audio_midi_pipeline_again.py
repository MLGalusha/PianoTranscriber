import pretty_midi
import pandas as pd
import numpy as np
import librosa

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

def create_extended_piano_roll(df, time_df):
    """
    Creates an extended piano roll with binary columns for note on/off states.
    """
    num_times = len(time_df)
    num_pitches = 88  # 88 piano keys (MIDI notes 21 to 108)
    
    # Initialize piano roll with zeros
    piano_roll = np.zeros((num_times, num_pitches), dtype=int)
    
    # Create additional DataFrames for on/off states
    on_states = np.zeros((num_times, num_pitches), dtype=int)
    off_states = np.zeros((num_times, num_pitches), dtype=int)

    # Iterate over each note in the MIDI dataframe
    for _, row in df.iterrows():
        start_seconds = row['start_time']
        end_seconds = row['end_time']
        pitch = row['pitch']
        pitch_index = int(pitch) - 21  # MIDI note 21 corresponds to index 0

        if 0 <= pitch_index < num_pitches:
            # Find indices where the note is active
            # Create masks for seconds
            start_mask = (time_df['seconds'] >= start_seconds) & (time_df['seconds'] < end_seconds)
            start_indices = time_df[start_mask].index

            # Mark note as active in piano roll
            piano_roll[start_indices, pitch_index] = 1
            
            # Mark note on and off states
            if len(start_indices) > 0:
                on_states[start_indices[0], pitch_index] = 1
                off_states[start_indices[-1], pitch_index] = 1
        else:
            print(f"Pitch {pitch} is out of piano range.")

    return piano_roll, on_states, off_states

def process_files(location):
    """
    Main function to process a MIDI file and return an extended DataFrame.
    """
    # Process MIDI data
    df = midi_to_df(location)
    
    # Generate spectrogram
    duration, spectrogram = generate_spectrogram(location)
    
    # Generate time DataFrame with step size based on spectrogram
    time_df = generate_time_df(duration, duration/len(spectrogram[0]))
    
    # Create extended piano roll
    piano_roll, on_states, off_states = create_extended_piano_roll(df, time_df)
    
    # Prepare spectrogram DataFrame
    spectrogram_df = pd.DataFrame(spectrogram.T)
    
    # Create column lists with interleaved order
    final_columns = []
    
    # First, add all spectrogram columns
    for i in range(spectrogram_df.shape[1]):
        final_columns.append(spectrogram_df.iloc[:, i])
    
    # Then interleave pitch, pitch_on, and pitch_off
    for pitch in range(21, 109):
        pitch_index = pitch - 21
        
        # Pitch column
        pitch_col = pd.Series(piano_roll[:, pitch_index], name=f'pitch_{pitch}')
        final_columns.append(pitch_col)
        
        # Pitch on column
        on_col = pd.Series(on_states[:, pitch_index], name=f'pitch_{pitch}_on')
        final_columns.append(on_col)
        
        # Pitch off column
        off_col = pd.Series(off_states[:, pitch_index], name=f'pitch_{pitch}_off')
        final_columns.append(off_col)
    
    # Create final DataFrame with the interleaved columns
    final_df = pd.concat(final_columns, axis=1)
    
    return final_df

# Example usage
# result = process_files('/path/to/your/file')