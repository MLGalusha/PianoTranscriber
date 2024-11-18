# MusicTranscriber

An AI-powered tool for transcribing solo piano performances from MP3 files into MIDI and sheet music using deep learning.

## Overview

**MusicTranscriber** aims to convert raw audio recordings of solo piano performances into accurate MIDI files and corresponding sheet music. By leveraging the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) and deep learning techniques, we focus on capturing the nuances of piano music—including notes, chords, timing, and dynamics.

## Objectives

- **Accurate Transcription**: Convert MP3 audio files of piano performances into MIDI files with high fidelity.
- **Polyphonic Handling**: Effectively transcribe multiple notes played simultaneously (chords).
- **Dynamic Expression**: Capture the velocity (loudness) and articulation of each note.
- **Sheet Music Generation**: Translate MIDI files into readable sheet music for musicians.
- **Model Robustness**: Enhance the model's performance through data augmentation and architectural improvements.

## Data Source

We are utilizing the **MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) dataset**, which comprises approximately 200 hours of virtuosic piano performances with fine alignment (~3 ms) between note labels and audio waveforms.

- **Content**: Paired audio and MIDI recordings from ten years of International Piano-e-Competition.
- **Features**: Key strike velocities, sustain/sostenuto/una corda pedal positions, and detailed metadata.
- **Quality**: Uncompressed audio of CD quality or higher (44.1–48 kHz 16-bit PCM stereo).
- **License**: Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0).

---

## Planned Workflow

### 1. Data Preparation

- **Download & Organize Data**:

  - Obtain MP3 audio files and MIDI files from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro).
  - Ensure data integrity and proper alignment between audio and MIDI files.

- **Segment Tracks**:
  - Normalize all tracks to 1-minute segments.
    - Pad shorter tracks with silence.
    - Truncate longer tracks to fit the desired length.
  - Optionally, use overlapping segments (e.g., 50% overlap) to capture more context.

### 2. Feature Extraction

- **From MIDI Files (Labels)**:

  - Extract note-level features:
    - **Pitch**: Specific notes being played.
    - **Onset Time**: When each note starts.
    - **Offset Time**: When each note ends.
    - **Velocity**: Loudness of each note.
    - **Duration**: How long each note is held.
  - Compile these into a structured format for model training.

- **From MP3 Files (Input)**:
  - Convert MP3 files into **Mel-spectrograms** using `librosa` or a similar library.
    - Provides a time-frequency representation of the audio.
  - Normalize spectrograms for consistent input.

### 3. Model Development

- **Start Simple (Pitch, Onset & Offset Detection)**:

  - **Input**: Mel-spectrograms.
  - **Output**:
    - Predict which notes are playing (pitch detection).
    - Predict when each note starts (onset detection).
    - Predict when each note ends (offset detection).

- **Model Architecture**:
  1. **Convolutional Neural Network (CNN) Layers**:
     - Extract time-frequency patterns from spectrograms.
  2. **Recurrent Neural Network (RNN) Layers (LSTM/GRU)**:
     - Capture temporal dependencies between notes.
  3. **Output Layers**:
     - **Pitch Detection**: Binary vector for active notes per time frame.
     - **Onset Detection**: Binary vector indicating note starts.
     - **Offset Detection**: Binary vector indicating note ends.

### 4. Training Strategy

- **Loss Functions**:

  - **Pitch Detection**: Binary cross-entropy loss.
  - **Onset Detection**: Binary cross-entropy loss.
  - **Offset Detection**: Binary cross-entropy loss.

- **Optimization**:
  - Use optimizers like Adam or RMSprop.
  - Implement learning rate scheduling and early stopping.

### 5. Gradual Complexity Increase

- **Incorporate Velocity and Dynamics**:

  - Predict the velocity (loudness) of each note.
  - Add multi-output layers to handle additional features like dynamics and articulation.

- **Handling Chords and Polyphony**:

  - Ensure the model effectively transcribes multiple notes played simultaneously.
  - Use advanced architectures known for handling polyphonic music.

- **Data Augmentation with Digital Pianos**:
  - Record custom datasets using digital pianos to include complex chords and sequences.
  - Synchronize audio and MIDI data for accurate labeling.

### 6. Evaluation & Metrics

- **Quantitative Metrics**:

  - **Pitch Detection**: Accuracy and F1-score.
  - **Onset Detection**: Precision, recall, and F1-score.
  - **Offset Detection**: Evaluate timing accuracy.
  - **Chord Recognition**: Specific metrics for chord transcription accuracy.

- **Transcription Quality**:
  - **Edit Distance/BLEU Score**: Compare predicted MIDI files with ground truth.
  - **Subjective Evaluation**: Have pianists assess the playability and correctness of transcriptions.

### 7. Optimization & Augmentation

- **Data Augmentation Techniques**:

  - **Pitch Shifting**: To generalize across different keys.
  - **Time Stretching**: To handle variations in tempo.
  - **Dynamic Range Augmentation**: Vary note velocities.
  - **Noise Injection**: Add background noise for robustness.

- **Model Regularization**:

  - Use dropout layers and L2 regularization to prevent overfitting.
  - Implement batch normalization.

- **Batch Augmentation**:
  - Apply augmentations to each mini-batch during training for variety.

### 8. Computational Resources

- **Cloud Computing with Google Cloud**:
  - Set up virtual machines with appropriate specifications.
  - Monitor resource usage to stay within free trial limits.
  - Optimize code to reduce computational load.

### 9. Future Considerations

- **Genre Expansion**:

  - Incorporate music from genres beyond classical to improve generalization.

- **Instrument Expansion**:

  - Adapt the model for other instruments like guitar, considering their unique notation systems.

- **User Interface Development**:

  - Develop a simple application or web interface for users to input MP3 files and receive sheet music.

- **Incorporate Music Theory**:
  - Embed music theory rules to enhance prediction accuracy, especially for unseen chords.

---

## Additional Notes

- **Collaboration Tools**:

  - Use GitHub issues and pull requests for code collaboration.
  - Schedule regular meetings for theory discussion and progress updates.

- **Challenges to Address**:

  - **Chord Complexity**: Due to the vast number of possible chords, ensure sufficient training data and model capacity.
  - **Data Volume**: Manage the large size of the MAESTRO dataset effectively.
  - **Computational Limitations**: Optimize algorithms to run efficiently on available hardware.

- **Team Goals**:
  - Divide tasks based on expertise (data preparation, model development, evaluation).
  - Encourage knowledge sharing, especially in areas like music theory and machine learning techniques.

---

## Getting Started

- **Prerequisites**:

  - Python 3.x environment.
  - Libraries: TensorFlow or PyTorch, `librosa`, NumPy, pandas, etc.
  - Sufficient storage for the dataset (~120 GB uncompressed).

- **Setup Instructions**:
  1. Clone the repository: `git clone https://github.com/YourUsername/MusicTranscriber.git`
  2. Download the MAESTRO dataset and place it in the designated data folder.
  3. Install required packages: `pip install -r requirements.txt`
  4. Run initial data processing scripts.

---

## References

- **MAESTRO Dataset**: [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)
- **Related Research**:
  - [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://openreview.net/forum?id=r1lYRjC9F7)
  - [Wave2Midi2Wave Blog Post](https://magenta.tensorflow.org/maestro-wave2midi2wave)

---

_Let's collaborate to make MusicTranscriber a cutting-edge solution for music transcription!_
