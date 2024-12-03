# PianoTranscriber

## Detailed Explanation

PianoTranscriber is a project designed to bridge the gap between live piano performances and their digital transcription into MIDI files, with the ultimate goal of generating sheet music. This endeavor combines the fields of audio processing, machine learning, and musicology to create a tool that can analyze raw piano audio, determine the notes being played, and produce a digital representation in MIDI format. Below is an in-depth explanation of the project, its components, and its purpose.

### What is PianoTranscriber?

PianoTranscriber is a deep learning application that leverages audio-to-MIDI transcription to convert piano recordings into a digital format. The ultimate aim is to simplify the process of capturing live performances or compositions, enabling musicians to refine and replay their work. It is particularly valuable for those who want to generate sheet music from their playing or analyze compositions without manually transcribing them.

At its core, the project uses a Convolutional Neural Network (CNN) to analyze spectrograms of piano audio and predict which notes are being played at specific time intervals. These predictions are stored as MIDI files, which can then be converted into sheet music using specialized tools.

### Why is this project valuable?

The ability to transcribe audio into MIDI and sheet music has multiple applications:

1. **For Musicians:** Enables players to document and refine improvisations or compositions without requiring manual transcription.
2. **For Educators:** Provides a way to analyze and understand complex piano pieces.
3. **For Learners:** Helps students visualize the notes and timing of a performance for practice and study.
4. **For Researchers:** Offers insights into how machine learning can interpret complex audio data in real time.

### How does it work?

The project can be broken down into several key components:

#### Dataset

The project relies on the MAESTRO dataset, a high-quality collection of paired piano audio and MIDI recordings. This dataset was created using Yamaha Disklavier pianos, which are capable of capturing MIDI data with millisecond-level precision during live performances. The dataset includes around 200 hours of music from classical piano competitions, providing a rich source for training and validating the model.

#### Data Preprocessing

Before feeding data into the model, the audio and MIDI files must be processed. The steps include:

- **Audio Processing:** Converting audio recordings into spectrograms, which are visual representations of sound that show frequency content over time. Transposing spectrograms so that each row corresponds to a specific moment in time (approximately 11 milliseconds per row).
- **MIDI Processing:** Extracting the timing and pitch of each note from the MIDI files. Aligning the MIDI data with the corresponding spectrogram rows to create a "piano roll," a binary matrix where each column represents a key on the piano and each row corresponds to a time interval.

This preprocessing step ensures that the model receives properly formatted input (spectrograms) and output (piano rolls) for training.

#### Model Architecture

The core of PianoTranscriber is a CNN designed to predict which notes are active at any given moment based on spectrogram input. The model:

- Processes the spectrogram data using convolutional layers to extract time-frequency patterns.
- Outputs probabilities for each of the 88 piano keys, representing whether a note is active during a given time slice.

The network is trained to minimize the difference between its predictions and the actual MIDI piano roll data.

#### Training Process

Due to the large size of the dataset (~100GB) and memory constraints, the data is processed in batches:

- The dataset is split into smaller, preprocessed files.
- These files are loaded into memory one at a time, and their data is passed through the model in batches.
- During training, the model learns to associate spectrogram patterns with the corresponding piano notes, adjusting its parameters to improve accuracy.

The training is performed on Google Cloud's virtual machines, which provide the necessary computational power to handle such large-scale data.

#### Output Generation

Once trained, the model can be used to predict the notes being played in new audio recordings. The workflow includes:

1. **Spectrogram Creation:** The input audio file is converted into a spectrogram.
2. **Model Prediction:** The spectrogram is passed through the trained model to generate predictions for each time slice.
3. **MIDI Conversion:** The predictions are converted into a MIDI file, representing the timing and pitch of the notes.
4. **Sheet Music Generation:** The MIDI file can be further processed using tools like MuseScore to produce readable sheet music.

### What makes this project challenging?

Several factors made the development of PianoTranscriber a complex task:

1. **Data Size:** The large size of the dataset required careful memory management and efficient batch processing.
2. **Accuracy:** Ensuring that the model correctly identifies overlapping notes and dynamic changes in playing.
3. **Training Environment:** Setting up and troubleshooting the Google Cloud virtual machine for model training required significant effort.
4. **Integration:** Aligning audio and MIDI data with millisecond precision was essential for accurate predictions.

### What has been achieved?

- **MIDI Output:** The model successfully converts piano audio into MIDI format.
- **Pipeline Automation:** The preprocessing and training pipelines are fully automated, making it easy to replicate the process on new data.
- **Cloud Integration:** The project leverages cloud computing to handle the computational demands of training large-scale models.

### Future Work

While the project demonstrates the feasibility of audio-to-MIDI transcription, there is room for improvement:

1. **Enhanced Accuracy:** Fine-tuning the model and training on additional data could improve performance.
2. **Sheet Music Integration:** Developing a seamless pipeline from audio to sheet music.
3. **Real-Time Transcription:** Optimizing the system for real-time applications, such as live performances.
4. **User Interface:** Creating a user-friendly app for musicians and educators.

### Conclusion

PianoTranscriber is a step forward in leveraging AI to simplify music transcription. By combining advanced machine learning techniques with high-quality datasets, it paves the way for tools that can make music more accessible to creators, learners, and enthusiasts alike. While there is still work to be done, the progress achieved demonstrates the potential of AI in the field of music technology.
