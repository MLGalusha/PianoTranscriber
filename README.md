# PianoTranscriber

PianoTranscriber is a project designed to bridge the gap between live piano performances and their digital transcription into MIDI files, with the ultimate goal of generating sheet music. This project combines audio processing, machine learning, and music knowledge to create a tool that can analyze raw piano audio, figure out the notes being played, and produce a digital MIDI file. Here’s an explanation of the project, its parts, and its purpose.

### What is PianoTranscriber?

PianoTranscriber is a deep learning tool that converts piano recordings into a MIDI format using audio-to-MIDI transcription. The goal is to make it easier for musicians to record their live performances or compositions and turn them into something they can refine or replay. It's especially useful for creating sheet music from recordings or analyzing compositions without needing to do it manually.

At the heart of the project is a Convolutional Neural Network (CNN) that analyzes spectrograms of piano audio and predicts which notes are being played at specific times. The predictions are saved as MIDI files, which can later be turned into sheet music using other tools.

### Why is this project valuable?

Turning audio into MIDI and sheet music has a lot of potential uses:

1. **For Musicians:** Helps players document and improve improvisations or compositions without having to write them out manually.
2. **For Educators:** Makes it easier to break down and understand complex piano pieces.
3. **For Learners:** Lets students see the notes and timing of a performance for practice or study.
4. **For Researchers:** Shows how machine learning can handle complex audio data in real time.

### How does it work?

The project is broken into a few key parts:

#### Dataset

PianoTranscriber uses the MAESTRO dataset, which is a collection of paired piano audio and MIDI recordings. It was made using Yamaha Disklavier pianos, which can capture MIDI data very precisely during live performances. The dataset includes about 200 hours of classical piano music from competitions, making it a great source for training the model.

#### Data Preprocessing

Before data is sent into the model, both the audio and MIDI files need some prep work:

- **Audio Processing:** Convert audio recordings into spectrograms (visual representations of sound over time). Adjust the spectrograms so each row represents about 11 milliseconds of time.
- **MIDI Processing:** Pull the timing and pitch information from the MIDI files. Match the MIDI data to the spectrogram rows to create a "piano roll," which is like a binary grid where each column is a piano key and each row is a time step.

This preprocessing ensures the model gets properly formatted inputs (spectrograms) and outputs (piano rolls) for training.

#### Model Architecture

The core of the project is a CNN that predicts which notes are active at any moment based on the spectrogram input. The model:

- Uses convolutional layers to find patterns in the spectrogram data.
- Outputs probabilities for all 88 piano keys, showing whether a note is active at a specific time slice.

The model is trained to minimize the difference between its predictions and the actual piano roll data.

#### Training Process

Because the dataset is large (~100GB), the data is handled in smaller batches:

- The dataset is divided into smaller preprocessed files.
- Files are loaded one at a time into memory and passed through the model in batches.
- The model learns to match spectrogram patterns to piano notes, adjusting its parameters to improve accuracy.

Training runs on Google Cloud’s virtual machines, which have the power needed for large-scale data processing.

#### Output Generation

Once trained, the model can predict notes from new audio recordings. The workflow looks like this:

1. **Spectrogram Creation:** Turn the audio file into a spectrogram.
2. **Model Prediction:** Pass the spectrogram through the trained model to predict notes for each time slice.
3. **MIDI Conversion:** Convert the predictions into a MIDI file that represents the timing and pitch of notes.
4. **Sheet Music Generation:** Use tools like MuseScore to create readable sheet music from the MIDI file.

### What makes this project challenging?

A few things made this project tricky:

1. **Data Size:** The dataset’s size required careful memory management and efficient batch processing.
2. **Accuracy:** The model needed to handle overlapping notes and changes in dynamics accurately.
3. **Training Environment:** Setting up and troubleshooting the Google Cloud virtual machine for training was a challenge.
4. **Integration:** Aligning audio and MIDI data with millisecond precision was key for good predictions.

### What has been achieved?

- **MIDI Output:** The model can successfully turn piano audio into MIDI files.
- **Pipeline Automation:** The preprocessing and training pipelines are automated, making it easier to repeat the process with new data.
- **Cloud Integration:** The project uses cloud computing to handle the heavy lifting during training.

### Future Work

While the project shows that audio-to-MIDI transcription works, there’s still room to grow:

1. **Enhanced Accuracy:** Fine-tune the model and train it on more data to improve performance.
2. **Sheet Music Integration:** Build a more efficient pipeline to go straight from audio to sheet music.
3. **Real-Time Transcription:** Optimize the system for real-time use, like during live performances.
4. **User Interface:** Create an app that’s easy for musicians and educators to use.
5. **Better Labeled Data:** Add more details like tempo, dynamics, and other features from the music to make the transcription more useful.
6. **More Features:** Include things like chord detection or key changes to give a more complete picture of the performance.

### Conclusion

PianoTranscriber is a step forward in leveraging AI to simplify music transcription. By combining machine learning techniques with high-quality datasets, it opens up new ways for musicians, students, and researchers to work with music. While there is still work to be done, the progress achieved demonstrates the potential of AI in the field of music technology.
