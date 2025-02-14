# Music Generation Using LSTM

## Overview
This project utilizes Long Short-Term Memory (LSTM) neural networks to generate MIDI music. The model is trained on a dataset of MIDI files, learns patterns in musical notes and chords, and generates new compositions.

## Features
- Reads and processes MIDI files
- Extracts musical notes and chords
- Filters notes based on frequency
- Creates input sequences for LSTM training
- Builds an LSTM-based neural network for music generation
- Generates new musical sequences based on trained patterns
- Saves and downloads generated MIDI files

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install music21 numpy tqdm tensorflow scikit-learn
```

## Dataset Preparation
1. Mount Google Drive to access MIDI files:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Load MIDI files from the specified directory:
   ```python
   import glob
   all_files = glob.glob('/content/drive/MyDrive/All Midi Files/**/*.mid', recursive=True)
   ```

## Processing MIDI Files
- Convert MIDI files into a sequence of notes and chords.
- Store unique notes and their frequency counts.
- Filter notes appearing at least five times to reduce noise.

## Model Training
- Uses LSTM layers with dropout for sequence modeling.
- Categorical cross-entropy loss with Adam optimizer.
- Splits dataset into 80% training and 20% testing.
- Trains for 120 epochs with a batch size of 128.

```python
model.fit(x_train, y_train, batch_size=128, epochs=120, validation_data=(x_test, y_test))
```

## Generating Music
- The model predicts sequences of 100 notes.
- Converts predictions into a MIDI file.
- Saves and downloads the generated composition.

```python
midi_stream.write('midi', fp='/content/drive/MyDrive/test_output.mid')
files.download('/content/drive/MyDrive/test_output.mid')
```

## Output
- A generated MIDI file containing an AI-composed musical piece.
- The composition follows learned patterns from the training dataset.

## Future Improvements
- Expand training dataset for better diversity.
- Fine-tune model hyperparameters for improved accuracy.
- Implement more complex neural architectures.
- Explore different musical representations for better results.

