# YouLockbutIFlip

A comprehensive audio analysis and classification system for automotive sound detection.

## Project Structure

This repository is organized into the following directories:

### ClipTool

The ClipTool directory contains tools for audio processing, segmentation, and labeling:

- `main.py`: A GUI application for visualizing audio spectrograms and creating labeled segments
- `clip.py`: Core functionality for splitting audio files into smaller segments based on markers
- `preprocess.py`: Preprocessing utilities for audio data
- `requirements.txt`: Python dependencies for the ClipTool

#### Templates

- Contains HTML templates for the web interface (`index.html`)

### Code

The Code directory contains the machine learning and classification components:

- `classification.py`: Implementation of audio classification algorithms
- `cnn.py`: Convolutional Neural Network model for audio classification
- `requirements.txt`: Python dependencies for the classification code

### Model

The Model directory stores trained machine learning models:

- `model.pth`: Trained PyTorch model for audio classification

### Demo

The Demo directory contains demonstration videos showing the system in action:

- Various masked videos (`audi_01_mask.mp4`, `audi_02_mask.mp4`, etc.) demonstrating the system's capabilities

### Paper

The Paper directory contains research documentation:

- `manuscript.pdf`: Research paper documenting the methodology and results

## Getting Started

### ClipTool

1. Install dependencies:
   ```
   cd ClipTool
   pip install -r requirements.txt
   ```

2. Run the GUI application:
   ```
   python main.py
   ```


### Classification

1. Install dependencies:
   ```
   cd Code
   pip install -r requirements.txt
   ```

2. Run classification:
   ```
   python classification.py
   ```

## Features

- Audio visualization and segmentation with interactive GUI
- Automated audio clip extraction based on markers
- CNN-based classification of automotive sounds
- Web interface for audio processing

## Requirements

- Python 3.6+
- PyTorch
- Librosa
- Matplotlib
- Tkinter (for GUI)
- Flask (for web interface)

## License

This project is proprietary and confidential.
