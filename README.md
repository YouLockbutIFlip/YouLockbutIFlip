# Do A Kick Flip

A acoustic side channel signals collectection, processing and classification pipeline for determining the precise state transition driven signal injection.


## Project Structure

This repository is organized into the following directories:

### Code

The AudioProcess directory contains tools for audio processing, segmentation, and labeling:

- `main.py`: A GUI application for visualizing audio spectrograms and creating labeled segments
- `clip.py`: Core functionality for splitting audio files into smaller segments based on markers
- `preprocess.py`: Preprocessing utilities for audio data
- `requirements.txt`: Python dependencies for the ClipTool

The Detection directory contains

- `audio_cnn_model_rpi_2ch.pth`:
- `classification_new.py`: code needed to be download onto the Raspberry Pi to classify the first chime once tailgate starting lowering
- `model_info_rpi_2ch.py`:
- `requirements.txt`: Python dependencies for the the audio classification
- 
#### Description

- Contains the description of key steps to conduct the end-to-end attack on the target Audi Q5.

### Design

The Design directory contains wiring configurations for both indoor and outdoor attack experiments:

- `IndoorOscilloscope.pdf`: wiring configurations of indoor tests. The sensing module's response to kick motion can be observed through the signals on the connected oscilloscope. 
- `gadgetsize.pdf`: all the components comprised in the end-to-end attack device and their wiring configuration
- `shieldsensortest.pdf`: bench test configuration to identify the frequencies at which the adversary can inject a kick event

### LIN BUS

The LIN BUS directory stores all datasets exported from an Audi Q5 via its LIN bus connected to the kick sensing module and the corresponding waveforms caputred from an connectedd oscilloscope. 
-'box no key' refers to the signal is triggered by the portable attack device while vehicle's keyfob is not within the effective range
-'box with key' refers to the signal is triggered by the portable attack device while vehicle's keyfob is within the effective range
-'kick no key' refers to the signal is triggered by an kick motion while vehicle's keyfob is not within the effective range
-'kick with key' refers to the signal is triggered by an kick motion while vehicle's keyfob is within the effective range

### Demo

The Demo directory contains demonstration videos showing the system in action:

- Various masked videos (`audi_01_mask.mp4`, `audi_02_mask.mp4`, etc.) demonstrating the system's capabilities

### Module Testing
The Module Testing directory contains 4 typical sensing modules: two from Brose and two from Huf. Each file contains the whole sensing module and the label printed on the control part.


### Paper

The Paper directory contains research documentation:

- `LiftgateFinal.pdf`: Research paper documenting the methodology and results


### Results
The Results directory contains one demo and test results:
- `Demo': the demo link of a comlete end-to-end attack scenario
- `End-to-End Attack Test Records': the 100 liftgate gap measurements recorded from two parking lots under different ambient noise and wind speed conditions
  
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
   python classification_new.py
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
