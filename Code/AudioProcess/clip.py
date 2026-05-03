import os
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def load_json_labels(json_path):
    """Load labels from a JSON file"""
    with open(json_path, 'r') as f:
        labels = json.load(f)
    return labels

def get_label_for_segment(start_time, end_time, labels):
    """
    Check the labeling status of an audio segment:
    - If it overlaps with or is completely within a label, return that label value
    - If it's completely outside any label, return '0'
    """
    # Check if the segment overlaps with or is completely within any label
    for label in labels:
        # If the segment is completely within the label
        if start_time >= label['start_time'] and end_time <= label['end_time']:
            return label['name'].split(':')[1]
        # If the segment overlaps with the label (not completely before or after)
        elif not (end_time <= label['start_time'] or start_time >= label['end_time']):
            return label['name'].split(':')[1]

    # If it doesn't overlap with any label, return '0'
    return '0'

def process_audio_file(wav_path, json_path, output_dir):
    """Process a single audio file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    y, sr = librosa.load(wav_path, sr=None)
    
    # Load labels
    labels = load_json_labels(json_path)
    
    # Calculate number of samples for each segment (0.2 seconds)
    segment_samples = int(0.2 * sr)
    
    # Track index for each segment
    segment_index = 0
    
    # Split audio
    for i in range(0, len(y), segment_samples):
        # Get current segment
        segment = y[i:i + segment_samples]
        
        # Skip if the last segment is incomplete
        if len(segment) < segment_samples:
            continue
        
        # Calculate time range of the segment
        start_time = i / sr
        end_time = (i + segment_samples) / sr
        
        # Get label
        label = get_label_for_segment(start_time, end_time, labels)
        
        # Construct output filename
        base_name = Path(wav_path).stem
        output_filename = f"{base_name}_{segment_index}_{label}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save audio segment
        sf.write(output_path, segment, sr)
        
        # Increment segment index
        segment_index += 1

def main():
    # Set input and output directories
    input_dir = "newvoice"
    output_dir = "clips"
    
    # Get all wav files, ignore files starting with .
    wav_files = [f for f in os.listdir(input_dir) 
                if f.endswith('.wav') 
                and not f.startswith('.')]
    
    # Process each wav file
    for wav_file in wav_files:
        wav_path = os.path.join(input_dir, wav_file)
        json_file = wav_file.replace('.wav', '.json')
        json_path = os.path.join(input_dir, json_file)
        
        # Check if the corresponding json file exists and doesn't start with .
        if os.path.exists(json_path) and not os.path.basename(json_path).startswith('.'):
            print(f"Processing {wav_file}...")
            process_audio_file(wav_path, json_path, output_dir)
            print(f"Finished processing {wav_file}")

if __name__ == "__main__":
    main()
