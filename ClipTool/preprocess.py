import librosa
import soundfile as sf
import os
import numpy as np

def split_audio(input_file, output_dir=None, segment_duration=60):
    """
    Split an audio file into segments of specified duration if it exceeds that duration.
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str, optional): Directory to save the segments. If None, uses same directory as input
        segment_duration (int): Duration of each segment in seconds (default: 60)
        
    Returns:
        list: List of paths to the generated segment files
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Calculate duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)
    
    # If duration is less than or equal to segment_duration, return original file
    if duration <= segment_duration:
        return [input_file]
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of samples per segment
    samples_per_segment = int(segment_duration * sr)
    
    # Split the audio into segments
    segments = []
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    for i, start_idx in enumerate(range(0, len(y), samples_per_segment)):
        segment = y[start_idx:start_idx + samples_per_segment]
        
        # If this is the last segment and it's too short, pad with silence
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)))
        
        # Generate output filename
        output_file = os.path.join(output_dir, f"{base_filename}_segment_{i+1}.wav")
        
        # Save the segment
        sf.write(output_file, segment, sr)
        segments.append(output_file)
    
    return segments

if __name__ == "__main__":
    # Example usage
    input_path = 'newvoice/'
    input_file = "0104Q5_6.wav"
    output_dir = "newvoice/"
    segments = split_audio(input_path+input_file, output_dir=output_dir)
    print(f"Segments saved to: {output_dir}")