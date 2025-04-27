import pyaudio
import numpy as np
import torch
import queue
import threading
import time
from scipy import signal
from cnn import AudioCNN
import os


import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)


RelayPin = 4

waittime=4.2




# Global model initialization
saved_model_path = 'audio_cnn_model.pth'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') 
model = AudioCNN(2, [18, 94]).to(device)  # Using known dimensions
model.load_state_dict(torch.load(saved_model_path))
model.eval()


def makerobo_loop():
    GPIO.output(RelayPin, GPIO.HIGH)
    time.sleep(1)                      #wait for 1s
    GPIO.output(RelayPin, GPIO.LOW)
    GPIO.cleanup() 
    
# clean up the GPIO settings
#def makerobo_destroy():
    #GPIO.output(RelayPin, GPIO.LOW) # close relay
    #GPIO.cleanup() 



class AudioProcessor:
    def __init__(self):
        # audio parameter setting
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = int(0.2 * self.RATE)  # 0.2s data
        self.WINDOW_SIZE = self.RATE  # 1s data
        
        # save recordings
        self.save_dir = "recorded_chunks"
        os.makedirs(self.save_dir, exist_ok=True)
        self.chunk_counter = 0
        
        # initilize buffer and queue
        self.audio_buffer = np.zeros(self.WINDOW_SIZE, dtype=np.float32)
        self.data_queue = queue.Queue()
        
        # execution control flag
        self.is_running = False

        # initialize the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioCNN(2, [18, 94]).to(self.device)
        self.model.load_state_dict(torch.load('audio_cnn_model.pth'))
        self.model.eval()

    def record_audio(self):
        """Function of recording thread"""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording start...")
        
        while self.is_running:
            try:
                # read audios
                data = stream.read(self.CHUNK)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # put data into queues
                self.data_queue.put(audio_data)
            except Exception as e:
                print(f"recording error: {str(e)}")
                break

        # audio terminate
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("ending recording")
        
        
  
    # Ensure the audio data is in floating-point format
    #if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
    #    audio_data = audio_data.astype(np.float32)


    def process_audio(self):
        """function of audio processing thread"""
        print("start audio processing...")
        
        while self.is_running:
            if not self.data_queue.empty():
                try:
                    # get new audio data
                    new_data = self.data_queue.get()
                    
                    # update sliding window
                    self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
                    self.audio_buffer[-len(new_data):] = new_data
                    
                    # audio analysis
                    self.analyze_audio(self.audio_buffer.copy())
                except Exception as e:
                    print(f"processing audio error: {str(e)}")
            else:
                # a short delay to reduce CPU load
                time.sleep(0.01)
                
                
                
                

    def save_audio_chunk(self, audio_data, prediction):
        """recording saving"""
        
            
        try:
            # generating file names including time stamp and prediction results
            
            timestamp = int(time.time())
            label = "positive" if prediction == 1 else "negative"
            filename = f"chunk_{timestamp}_{self.chunk_counter}_{label}.wav"
            filepath = os.path.join(self.save_dir, filename)
            
            
            # convert float32 data into int16
            audio_data_int = (audio_data * 32767).astype(np.int16)
            
            # save to .wav files
            import wave
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data_int.tobytes())
            
            self.chunk_counter += 1
            print(f"saving file clips: {filename}")
            
        except Exception as e:
            print(f"audio saving error: {str(e)}")

    def analyze_audio(self, audio_data):
        """audio data analying"""
        try:
            # 1. ensure correct data type
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
        
        
            # 2. spectrogram
            spectrum = self.to_spectrum(audio_data)
            
            # 3. model input
            input_data = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 4. model output
            with torch.no_grad():
                output = self.model(input_data)
                prediction = int(torch.argmax(output))
                confidence = torch.softmax(output, dim=1)[0]
                
                # result printing
                if prediction == 1:
                    print(f"Target detected! (Confidence: {confidence[1]:.2f})")
                    time.sleep(waittime)
                    makerobo_loop()
                    print(f"Total Chunk Number is {self.chunk_counter}.")
                    exit()
                else:
                    print(f"noise (Confidence: {confidence[0]:.2f})")
                
                # save audio recording
                self.save_audio_chunk(audio_data, prediction)
                
        except Exception as e:
            print(f"audio analysis error: {str(e)}")

    def to_spectrum(self, audio_data):
        """convert audio data to spectrogram"""
        # STFT
        n_fft = 2048
        hop_length = 512
        
        # spectrogram
        f, t, Zxx = signal.stft(audio_data, fs=self.RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
        
        
        spectrogram = np.abs(Zxx)
        
        # convert to dB
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # find the index: 3600-4000 Hz
        freq_mask = (f >= 3600) & (f <= 4000)
        spectrogram_filtered = spectrogram_db[freq_mask]
        
        # adjust the size to (18, 94)  
        from scipy.ndimage import zoom
        zoom_factors = (18 / spectrogram_filtered.shape[0],
                       94 / spectrogram_filtered.shape[1])
        resized_spectrogram = zoom(spectrogram_filtered, zoom_factors)
        
        return resized_spectrogram

    def start(self):
        """starting"""
        self.is_running = True
        
        # recording thread start
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()
        
        # processing thread start
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

    def stop(self):
        """stop processing"""
        self.is_running = False
        
        # waiting for thread ending
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        print("The program has been terminated")

def main():
    processor = AudioProcessor()
    processor.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nProgram is ending...")
        processor.stop()

if __name__ == "__main__":
    main()
