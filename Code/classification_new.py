"""
Gate Beep Classifier — Raspberry Pi Edition
============================================
Updated for the 16 kHz / 2-channel / binary model trained with
train_model_rpi_2ch.py.

Sliding-window detection:
  - Window : 200 ms = 3200 samples @ 16 kHz
  - Step   :  50 ms =  800 samples @ 16 kHz  (75% overlap)
  - Confirm:  3 consecutive windows classified as Beep → real beep
  - Cooldown: 500 ms after confirmed beep (prevents re-trigger)

Usage (run on Raspberry Pi from the directory containing the model files):
    python classification_rpi.py
"""

import os
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import torch
import torch.nn as nn
from scipy import signal as scipy_signal
from scipy.ndimage import zoom

# ── GPIO ──────────────────────────────────────────────────────────────────────
import RPi.GPIO as GPIO

RELAY_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)


def trigger_relay():
    """Pulse relay HIGH for 1 s then clean up."""
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()


# ── Audio parameters (must match train_model_rpi_2ch.py) ─────────────────────
RATE          = 16_000
WINDOW_SIZE   = int(RATE * 0.200)    # 3200 samples — model input window
STEP_SIZE     = int(RATE * 0.050)    # 800  samples — slide step (50 ms)

N_FFT      = 512
HOP        = 85
BAND_0     = (2000.0, 2600.0)        # primary tone   ~2280 Hz
BAND_1     = (3500.0, 4200.0)        # secondary tone ~3800 Hz
SPEC_H     = 32
SPEC_W     = 32
N_CHANNELS = 2

# ── Detection parameters ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50          # beep probability threshold
CONFIRM_N            = 3             # consecutive beep windows to confirm
COOLDOWN_STEPS       = int(0.500 / 0.050)   # 10 steps = 500 ms cooldown
WAIT_TIME            = 3             # seconds to wait before triggering relay

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH      = 'audio_cnn_model_rpi_2ch.pth'
MODEL_INFO_PATH = 'model_info_rpi_2ch.pth'


# ── CNN Architecture (must match train_model_rpi.py exactly) ──────────────────
class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super().__init__()
        self.conv1   = nn.Conv2d(input_shape[0], 32,  kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1     = nn.BatchNorm2d(32)
        self.bn2     = nn.BatchNorm2d(64)
        self.bn3     = nn.BatchNorm2d(128)

        fc_in = self._conv_output_size(input_shape)
        self.fc1 = nn.Linear(fc_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        return self._forward_conv(x).view(1, -1).size(1)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)


# ── Spectrogram conversion ────────────────────────────────────────────────────
def _band_to_channel(spec_db, freqs, f_lo, f_hi):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    band = spec_db[mask] if mask.any() else spec_db
    h, w = band.shape
    return zoom(band, (SPEC_H / h, SPEC_W / w), order=1).astype(np.float32)


def to_spectrogram(audio: np.ndarray) -> np.ndarray:
    """200 ms audio → 2-channel (2, 32, 32) spectrogram matching train_model_rpi_2ch.py."""
    f, _, Zxx = scipy_signal.stft(audio, fs=RATE, nperseg=N_FFT,
                                   noverlap=N_FFT - HOP)
    spec_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    ch0 = _band_to_channel(spec_db, f, *BAND_0)
    ch1 = _band_to_channel(spec_db, f, *BAND_1)
    return np.stack([ch0, ch1], axis=0)   # (2, 32, 32)


# ── Main processor ────────────────────────────────────────────────────────────
class AudioProcessor:
    def __init__(self):
        # Sliding audio buffer (1 s)
        self.audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.data_queue   = queue.Queue()
        self.is_running   = False

        # Detection state
        self.consec    = 0     # consecutive beep windows
        self.cooldown  = 0     # steps remaining in cooldown
        self.triggered = False

        # Optional: save chunks for offline review
        self.save_dir      = "recorded_chunks"
        self.chunk_counter = 0
        os.makedirs(self.save_dir, exist_ok=True)

        # Load model
        self.device = torch.device('cpu')  # RPi has no CUDA
        info = torch.load(MODEL_INFO_PATH, map_location='cpu', weights_only=False)
        num_classes  = info.get('num_classes', 2)
        input_shape  = info.get('input_shape', (N_CHANNELS, SPEC_H, SPEC_W))
        self.class_names = info.get('class_names', ['Noise', 'Beep'])

        self.model = AudioCNN(num_classes, input_shape).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        self.model.eval()
        print(f"Model loaded — classes: {self.class_names}")
        print(f"Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/RATE*1000:.0f} ms)  "
              f"Step: {STEP_SIZE} samples ({STEP_SIZE/RATE*1000:.0f} ms)  SR: {RATE} Hz")
        print(f"Confirm: {CONFIRM_N} consecutive windows  |  "
              f"Threshold: {CONFIDENCE_THRESHOLD}")

    # ── Recording thread ───────────────────────────────────────────────────────
    def record_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=STEP_SIZE,
        )
        print("Recording started...")
        while self.is_running:
            try:
                data      = stream.read(STEP_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.data_queue.put(audio_data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Recording stopped.")

    # ── Processing thread ──────────────────────────────────────────────────────
    def process_audio(self):
        print("Processing started...")
        while self.is_running:
            if not self.data_queue.empty():
                try:
                    new_data = self.data_queue.get()

                    # Roll sliding window: drop oldest CHUNK, append newest
                    self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
                    self.audio_buffer[-len(new_data):] = new_data

                    self.classify_window(self.audio_buffer.copy())
                except Exception as e:
                    print(f"Processing error: {e}")
            else:
                time.sleep(0.005)   # yield CPU

    # ── Classify one sliding window ────────────────────────────────────────────
    def classify_window(self, audio: np.ndarray):
        if self.triggered:
            return

        # Cooldown: skip inference, drain counter
        if self.cooldown > 0:
            self.cooldown -= 1
            self.consec = 0
            return

        spec   = to_spectrogram(audio)                          # (2, 32, 32)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 2, 32, 32)

        with torch.no_grad():
            probs     = torch.softmax(self.model(tensor), dim=1)[0]
            prob_beep = float(probs[1])

        is_beep = prob_beep >= CONFIDENCE_THRESHOLD
        if is_beep:
            self.consec += 1
        else:
            self.consec = 0

        print(f"[{time.strftime('%H:%M:%S')}] {'Beep' if is_beep else 'Noise':5s}  "
              f"p={prob_beep:.2f}  consec={self.consec}/{CONFIRM_N}")

        if self.consec >= CONFIRM_N:
            self.consec   = 0
            self.cooldown = COOLDOWN_STEPS
            print(f"BEEP CONFIRMED")
            self.triggered  = True
            self.is_running = False
            
        self.save_chunk(audio, "beep" if is_beep else "noise")

    # ── Save chunk (optional, for debugging) ──────────────────────────────────
    def save_chunk(self, audio: np.ndarray, label: str):
        try:
            ts       = int(time.time())
            filename = f"chunk_{ts}_{self.chunk_counter}_{label}.wav"
            filepath = os.path.join(self.save_dir, filename)
            audio_int = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(audio_int.tobytes())
            self.chunk_counter += 1
        except Exception as e:
            print(f"Save error: {e}")

    # ── Start / Stop ───────────────────────────────────────────────────────────
    def start(self):
        self.is_running = True
        self.record_thread  = threading.Thread(target=self.record_audio,  daemon=True)
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.record_thread.start()
        self.process_thread.start()

    def stop(self):
        self.is_running = False
        self.record_thread.join(timeout=2)
        self.process_thread.join(timeout=2)
        print("Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    processor = AudioProcessor()
    processor.start()
    try:
        while processor.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted — stopping...")
        processor.stop()
        return 

    if processor.triggered:
        print(f"Waiting {WAIT_TIME}s then triggering relay...")
        time.sleep(WAIT_TIME)
        trigger_relay()
        print("Relay triggered. Exiting.")


if __name__ == "__main__":
    main()
