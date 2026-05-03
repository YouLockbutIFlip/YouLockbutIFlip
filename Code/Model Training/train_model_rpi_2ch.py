"""
Adversarial Training for Beep Signal Detection — RPi 2-Channel Edition
=======================================================================
Same as train_model_rpi.py but uses a 2-channel spectrogram input:
  Channel 0: 2000-2600 Hz  (primary tone   ~2280 Hz)
  Channel 1: 3500-4200 Hz  (secondary tone ~3800 Hz)

Each channel is zoomed independently to SPEC_H x SPEC_W, so the model
sees only the energy that matters and avoids the ~1500 Hz dead zone
between the two tones.  The CNN first conv layer changes from
Conv2d(1, 32) to Conv2d(2, 32); everything else is identical to the
single-channel RPi script.

Usage:
    python train_model_rpi_2ch.py
    python train_model_rpi_2ch.py --epochs 80 --batch-size 64
"""

import os
import re
import sys
import wave
from math import gcd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
SR          = 16_000      # target sample rate for Raspberry Pi
N_FFT       = 512         # ~31 Hz/bin at 16 kHz
HOP         = 85          # ~5 ms time resolution at 16 kHz

# Two narrow bands — one per beep tone
BAND_0      = (2000.0, 2600.0)   # primary tone   ~2280 Hz
BAND_1      = (3500.0, 4200.0)   # secondary tone ~3800 Hz

SPEC_H      = 32          # spectrogram height after zoom (per channel)
SPEC_W      = 32          # spectrogram width  after zoom (per channel)
N_CHANNELS  = 2
INPUT_SHAPE = (N_CHANNELS, SPEC_H, SPEC_W)
NUM_CLASSES = 2           # 0=Noise, 1=Beep (any of beeps 1-7)

FGSM_EPS       = 1.0      # perturbation in dB-space
FGSM_FRACTION  = 0.5      # fraction of each batch replaced by adversarial examples
AUG_NOISE_MAX  = 0.015
AUG_AMP_RANGE  = (0.6, 1.5)
AUG_SHIFT_MAX  = 67       # ~4 ms at 16 kHz
SPEC_MASK_F    = 4        # max frequency bins to zero (SpecAugment, per channel)
SPEC_MASK_T    = 4        # max time steps to zero (per channel)

# ── GPU ───────────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CNN Architecture ──────────────────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super().__init__()
        # input_shape = (2, H, W) — 2 channels in
        self.conv1 = nn.Conv2d(input_shape[0], 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,             64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,             128, kernel_size=3, padding=1)
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


# ── Feature Extraction ────────────────────────────────────────────────────────
def load_wav_float(path: str) -> tuple:
    with wave.open(path, "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        sw  = wf.getsampwidth()
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if sr != SR:
        g = gcd(SR, sr)
        audio = scipy_signal.resample_poly(audio, SR // g, sr // g)
    return audio, SR


def _band_to_channel(spec_db: np.ndarray, freqs: np.ndarray,
                     f_lo: float, f_hi: float) -> np.ndarray:
    """Extract one frequency band from a full STFT and zoom to SPEC_H x SPEC_W."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    band = spec_db[mask]
    if band.shape[0] == 0:
        band = spec_db
    h, w = band.shape
    return zoom(band, (SPEC_H / h, SPEC_W / w), order=1).astype(np.float32)


def audio_to_spectrogram(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Returns a 2-channel spectrogram (2, SPEC_H, SPEC_W):
      channel 0 = BAND_0  (primary tone   ~2280 Hz)
      channel 1 = BAND_1  (secondary tone ~3800 Hz)
    """
    f, _, Zxx = scipy_signal.stft(audio, fs=sr, nperseg=N_FFT,
                                   noverlap=N_FFT - HOP)
    spec_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)

    ch0 = _band_to_channel(spec_db, f, *BAND_0)
    ch1 = _band_to_channel(spec_db, f, *BAND_1)
    return np.stack([ch0, ch1], axis=0)   # (2, H, W)


# ── Data Augmentation ─────────────────────────────────────────────────────────
def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    audio = audio * rng.uniform(*AUG_AMP_RANGE)
    noise_std = rng.uniform(0.0, AUG_NOISE_MAX)
    audio = audio + rng.standard_normal(len(audio)).astype(np.float32) * noise_std
    shift = int(rng.integers(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1))
    audio = np.roll(audio, shift)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def spec_augment(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SpecAugment applied independently to each channel."""
    spec = spec.copy()                    # (2, H, W)
    for ch in range(spec.shape[0]):
        f_mask = int(rng.integers(0, SPEC_MASK_F + 1))
        if f_mask > 0:
            f0 = int(rng.integers(0, max(1, SPEC_H - f_mask)))
            spec[ch, f0:f0 + f_mask, :] = spec[ch].min()
        t_mask = int(rng.integers(0, SPEC_MASK_T + 1))
        if t_mask > 0:
            t0 = int(rng.integers(0, max(1, SPEC_W - t_mask)))
            spec[ch, :, t0:t0 + t_mask] = spec[ch].min()
    return spec


# ── FGSM Attack ───────────────────────────────────────────────────────────────
def fgsm_attack(model, criterion, inputs, labels, eps):
    inputs_adv = inputs.clone().detach().requires_grad_(True)
    loss = criterion(model(inputs_adv), labels)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        adv = inputs_adv + eps * inputs_adv.grad.sign()
    return adv.detach()


# ── Dataset ────────────────────────────────────────────────────────────────────
class BeepDataset(Dataset):
    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.specs   = spectrograms   # (N, 2, H, W)
        self.labels  = labels
        self.augment = augment
        self.rng     = np.random.default_rng(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx].copy()   # (2, H, W)
        if self.augment:
            spec = spec_augment(spec, self.rng)
        t = torch.tensor(spec, dtype=torch.float32)   # (2, H, W) — no unsqueeze
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return t, y


# ── Data Loading ───────────────────────────────────────────────────────────────
_LABEL_RE = re.compile(r"^.+_(\d+)\.wav$")


def collect_wav_paths(clips_dir: str) -> list:
    result = []
    for root, dirs, files in os.walk(clips_dir):
        dirs[:] = [d for d in sorted(dirs) if d != "desktop.ini"]
        for fname in sorted(files):
            if fname.endswith(".wav") and fname != "desktop.ini":
                result.append((fname, os.path.join(root, fname)))
    return result


def load_specs_from_dir(clips_dir: str,
                        augment_copies: int = 0,
                        rng: np.random.Generator = None,
                        label: str = "") -> tuple[np.ndarray, np.ndarray]:
    files = collect_wav_paths(clips_dir)
    specs_list, labels_list = [], []
    skipped = 0

    for i, (fname, fpath) in enumerate(files):
        if (i + 1) % 2000 == 0:
            print(f"  [{label}] {i + 1}/{len(files)}")
        m = _LABEL_RE.match(fname)
        if not m:
            skipped += 1
            continue
        raw_lbl = int(m.group(1))
        if raw_lbl > 7:
            skipped += 1
            continue
        lbl = 0 if raw_lbl == 0 else 1   # binary: Noise=0, Beep=1
        try:
            audio, sr = load_wav_float(fpath)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            skipped += 1
            continue

        spec = audio_to_spectrogram(audio, sr)   # (2, H, W)
        specs_list.append(spec)
        labels_list.append(lbl)

        for _ in range(augment_copies):
            specs_list.append(audio_to_spectrogram(augment_audio(audio, rng), sr))
            labels_list.append(lbl)

    if skipped:
        print(f"  [{label}] skipped {skipped} files")

    return (np.array(specs_list,  dtype=np.float32),   # (N, 2, H, W)
            np.array(labels_list, dtype=np.int64))


# ── Training ───────────────────────────────────────────────────────────────────
def train(clips_dir: str,
          adv_dir: str = None,
          epochs: int = 60,
          batch_size: int = 32,
          lr: float = 8e-4,
          test_size: float = 0.2,
          run_tag: str = "rpi_2ch"):

    print("=" * 70)
    print("ADVERSARIAL CNN TRAINING — RPi 2-Channel Edition (16 kHz)")
    print(f"  Sample rate     : {SR} Hz")
    print(f"  Channel 0       : {BAND_0[0]:.0f}-{BAND_0[1]:.0f} Hz  (primary tone  ~2280 Hz)")
    print(f"  Channel 1       : {BAND_1[0]:.0f}-{BAND_1[1]:.0f} Hz  (secondary tone ~3800 Hz)")
    print(f"  Spectrogram size: {SPEC_H} x {SPEC_W} x {N_CHANNELS} channels")
    print(f"  FGSM eps        : {FGSM_EPS} dB  (fraction={FGSM_FRACTION:.0%})")
    print("=" * 70)

    rng = np.random.default_rng(42)
    class_names = ["Noise", "Beep"]

    print(f"\nLoading real clips from {clips_dir} ...")
    X_real, y_real = load_specs_from_dir(clips_dir, augment_copies=0, label="real")
    print(f"  Real clips: {len(X_real)}")
    for c, name in enumerate(class_names):
        n = int(np.sum(y_real == c))
        print(f"    {name:8s}: {n}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=test_size, random_state=42, stratify=y_real
    )
    print(f"\n  Train (real): {len(X_tr)}   Test (real only): {len(X_te)}")

    if adv_dir and os.path.isdir(adv_dir):
        print(f"\nLoading adversarial clips from {adv_dir} ...")
        X_adv, y_adv = load_specs_from_dir(adv_dir, augment_copies=0, label="adv")
        print(f"  Adversarial clips: {len(X_adv)}")
        for c, name in enumerate(class_names):
            n = int(np.sum(y_adv == c))
            print(f"    {name:8s}: {n}")
        X_tr = np.concatenate([X_tr, X_adv], axis=0)
        y_tr = np.concatenate([y_tr, y_adv], axis=0)
    else:
        print("\nNo adversarial directory — training on real clips only.")

    print(f"\nFinal train size : {len(X_tr)}")
    print(f"Final test  size : {len(X_te)}")

    train_ds = BeepDataset(X_tr, y_tr, augment=True)
    test_ds  = BeepDataset(X_te, y_te, augment=False)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=use_cuda,
                              num_workers=2 if use_cuda else 0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              pin_memory=use_cuda,
                              num_workers=2 if use_cuda else 0)

    model     = AudioCNN(NUM_CLASSES, INPUT_SHAPE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, test_accs = [], []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_adv_used   = 0

        for inputs, labels_b in train_loader:
            inputs   = inputs.to(DEVICE, non_blocking=True)
            labels_b = labels_b.to(DEVICE, non_blocking=True)

            n_adv = int(len(inputs) * FGSM_FRACTION)
            if n_adv > 0:
                adv_inputs = fgsm_attack(model, criterion,
                                         inputs[:n_adv], labels_b[:n_adv], FGSM_EPS)
                inputs = torch.cat([adv_inputs, inputs[n_adv:]], dim=0)
                n_adv_used += n_adv

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels_b in test_loader:
                inputs   = inputs.to(DEVICE, non_blocking=True)
                labels_b = labels_b.to(DEVICE, non_blocking=True)
                preds = model(inputs).argmax(dim=1)
                total   += labels_b.size(0)
                correct += (preds == labels_b).sum().item()

        acc = 100.0 * correct / total
        test_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"audio_cnn_model_{run_tag}.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"test_acc={acc:.2f}%  best={best_acc:.2f}%  "
                  f"adv/batch={n_adv_used // len(train_loader)}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION (best checkpoint)")
    print("=" * 70)

    model.load_state_dict(torch.load(f"audio_cnn_model_{run_tag}.pth",
                                      map_location=DEVICE))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, labels_b in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            preds  = model(inputs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels_b.numpy())

    present = sorted(set(all_true))
    print(classification_report(
        all_true, all_preds,
        labels=present,
        target_names=[class_names[c] for c in present],
        digits=3
    ))

    cm = confusion_matrix(all_true, all_preds, labels=present)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class_names[c] for c in present],
                yticklabels=[class_names[c] for c in present])
    plt.title("Confusion Matrix — RPi 2-Channel Adversarial Training (16 kHz)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{run_tag}.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to: {cm_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses); ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)
    ax2.plot(test_accs); ax2.set_title("Test Accuracy (%)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.grid(True)
    plt.tight_layout()
    hist_path = f"training_history_{run_tag}.png"
    plt.savefig(hist_path, dpi=150)
    print(f"Training history saved to: {hist_path}")

    info = {
        "input_shape": INPUT_SHAPE,
        "num_classes": NUM_CLASSES,
        "class_names": ["Noise", "Beep"],
        "band_0": BAND_0,
        "band_1": BAND_1,
        "spec_h": SPEC_H,
        "spec_w": SPEC_W,
        "n_fft": N_FFT,
        "hop": HOP,
        "sr": SR,
        "test_accuracy": best_acc,
        "adversarial": True,
        "fgsm_eps": FGSM_EPS,
    }
    info_path = f"model_info_{run_tag}.pth"
    torch.save(info, info_path)
    print(f"Model metadata saved to: {info_path}")
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    return model, best_acc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=8e-4)
    parser.add_argument("--test-size",  type=float, default=0.2)
    parser.add_argument("--no-adv",     action="store_true")
    parser.add_argument("--run-tag",    default=None)
    args = parser.parse_args()

    clips_dir = os.path.join(os.path.dirname(__file__), "clips_16k")
    adv_dir   = os.path.join(os.path.dirname(__file__), "clips_adversarial_16k")

    if not os.path.isdir(clips_dir):
        print(f"ERROR: {clips_dir} not found. Run prepare_dataset_16k.py first.")
        sys.exit(1)
    if args.no_adv:
        adv_dir = None
    elif not os.path.isdir(adv_dir):
        print(f"WARNING: {adv_dir} not found — training without adversarial data.")
        adv_dir = None

    if args.batch_size is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = 128 if vram_gb >= 8 else (64 if vram_gb >= 4 else 32)
        else:
            batch_size = 32
    else:
        batch_size = args.batch_size

    tag = args.run_tag or ("rpi_2ch_baseline" if args.no_adv else "rpi_2ch")
    print(f"Batch size: {batch_size}  |  run tag: {tag}")
    train(clips_dir, adv_dir, args.epochs, batch_size, args.lr, args.test_size,
          run_tag=tag)