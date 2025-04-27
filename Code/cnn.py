import numpy as np
import pandas as pd
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def read_labels(labels_file):
    labels_df = pd.read_csv(labels_file)
    print(f"Total rows in labels file: {len(labels_df)}")
    
    if len(labels_df.columns) < 2:
        raise ValueError("The labels file should have at least two columns: filename and label")
    
    file_names = labels_df.iloc[:, 0].apply(lambda x: os.path.splitext(x)[0]).values
    labels = labels_df.iloc[:, 1].values
    
    label_dict = dict(zip(file_names, labels))
    print(f"Unique labels: {set(labels)}")
    print(f"Number of unique file names: {len(set(file_names))}")
    return label_dict

def read_wav_files(audio_dir):
    audio_files = []
    file_names = []
    
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav') and not filename.startswith('.'):
            file_path = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(file_path, sr=None)
            
            # calculate STFT
            n_fft = 2048  
            hop_length = 512  
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            
            # calculate spectrogram
            spectrogram = np.abs(stft)
            
            # conveert to dB 
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            
            # get freq
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # locate 3600-4000 Hz index
            start_idx = np.argmin(np.abs(freqs - 3600))
            end_idx = np.argmin(np.abs(freqs - 4000))
            spectrogram_filtered = spectrogram_db[start_idx:end_idx+1, :]
            
            audio_files.append(spectrogram_filtered)
            file_names.append(filename)

    return np.array(audio_files), file_names

def load_data(audio_dir, labels_file):
    audio_data, file_names = read_wav_files(audio_dir)
    labels_dict = read_labels(labels_file)

    filtered_audio_data = []
    audio_labels = []

    for i, audio in enumerate(audio_data):
        wav_file_name = os.path.splitext(file_names[i])[0]
        
        if wav_file_name in labels_dict:
            filtered_audio_data.append(audio)
            audio_labels.append(labels_dict[wav_file_name])
        elif wav_file_name + "_spectrogram" in labels_dict:
            filtered_audio_data.append(audio)
            audio_labels.append(labels_dict[wav_file_name + "_spectrogram"])

    audio_data = np.array(filtered_audio_data)
    audio_labels = np.array(audio_labels)

    print(f"Matched files: {len(audio_data)}")
    print(f"Final number of audio files: {len(audio_data)}")
    print(f"Final number of labels: {len(audio_labels)}")

    assert len(audio_data) == len(audio_labels), "Mismatch between audio data and labels"

    return audio_data, audio_labels

def preprocess_data(audio_data, audio_labels, test_size=0.2, random_state=42):
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(audio_labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        audio_data, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
    )

    return X_train, X_test, y_train, y_test, label_encoder

class AudioCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
        # Calculate the size of the output from the last convolutional layer
        self.fc_input_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_conv(input)
        n_size = output.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'audio_cnn_model.pth')
    print("Model saved as 'audio_cnn_model.pth'")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

def plot_spectrograms(audio_data, audio_labels, file_names, num_samples_per_class=5):
    label_0_indices = np.where(audio_labels == 0)[0]
    label_1_indices = np.where(audio_labels == 1)[0]

    num_samples_per_class = min(num_samples_per_class, len(label_0_indices), len(label_1_indices))


    np.random.seed(42)  
    label_0_samples = np.random.choice(label_0_indices, num_samples_per_class, replace=False)
    label_1_samples = np.random.choice(label_1_indices, num_samples_per_class, replace=False)

    selected_indices = np.concatenate([label_0_samples, label_1_samples])

    fig, axes = plt.subplots(2, num_samples_per_class, figsize=(25, 10))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        spectrogram = audio_data[idx]
        label = audio_labels[idx]
        file_name = file_names[idx]
        
        im = axes[i].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f'File: {file_name}\nLabel: {label}', fontsize=8)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='both', which='major', labelsize=6)
    
    plt.tight_layout()
    plt.savefig('spectrogram_samples.png', dpi=300)
    plt.close()

    print("Spectrogram samples saved as 'spectrogram_samples.png'")
    print("Label distribution in the entire dataset:")
    unique_labels, counts = np.unique(audio_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count}")

def main():
    audio_dir = 'audioclips/'
    labels_file = 'labels1.csv'

    audio_data, audio_labels = load_data(audio_dir, labels_file)

    print(f"Number of audio files: {len(audio_data)}")
    print(f"Number of labels: {len(audio_labels)}")
    print(f"Unique labels: {np.unique(audio_labels)}")
    print(f"Shape of audio data: {audio_data.shape}")


    file_names = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.')]
    file_names = file_names[:len(audio_data)]  


    plot_spectrograms(audio_data, audio_labels, file_names, num_samples_per_class=5)

    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(audio_data, audio_labels)

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)
    input_shape = (1, X_train.shape[2], X_train.shape[3])  # (channels, height, width)
    model = AudioCNN(num_classes, input_shape).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Load the saved model for evaluation
    saved_model_path = 'audio_cnn_model.pth'
    loaded_model = AudioCNN(num_classes, input_shape).to(device)
    loaded_model.load_state_dict(torch.load(saved_model_path))
    loaded_model.eval()

    print(f"Loaded model from {saved_model_path}")

    # Evaluate the loaded model
    evaluate_model(loaded_model, test_loader, device)

if __name__ == "__main__":
    main()
