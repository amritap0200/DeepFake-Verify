import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

SR = 16000
MAX_LEN = SR * 8

REAL_DIR = "/home/hp/projects/spock--/audio-dataset2/real"
FAKE_DIR = "/home/hp/projects/spock--/audio-dataset2/fake"

def fix_length(audio):
    if len(audio) > MAX_LEN:
        return audio[:MAX_LEN]
    return np.pad(audio, (0, MAX_LEN - len(audio)))

def wav_to_mel(path):
    audio, _ = librosa.load(path, sr=SR)
    audio = fix_length(audio)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=128, n_fft=1024, hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.nan_to_num(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return torch.tensor(mel).unsqueeze(0).float()

class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.real = glob.glob(real_dir + "/*.wav")
        self.fake = glob.glob(fake_dir + "/*.wav")
        
        self.real = self.real[:10000]
        self.fake = self.fake[:10000]

        self.files = self.real + self.fake
        self.labels = [0]*len(self.real) + [1]*len(self.fake)

        print("Real:", len(self.real))
        print("Fake:", len(self.fake))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = wav_to_mel(self.files[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.rnn = nn.GRU(32*32, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.fc(x)

dataset = AudioDataset(REAL_DIR, FAKE_DIR)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

model = CRNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for x, y in loader:
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss/len(loader))

torch.save(model.state_dict(), "crnn_audio_fake.pth")
print("✅ Model saved as crnn_audio_fake.pth")