import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from audio_dataset import load_audio_dataset

DEVICE = "cpu"

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load data
X, y = load_audio_dataset("audio-dataset")

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train).unsqueeze(1).float()
y_train = torch.tensor(y_train).float().unsqueeze(1)

model = AudioCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(5):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "audio_model.pth")
print("audio_model.pth saved")