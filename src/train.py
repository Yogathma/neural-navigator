import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 🔥 FIX: relative imports
from dataset import NeuralNavigatorDataset
from model import NeuralNavigator

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = NeuralNavigatorDataset("data")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = NeuralNavigator().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, text, target_path in loader:
            images = images.to(device)
            text = text.to(device)
            target_path = target_path.to(device)

            pred_path = model(images, text)
            loss = criterion(pred_path, target_path)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "outputs/neural_navigator.pth")
    print("✅ Model saved at outputs/neural_navigator.pth")

if __name__ == "__main__":
    train()
