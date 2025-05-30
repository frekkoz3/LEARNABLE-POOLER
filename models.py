"""
    This material is develop for academic purpose. 
    It is develop by Francesco Bredariol as final project of the Introduction to ML course (year 2024-2025).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from poolers import *

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels = 1,pooler = nn.MaxPool2d):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 3, 32, 32)
        self.pool1 = pooler(kernel_size=2, stride=2, in_channels = 16)  # Output: (B, 16, 16, 16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 16, 16)
        self.pool2 = pooler(kernel_size=2, stride=2, in_channels =32)  # Output: (B, 32, 8, 8)

        self.fc1 = nn.Linear(32 * 7 * 7 , 128) # It wasn't 8x8 but 7x7 (for the stride)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64)

    # Initialize model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(pooler=MixingPooling2d).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model.pool1.alpha)
    print(model.pool2.alpha)

    for epoch in range(5):  # number of epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    print(model.pool1.alpha)
    print(model.pool2.alpha)
    


