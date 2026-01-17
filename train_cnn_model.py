import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2x2 pooling: 64*7*7
        self.fc2 = nn.Linear(128, 10)  # Output layer
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Store original input for visualization
        input_x = x
        
        # Convolutional layers with ReLU
        x = nn.functional.relu(self.conv1(x))  # [batch, 32, 28, 28]
        conv1_out = self.pool(x)  # [batch, 32, 14, 14]
        
        x = nn.functional.relu(self.conv2(conv1_out))  # [batch, 64, 14, 14]
        conv2_out = self.pool(x)  # [batch, 64, 7, 7]
        
        # Flatten for fully connected layers
        x = conv2_out.view(conv2_out.size(0), -1)  # [batch, 64*7*7]
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))  # [batch, 128]
        fc1_out = self.dropout(x)
        x = self.fc2(fc1_out)  # [batch, 10]
        
        # Return output and intermediate activations for visualization
        return x, fc1_out, conv2_out

def train_cnn_model():
    """Train CNN model on MNIST dataset"""
    print("🧠 Training CNN Model on MNIST Dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, _, _ = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/mnist_cnn_model.pth')
    print("✅ CNN Model saved to models/mnist_cnn_model.pth")
    
    return model

if __name__ == '__main__':
    train_cnn_model()
