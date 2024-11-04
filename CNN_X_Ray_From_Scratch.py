'''
# Below is for running on google.colab only
from google.colab import drive
drive.mount('/content/drive')

# # Make sure to run this cell to use torchmetrics.
!pip install torch torchvision torchmetrics
'''


# Import required libraries
# -------------------------
# Data loading
import random
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Train model
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# Evaluate model
from torchmetrics import Accuracy, F1Score

# Set random seeds for reproducibility
torch.manual_seed(101010)
np.random.seed(101010)
random.seed(101010)

# Define the transformations to apply to the images for use with ResNet-18
transform_mean = [0.485, 0.456, 0.406]
transform_std =[0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomRotation((0,30)),
                                transforms.Normalize(mean=transform_mean, std=transform_std)])

# Apply the image transforms
# Path is set for google colab
train_dataset = ImageFolder("/content/drive/MyDrive/Colab Notebooks/CNN-Chest_X_Ray/data/chestxrays/train", transform=transform)
test_dataset = ImageFolder('/content/drive/MyDrive/Colab Notebooks/CNN-Chest_X_Ray/data/chestxrays/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset) // 2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Start coding here
# Use as many cells as you need
image, label =  next(iter(test_loader))
print(image.shape)
print(label.shape)

# Create CNN
class ResNet18(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*28*28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='elu')  # Using ELU for activation
                if layer.bias is not None:
                    init.zeros_(layer.bias)  # Initialize biases to zero
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')  # For Linear layers
                init.zeros_(layer.bias)  # Initialize biases to zero

    def forward(self,x):
        x = self.pool(self.elu(self.conv1(x))) # output dimension: 2x112x112
        x = self.pool(self.elu(self.conv2(x))) # output dimension: 64x56x56
        x = self.pool(self.elu(self.conv3(x))) # output dimension: 128x28x28
        x = self.flatten(x)
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training Loop
model.train()
model = ResNet18()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
num_epoch = 50

for epoch in range(num_epoch):
    running_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch:{epoch + 1}/{num_epoch}; Loss:{avg_loss:.4f}')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # You can also save epoch and loss if needed
    'epoch': epoch,
    'loss': loss,
}, "/content/drive/MyDrive/Colab Notebooks/CNN-Chest_X_Ray/20241103_model_checkpoint.pth")


"""### Below is the provided model evaluation code. Run the below cell to help you evaluate the accuracy and F1-score of your fine-tuned model."""

#-------------------
# Evaluate the model
#-------------------

# Set model to evaluation mode
model = model
model.eval()

# Initialize metrics for accuracy and F1 score
accuracy_metric = Accuracy(task="binary")
f1_metric = F1Score(task="binary")

# Create lists store all predictions and labels
all_preds = []
all_labels = []

# Disable gradient calculation for evaluation
with torch.no_grad():
  for inputs, labels in test_loader:
    # Forward pass
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).view(-1,1)  # Use argmax for class prediction

    # Extend the lists with predictions and labels
    all_preds.extend(preds.tolist())
    all_labels.extend(labels.unsqueeze(1).tolist())

    # Convert lists back to tensors
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Calculate accuracy and F1 score
    test_acc = accuracy_metric(all_preds, all_labels).item()
    test_f1 = f1_metric(all_preds, all_labels).item()
    print(test_acc)
    print(test_f1)

'''
2024-11-03
test_acc: 0.73
test_f1: 0.77
'''