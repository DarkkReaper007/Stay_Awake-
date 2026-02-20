"""
Custom CNN for Face Presence Detection
A lightweight convolutional neural network built from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionCNN(nn.Module):
    """
    Custom CNN for binary classification: Face Present or Not Present
    
    Architecture:
    - 3 Convolutional blocks with max pooling
    - 2 Fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, input_channels=3, num_classes=2):
        super(FaceDetectionCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 8 * 8)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_architecture_summary(self):
        """Return a summary of the model architecture"""
        return """
        FaceDetectionCNN Architecture:
        ===============================
        Input: 3x128x128 RGB image
        
        Conv Block 1: Conv(3->32) + BatchNorm + ReLU + MaxPool -> 32x64x64
        Conv Block 2: Conv(32->64) + BatchNorm + ReLU + MaxPool -> 64x32x32
        Conv Block 3: Conv(64->128) + BatchNorm + ReLU + MaxPool -> 128x16x16
        Conv Block 4: Conv(128->256) + BatchNorm + ReLU + MaxPool -> 256x8x8
        
        Flatten: 256x8x8 -> 16384
        
        FC Layer 1: 16384 -> 512 + ReLU + Dropout(0.5)
        FC Layer 2: 512 -> 128 + ReLU + Dropout(0.3)
        FC Layer 3: 128 -> 2 (Face Present / No Face)
        
        Output: 2 class scores (logits)
        ===============================
        """


class SimpleFaceDetectionCNN(nn.Module):
    """
    Lighter version of the CNN for faster training and inference
    Good for laptops with limited GPU/CPU
    """
    
    def __init__(self, input_channels=3, num_classes=2):
        super(SimpleFaceDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with random input"""
    print("Testing FaceDetectionCNN...")
    model = FaceDetectionCNN()
    print(model.get_architecture_summary())
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\n" + "="*60)
    print("\nTesting SimpleFaceDetectionCNN...")
    simple_model = SimpleFaceDetectionCNN()
    print(f"Total trainable parameters: {count_parameters(simple_model):,}")
    
    output = simple_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    test_model()
