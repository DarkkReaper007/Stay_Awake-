"""
Training Script for Custom Face Detection CNN
Trains the neural network on collected data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import json
from model import FaceDetectionCNN, SimpleFaceDetectionCNN, count_parameters


class FaceDataset(Dataset):
    """Custom dataset for face detection"""
    
    def __init__(self, data_dir='training_data', transform=None):
        """
        Args:
            data_dir: Root directory with 'face' and 'no_face' subdirectories
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Load face images (label=1)
        face_dir = self.data_dir / 'face'
        if face_dir.exists():
            for img_path in face_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(1)
        
        # Load no-face images (label=0)
        no_face_dir = self.data_dir / 'no_face'
        if no_face_dir.exists():
            for img_path in no_face_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images:")
        print(f"  - Face: {sum(self.labels)} images")
        print(f"  - No Face: {len(self.labels) - sum(self.labels)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_name='simple', device=None):
        """
        Args:
            model_name: 'simple' or 'full'
            device: torch device (auto-detected if None)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        if model_name == 'simple':
            self.model = SimpleFaceDetectionCNN()
        else:
            self.model = FaceDetectionCNN()
        
        self.model = self.model.to(self.device)
        print(f"Model: {model_name.upper()}")
        print(f"Parameters: {count_parameters(self.model):,}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100 * correct / total:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_val_acc = 0.0
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"  ✅ New best model! (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, filename='model.pth'):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='model.pth'):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {filename}")
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
        plt.show()


def main():
    """Main training pipeline"""
    print("="*60)
    print("Custom Face Detection CNN - Training Pipeline")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'training_data'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    MODEL_TYPE = 'simple'  # 'simple' or 'full'
    
    # Check if data exists
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"\n❌ Error: Data directory '{DATA_DIR}' not found!")
        print("Run 'python collect_data.py' first to collect training data.")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = FaceDataset(DATA_DIR)
    
    if len(dataset) < 50:
        print(f"\n⚠️  Warning: Only {len(dataset)} images found.")
        print("Recommendation: Collect at least 200 images per class for better results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Split dataset
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize trainer
    trainer = Trainer(model_name=MODEL_TYPE)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    
    # Save final model
    trainer.save_model('final_model.pth')
    
    # Plot training history
    trainer.plot_history()
    
    print("\n✅ Training complete!")
    print("   - Best model saved as 'best_model.pth'")
    print("   - Final model saved as 'final_model.pth'")
    print("   - Training plot saved as 'training_history.png'")
    print("\nNext step: Run 'python detect_with_screen_control.py' to use your trained model!")


if __name__ == "__main__":
    main()
