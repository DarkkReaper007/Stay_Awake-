# Custom CNN Face Detection - Screen Power Manager

Build and train your own neural network from scratch to detect faces and control screen power!

## 🎯 Project Overview

This project lets you:
1. **Build a custom CNN** architecture from scratch
2. **Collect your own training data** using your webcam
3. **Train the neural network** on your data
4. **Deploy the trained model** to control your screen power automatically

## 🧠 What You'll Learn

- Building CNN architectures with PyTorch
- Data collection and preprocessing
- Training neural networks with backpropagation
- Model evaluation and validation
- Real-time inference with computer vision
- System integration (screen power control)

## 📁 Project Structure

```
deep_learning_project/
├── model.py                          # CNN architecture definitions
├── collect_data.py                   # Data collection script
├── train.py                          # Training pipeline
├── detect_with_screen_control.py    # Inference with screen control
├── requirements_custom.txt           # Dependencies
├── training_data/                    # Collected training images
│   ├── face/                        # Face images (class 1)
│   └── no_face/                     # No face images (class 0)
├── best_model.pth                    # Best model checkpoint (after training)
└── training_history.png              # Training visualization
```

## 🚀 Complete Workflow

### Step 1: Install Dependencies

```bash
pip install -r requirements_custom.txt
```

This installs:
- PyTorch (deep learning framework)
- OpenCV (computer vision)
- NumPy (numerical computing)
- Matplotlib (visualization)
- And more...

### Step 2: Test the Model Architecture

```bash
python model.py
```

This displays:
- Model architecture summary
- Number of parameters
- Output shapes

**Two Models Available:**
- `SimpleFaceDetectionCNN` - ~250K parameters (faster, good for laptops)
- `FaceDetectionCNN` - ~2M parameters (more accurate, needs good GPU)

### Step 3: Collect Training Data

```bash
python collect_data.py
```

**Instructions:**
1. Position yourself in front of the camera
2. Press **'1'** to capture FACE images (with you in frame)
3. Press **'0'** to capture NO FACE images (step away from camera)
4. Collect **200-300 images per class** minimum
5. Vary your position, lighting, distance, and expressions
6. Press **'q'** to quit

**Tips for Good Data:**
- Different lighting conditions (bright, dim, side-lit)
- Different positions (centered, left, right, close, far)
- Different expressions (neutral, smiling, etc.)
- Include glasses on/off if you wear them
- More diverse data = better model performance!

### Step 4: Train the Neural Network

```bash
python train.py
```

**What Happens:**
1. Loads your collected data
2. Splits into training (80%) and validation (20%)
3. Trains the CNN for 20 epochs
4. Uses Adam optimizer with learning rate 0.001
5. Applies learning rate scheduling
6. Saves best model based on validation accuracy
7. Generates training history plot

**Training Output:**
- Real-time progress bars
- Loss and accuracy metrics per epoch
- Best model saved as `best_model.pth`
- Final model saved as `final_model.pth`
- Training plot saved as `training_history.png`

**Expected Training Time:**
- SimpleCNN: 5-10 minutes (CPU) / 1-2 minutes (GPU)
- FullCNN: 15-30 minutes (CPU) / 3-5 minutes (GPU)

**Expected Accuracy:**
- With 200+ images per class: 85-95%
- With 500+ images per class: 90-98%

### Step 5: Run Face Detection with Screen Control

```bash
python detect_with_screen_control.py
```

**How It Works:**
1. Loads your trained model
2. Starts webcam feed
3. Processes each frame through your CNN
4. Detects face with confidence score
5. Controls screen power based on presence
6. Screen turns OFF after 10 seconds of absence
7. Screen turns ON immediately when you return

**Press 'q' to quit** - screen automatically turns back on

## ⚙️ Configuration

### Model Selection

Edit in `train.py` and `detect_with_screen_control.py`:

```python
MODEL_TYPE = 'simple'  # or 'full'
```

### Training Parameters

Edit in `train.py`:

```python
BATCH_SIZE = 16           # Increase if you have more RAM/VRAM
NUM_EPOCHS = 20           # More epochs for more training
LEARNING_RATE = 0.001     # Lower = slower but more stable
VAL_SPLIT = 0.2          # 20% for validation
```

### Detection Parameters

Edit in `detect_with_screen_control.py`:

```python
ABSENCE_TIMEOUT = 10           # Seconds before screen turns off
CONFIDENCE_THRESHOLD = 0.7     # 0.0-1.0 (higher = more strict)
```

## 📊 Understanding the CNN Architecture

### SimpleFaceDetectionCNN

```
Input: 3x128x128 RGB image
    ↓
Conv Layer 1: 3→16 channels + ReLU + MaxPool → 16x64x64
    ↓
Conv Layer 2: 16→32 channels + ReLU + MaxPool → 32x32x32
    ↓
Conv Layer 3: 32→64 channels + ReLU + MaxPool → 64x16x16
    ↓
Flatten: 16384 features
    ↓
FC Layer 1: 16384→128 + ReLU + Dropout(0.5)
    ↓
FC Layer 2: 128→2 (Face/No Face)
    ↓
Output: 2 class scores
```

**Key Concepts:**
- **Convolutional layers**: Extract features (edges, shapes, patterns)
- **MaxPooling**: Reduce spatial dimensions, keep important features
- **ReLU**: Non-linear activation function
- **Dropout**: Prevents overfitting during training
- **Fully Connected**: Final classification based on extracted features

## 🔧 Troubleshooting

### "Not enough training data"
- Collect at least 200 images per class
- More is better! Aim for 300-500 per class

### "Low training accuracy"
- Collect more diverse data
- Increase number of epochs
- Try the full CNN model instead of simple

### "Low validation accuracy but high training accuracy"
- Model is overfitting
- Collect more data
- Increase dropout rates in `model.py`

### "Model detects faces incorrectly"
- Lower confidence threshold
- Collect more varied training data
- Include edge cases in training data

### "Screen doesn't turn on/off"
- Check Windows power settings
- Run as administrator
- Some laptops may have hardware overrides

### "Camera not working"
- Close other apps using the camera
- Check camera permissions
- Try different camera index in code

## 📈 Improving Your Model

1. **Collect More Data**: 1000+ images per class = excellent results
2. **Data Augmentation**: Add flips, rotations, brightness changes
3. **Hyperparameter Tuning**: Try different learning rates, batch sizes
4. **Architecture Modifications**: Add more layers, change filter sizes
5. **Transfer Learning**: Start from a pretrained model (advanced)

## 🎓 Learning Resources

**Understanding CNNs:**
- 3Blue1Brown - Neural Networks series (YouTube)
- Stanford CS231n - Convolutional Neural Networks
- PyTorch tutorials - pytorch.org

**Deep Learning Concepts:**
- Backpropagation
- Gradient descent
- Overfitting vs Underfitting
- Activation functions
- Loss functions

## 💡 Project Ideas for Extension

1. **Multi-person detection** - Keep screen on if anyone is present
2. **Face recognition** - Only respond to specific people
3. **Attention tracking** - Detect if you're looking at screen
4. **Time tracking** - Log how long you're at your computer
5. **Break reminders** - Alert if you've been sitting too long
6. **Privacy mode** - Blur screen when others approach

## 🏆 Success Criteria

After completing this project, you'll have:
- ✅ Built a CNN from scratch using PyTorch
- ✅ Collected and preprocessed your own dataset
- ✅ Trained a neural network with backpropagation
- ✅ Evaluated model performance
- ✅ Deployed a real-time AI application
- ✅ Integrated with system hardware control

## 📝 Commands Summary

```bash
# Install dependencies
pip install -r requirements_custom.txt

# Test model architecture
python model.py

# Collect training data
python collect_data.py

# Train the neural network
python train.py

# Run detection with screen control
python detect_with_screen_control.py
```

## 🎉 Final Notes

This is a complete deep learning pipeline! You're building everything from scratch:
- Neural network architecture ✓
- Data collection ✓
- Training loop ✓
- Inference engine ✓
- System integration ✓

Enjoy learning and building your own AI! 🚀

---

**Questions or Issues?** Check the troubleshooting section or review the code comments for detailed explanations.
