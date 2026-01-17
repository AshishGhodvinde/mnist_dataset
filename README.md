# MNIST Neural Network Visualization

A live, interactive visualization of a deep learning model trained on the MNIST dataset, designed for educational demonstrations at technical exhibitions.

## System Architecture
```
Mobile Device (Browser)     Laptop (Local)
┌─────────────────┐         ┌─────────────────────────────────┐
│  Drawing Canvas │  HTTP   │  Pygame Visualization        │
│  280×280px     │◄──────►│  - 28×28 Input Grid        │
│  Touch Input   │         │  - Hidden Layer Animations   │
│  Clear/Send   │         │  - Output Layer Results      │
└─────────────────┘         └─────────────────────────────────┘
```

## Key Features
- **Mobile Input**: Draw digits on phone/tablet browser
- **Real-time Processing**: No training during visualization
- **Educational Flow**: Watch data travel through neural network
- **Layer-specific Colors**: Blue → Yellow → Green connections
- **Pre-trained Model**: Uses existing trained model (no live training)

## Quick Start

### Method 1: Easy Launcher
```bash
# Windows
start_mobile.bat
```

### Method 2: Manual Setup
```bash
# Terminal 1: Start mobile server
python mobile_server.py

# Terminal 2: Start visualization  
python educational_visualization.py

# On mobile device, open:
http://YOUR_LAPTOP_IP:5000
```

## Usage Instructions

1. **Find Laptop IP**: Run `ipconfig` and find IPv4 address
2. **Start Mobile Server**: Opens port 5000 for mobile connections
3. **Connect Mobile**: Open browser on phone/tablet to laptop's IP
4. **Draw & Send**: Sketch digit and tap "Send to Laptop"
5. **Watch Visualization**: See neural network process in real-time

## Technical Details

### Model Architecture
- **Input**: 784 neurons (28×28 pixels)
- **Hidden Layer 1**: 256 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)  
- **Output**: 10 neurons (Softmax - digits 0-9)

### Processing Pipeline
1. **Mobile Drawing** → Base64 image data
2. **HTTP Transfer** → Flask server receives image
3. **Preprocessing** → Grayscale → Center → Resize → Normalize
4. **Neural Network** → Forward pass through layers
5. **Visualization** → Animated connections and activations

### Animation Timing
- **Total Duration**: 6 seconds
- **Input → Hidden 1**: 0-2 seconds (Blue connections)
- **Hidden 1 → Hidden 2**: 2-4 seconds (Yellow connections)
- **Hidden 2 → Output**: 4-6 seconds (Green connections)

## File Structure
```
nmist/
├── mobile_server.py          # Flask server for mobile input
├── educational_visualization.py  # Main visualization (current)
├── train_model.py           # Model training (run once)
├── arduino_controller.py     # Arduino integration (optional)
├── requirements.txt          # Python dependencies
├── start_mobile.bat         # Easy launcher script
└── models/                 # Trained model storage
    └── mnist_model.pth      # Pre-trained neural network
```

## Dependencies
- **PyTorch**: Neural network framework
- **Pygame**: Visualization and UI
- **Flask**: Mobile web server
- **OpenCV**: Image preprocessing
- **NumPy**: Numerical operations
- **Requests**: HTTP communication

## Model Training
The system uses a **pre-trained model** for instant response. Training is done once via:

```bash
python train_model.py
```

This creates `models/mnist_model.pth` which the visualization loads automatically.

## Mobile Interface Features
- **Touch-optimized drawing canvas**
- **Real-time connection status**
- **Instant prediction feedback**
- **Clear button syncs both displays**
- **Works on any mobile browser**

## Educational Elements
- **Layer explanations** (toggle with 'E' key)
- **Connection animations** showing data flow
- **Neuron brightness** reflects activation strength
- **Color-coded connections** by layer type
- **Real activation values** (no simulation)

## Troubleshooting

### Connection Issues
- **Same WiFi Network**: Both devices must be on same network
- **Firewall**: Allow port 5000 through Windows Firewall
- **IP Address**: Use `ipconfig` to find correct laptop IP

### Performance Issues
- **Close other apps**: Free up CPU/GPU resources
- **Check dependencies**: Ensure all packages installed
- **Restart system**: Try closing and reopening both terminals
1. **Model not found**: Run `train_model.py` first
2. **Arduino connection**: Check COM port and baud rate
3. **Performance**: Close other applications for smooth animation
4. **Drawing issues**: Ensure mouse is within canvas bounds

### Performance Tips

- Use GPU if available for faster inference
- Reduce neuron display count for lower-end systems
- Disable Arduino if not needed

## Educational Value

This visualization demonstrates:
- How neural networks process visual information
- The concept of activation functions
- Information flow through network layers
- Confidence in machine learning predictions
- Real-time inference capabilities

Perfect for teaching deep learning concepts to students and the general public!
