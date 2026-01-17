# MNIST Mobile Input Setup Guide

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the System:**
   ```bash
   # Windows
   start_mobile.bat
   
   # Or manually:
   # Terminal 1: Start mobile server
   python mobile_server.py
   
   # Terminal 2: Start visualization
   python educational_visualization.py --arduino
   ```

3. **Connect Mobile Device:**
   - Find your laptop's local IP (run `ipconfig` on Windows)
   - On mobile device, open: `http://YOUR_LAPTOP_IP:5000`
   - Draw a digit and tap "Send to Laptop"

## System Architecture

```
Mobile Device (Browser)     Laptop (Local)
┌─────────────────┐         ┌─────────────────────────┐
│  Drawing Canvas │  HTTP   │  Flask Server         │
│  280x280px     │◄──────►│  Port 5000           │
│  Touch Input   │         │  Image Processing     │
└─────────────────┘         │  MNIST Preprocessing  │
                            │  PyTorch Inference    │
                            │  Pygame Visualization │
                            │  Arduino (Optional)    │
                            └─────────────────────────┘
```

## Features

### Mobile Input
- **Touch-friendly drawing canvas** (280x280px)
- **Real-time preprocessing** for MNIST compatibility
- **Instant feedback** with prediction results
- **Connection status** indicator

### Laptop Visualization
- **Real neural network activations** (no simulation)
- **Layer-specific connection colors:**
  - 🔵 Input → Hidden Layer 1 (Blue)
  - 🟡 Hidden Layer 1 → Hidden Layer 2 (Yellow) 
  - 🟢 Hidden Layer 2 → Output (Green)
- **Animated inference flow** with 300ms delays
- **Educational explanations** (toggle with 'E')

### Technical Details
- **Input Processing:** Grayscale → Invert → Threshold → Center → Resize → Normalize
- **Model Architecture:** 784 → 256 → 128 → 10 (ReLU + Softmax)
- **Connection Thickness:** Reflects activation strength
- **Neuron Brightness:** Proportional to activation value

## Troubleshooting

### Mobile Connection Issues
1. **Check laptop IP:** Run `ipconfig` and find IPv4 address
2. **Same Wi-Fi network:** Both devices must be on same network
3. **Firewall:** Allow port 5000 through Windows Firewall
4. **Server running:** Check console for "Server running on http://0.0.0.0:5000"

### Prediction Accuracy Issues
1. **Draw clearly:** Use thick strokes, fill in the digit
2. **Center the digit:** Draw in the middle of the canvas
3. **Good contrast:** Black strokes on white background
4. **Standard digits:** Write numbers 0-9 normally

### Performance Issues
1. **Close other apps:** Free up CPU/GPU resources
2. **Check dependencies:** Ensure all packages installed correctly
3. **Restart system:** Try closing and reopening both terminals

## Advanced Usage

### Custom Server URL
```bash
python educational_visualization.py --server=http://192.168.1.100:5000
```

### Arduino Integration
```bash
python educational_visualization.py --arduino --server=http://192.168.1.100:5000
```

### Development Mode
- **Mobile server:** `python mobile_server.py` (serves web interface)
- **Visualization:** `python educational_visualization.py` (runs neural network)
- **Both required:** Mobile server must run for input to work

## File Structure
```
nmist/
├── mobile_server.py          # Flask server for mobile input
├── educational_visualization.py  # Main visualization (modified)
├── train_model.py           # Model training (unchanged)
├── arduino_controller.py     # Arduino integration (unchanged)
├── requirements.txt          # Dependencies (updated)
├── start_mobile.bat         # Windows launcher script
└── MOBILE_SETUP.md         # This guide
```

## Network Requirements
- **Local Wi-Fi network** (no internet required)
- **Port 5000** open on laptop
- **HTTP/HTTPS** supported
- **Real-time communication** via polling

This system transforms your MNIST visualization into a mobile-powered interactive demo while maintaining all the educational features and technical accuracy.
