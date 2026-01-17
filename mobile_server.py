from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)  # Allow mobile device to connect

# Global variable to store the latest digit image
latest_digit = None

# HTML template for mobile drawing interface
MOBILE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Mobile Input</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            max-width: 400px;
            width: 100%;
        }
        h1 {
            color: white;
            margin-bottom: 10px;
            font-size: 24px;
        }
        #canvas {
            border: 3px solid #00ff00;
            background: white;
            border-radius: 10px;
            touch-action: none;
            cursor: crosshair;
            margin: 20px 0;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }
        .button-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }
        button {
            padding: 15px 25px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        #sendBtn {
            background: #00ff00;
            color: black;
            flex: 2;
        }
        #clearBtn {
            background: #ff4444;
            color: white;
            flex: 1;
        }
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        button:active {
            transform: scale(0.95);
        }
        .status {
            color: #00ff00;
            margin: 10px 0;
            font-size: 16px;
            min-height: 20px;
        }
        .instructions {
            color: #ccc;
            font-size: 14px;
            margin: 10px 0;
            line-height: 1.4;
        }
        .server-info {
            color: #888;
            font-size: 12px;
            margin-top: auto;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 MNIST Mobile Input</h1>
        <div class="instructions">
            Draw a digit (0-9) below and tap "Send to Laptop"
        </div>
        
        <canvas id="canvas" width="280" height="280"></canvas>
        
        <div class="button-group">
            <button id="clearBtn">Clear</button>
            <button id="sendBtn">Send to Laptop</button>
        </div>
        
        <div id="status" class="status">Ready to draw</div>
        
        <div class="server-info" id="serverInfo">
            Connecting to laptop...
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const serverInfo = document.getElementById('serverInfo');
        
        let drawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // Get laptop IP from URL or use default
        const urlParams = new URLSearchParams(window.location.search);
        const laptopIP = urlParams.get('ip') || window.location.hostname;
        const serverUrl = `http://${laptopIP}:5000`;
        
        serverInfo.innerHTML = `Connected to: ${serverUrl}`;
        
        // Set up canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Drawing functions
        function startDrawing(e) {
            drawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = (e.clientX || e.touches[0].clientX) - rect.left;
            lastY = (e.clientY || e.touches[0].clientY) - rect.top;
        }
        
        function draw(e) {
            if (!drawing) return;
            e.preventDefault();
            
            const rect = canvas.getBoundingClientRect();
            const currentX = (e.clientX || e.touches[0].clientX) - rect.left;
            const currentY = (e.clientY || e.touches[0].clientY) - rect.top;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();
            
            lastX = currentX;
            lastY = currentY;
        }
        
        function stopDrawing() {
            drawing = false;
        }
        
        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);
        
        // Clear canvas
        document.getElementById('clearBtn').addEventListener('click', async () => {
            // Clear local canvas
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            status.textContent = 'Canvas cleared';
            status.style.color = '#00ff00';
            
            // Send clear signal to laptop
            try {
                await fetch(serverUrl + '/clear', {
                    method: 'POST'
                });
                console.log('Clear signal sent to laptop');
            } catch (error) {
                console.error('Failed to send clear signal:', error);
            }
        });
        
        // Send to laptop
        document.getElementById('sendBtn').addEventListener('click', async () => {
            try {
                status.textContent = 'Sending to laptop...';
                status.style.color = '#ffff00';
                
                const imageData = canvas.toDataURL('image/png');
                
                const response = await fetch(serverUrl + '/digit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    status.textContent = `✅ Sent! Prediction: ${result.predicted_digit} (${(result.confidence * 100).toFixed(1)}%)`;
                    status.style.color = '#00ff00';
                } else {
                    throw new Error('Server error');
                }
            } catch (error) {
                status.textContent = '❌ Connection failed - check laptop IP';
                status.style.color = '#ff4444';
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
"""

# Global variables
latest_digit = None
clear_requested = False

def preprocess_mobile_image(image_data):
    """Convert mobile image to MNIST format using reference code approach"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array and grayscale
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # INVERT: White digits on black background (MNIST standard)
        image_array = 255 - image_array
        
        # Find bounding box of the digit (like reference code)
        coords = np.column_stack(np.where(image_array > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Crop to bounding box
            image_array = image_array[y_min:y_max+1, x_min:x_max+1]
        else:
            image_array = np.zeros((28, 28), dtype=np.uint8)
        
        # Resize to 20x20 while maintaining aspect ratio (like reference)
        if image_array.size > 0:
            image = Image.fromarray(image_array)
            image.thumbnail((20, 20), Image.Resampling.LANCZOS)
            image_array = np.array(image)
        
        # Center on 28x28 canvas (like reference)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        if image_array.size > 0:
            paste_x = (28 - image_array.shape[1]) // 2
            paste_y = (28 - image_array.shape[0]) // 2
            canvas[paste_y:paste_y+image_array.shape[0], paste_x:paste_x+image_array.shape[1]] = image_array
        
        # Normalize to [0,1] and then MNIST standard (like reference)
        digit = canvas.astype(np.float32) / 255.0
        
        return digit
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")

@app.route('/')
def mobile_interface():
    """Serve mobile drawing interface"""
    return MOBILE_HTML

@app.route('/digit', methods=['POST'])
def receive_digit():
    """Receive digit from mobile device"""
    global latest_digit
    
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_digit = preprocess_mobile_image(image_data)
        
        if processed_digit is not None:
            latest_digit = processed_digit
            return jsonify({
                'status': 'success',
                'message': 'Digit received and processed'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process image'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_digit')
def get_latest_digit():
    """Provide latest digit to visualization"""
    global latest_digit
    
    if latest_digit is not None:
        # Convert to list for JSON serialization
        digit_list = latest_digit.flatten().tolist()
        return jsonify({
            'digit': digit_list,
            'available': True
        })
    else:
        return jsonify({
            'digit': None,
            'available': False
        })

@app.route('/clear', methods=['POST'])
def handle_clear():
    """Handle clear request from mobile"""
    global latest_digit, clear_requested
    latest_digit = None
    clear_requested = True
    print("📱 Clear signal received, clearing latest digit")
    return jsonify({
        'status': 'success',
        'message': 'Clear signal sent and processed'
    })

@app.route('/check_clear')
def check_clear_signal():
    """Check if clear was requested from mobile"""
    global clear_requested
    if clear_requested:
        clear_requested = False
        return jsonify({
            'clear': True
        })
    else:
        return jsonify({
            'clear': False
        })

@app.route('/prediction_result', methods=['POST'])
def receive_prediction_result():
    """Receive prediction result from visualization to show on mobile"""
    try:
        data = request.get_json()
        predicted_digit = data.get('predicted_digit')
        confidence = data.get('confidence')
        
        return jsonify({
            'status': 'success',
            'predicted_digit': predicted_digit,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🌐 Starting MNIST Mobile Input Server")
    print("📱 Open your mobile browser and go to: http://YOUR_LAPTOP_IP:5000")
    print("💡 Replace YOUR_LAPTOP_IP with your laptop's local IP address")
    print("🔧 Server running on http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
