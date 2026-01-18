import pygame
import numpy as np
import torch
import torch.nn as nn
from train_model import MNISTNeuralNetwork
import cv2
from PIL import Image
import math
import time
import sys
import os
import requests
import threading
from queue import Queue
import math

class EducationalNeuralNetworkVisualizer:
    def __init__(self, width=1600, height=900, server_url="http://localhost:5000"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MNIST Neural Network - Mobile Input Visualization")
        self.clock = pygame.time.Clock()
        
        # Mobile server connection
        self.server_url = server_url
        self.mobile_digit_queue = Queue()
        self.latest_mobile_digit = None
        self.polling_thread = None
        self.start_mobile_polling()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        self.ORANGE = (255, 165, 0)
        
        # Mobile input display (replaces canvas)
        self.mobile_display_size = 280
        self.mobile_display_x = 50
        self.mobile_display_y = 150
        self.mobile_surface = pygame.Surface((self.mobile_display_size, self.mobile_display_size))
        self.mobile_surface.fill(self.WHITE)
        
        # Connection status
        self.mobile_connected = False
        self.last_mobile_check = time.time()
        
        # Neural network visualization areas
        self.input_x = 400
        self.input_y = 150
        self.hidden1_x = 800
        self.hidden1_y = 150
        self.hidden2_x = 1150
        self.hidden2_y = 150
        self.output_x = 1450
        self.output_y = 150
        
        # Load CNN model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load CNN model first, fallback to MLP
        model_loaded = False
        try:
            # Import CNN model class
            import sys
            sys.path.append('.')
            from train_cnn_model import MNIST_CNN
            
            self.model = MNIST_CNN().to(self.device)
            self.model.load_state_dict(torch.load('models/mnist_cnn_model.pth', map_location=self.device))
            self.model.eval()
            print(f"CNN Model loaded successfully on {self.device}")
            model_loaded = True
            self.model_type = "CNN"
        except FileNotFoundError:
            print("CNN model not found, trying MLP model...")
            try:
                from train_model import MNISTNeuralNetwork
                self.model = MNISTNeuralNetwork(hidden1_size=256, hidden2_size=128).to(self.device)
                self.model.load_state_dict(torch.load('models/mnist_model.pth', map_location=self.device))
                self.model.eval()
                print(f"MLP Model loaded successfully on {self.device}")
                model_loaded = True
                self.model_type = "MLP"
            except FileNotFoundError:
                print("No models found. Please run training script first")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            try:
                from train_model import MNISTNeuralNetwork
                self.model = MNISTNeuralNetwork(hidden1_size=256, hidden2_size=128).to(self.device)
                self.model.load_state_dict(torch.load('models/mnist_model.pth', map_location=self.device))
                self.model.eval()
                print(f"MLP Model loaded successfully on {self.device}")
                model_loaded = True
                self.model_type = "MLP"
            except FileNotFoundError:
                print("No models found. Please run training script first")
        
        # Animation variables
        self.activations = None
        self.predictions = None
        self.animation_time = 0
        self.predicted_digit = None
        self.confidence = 0
        self.animation_phase = 0
        self.current_explanation = ""
        
        # Educational content
        self.explanations = {
            "input": "Input Layer: Each pixel becomes a neuron (784 total)\nWhite pixels = high activation, Black = low",
            "hidden1": "Hidden Layer 1: Detects basic patterns\nEdges, curves, lines - building blocks of digits",
            "hidden2": "Hidden Layer 2: Combines patterns\nRecognizes more complex shapes and combinations",
            "output": "Output Layer: Final decision\nEach neuron votes for a digit (0-9)\nBrightest = most confident prediction",
            "connections": "Connections: Information flow\nBrightness shows signal strength\nThicker lines = stronger influence"
        }
        
        # Fonts
        try:
            self.font_tiny = pygame.font.Font(None, 18)
            self.font_small = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_large = pygame.font.Font(None, 48)
            self.font_title = pygame.font.Font(None, 42)
        except:
            self.font_tiny = pygame.font.SysFont('Arial', 14)
            self.font_small = pygame.font.SysFont('Arial', 20)
            self.font_medium = pygame.font.SysFont('Arial', 28)
            self.font_large = pygame.font.SysFont('Arial', 40)
            self.font_title = pygame.font.SysFont('Arial', 36)
        
        # Add screen rect for clipping
        self.screen_rect = self.screen.get_rect()
        
        # Status
        self.model_ready = model_loaded
        self.last_prediction_time = 0
        self.show_explanations = True
        self.processing_digit = False  # Add flag to prevent repeated processing
        
    def start_mobile_polling(self):
        """Start background thread to poll mobile server for digit input"""
        def poll_mobile():
            while True:
                try:
                    # Check for clear signal first
                    if self.check_mobile_clear():
                        pass  # Clear already handled in check_mobile_clear
                    
                    # Check for new digit
                    response = requests.get(f"{self.server_url}/get_digit", timeout=1.0)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('available') and data.get('digit'):
                            digit_array = np.array(data['digit'], dtype=np.float32).reshape(28, 28)
                            self.mobile_digit_queue.put(digit_array)
                            self.mobile_connected = True
                        else:
                            self.mobile_connected = False
                except requests.exceptions.RequestException:
                    self.mobile_connected = False
                except Exception as e:
                    print(f"Mobile polling error: {e}")
                    self.mobile_connected = False
                
                time.sleep(0.1)  # Poll every 100ms for faster response
        
        self.polling_thread = threading.Thread(target=poll_mobile, daemon=True)
        self.polling_thread.start()
        print("📱 Mobile polling thread started")
    
    def send_prediction_to_mobile(self, predicted_digit, confidence):
        """Send prediction result back to mobile device"""
        try:
            response = requests.post(
                f"{self.server_url}/prediction_result",
                json={
                    'predicted_digit': int(predicted_digit),
                    'confidence': float(confidence)
                },
                timeout=2.0
            )
            if response.status_code == 200:
                print(f"✓ Prediction sent to mobile: {predicted_digit}")
        except Exception as e:
            print(f"Failed to send prediction to mobile: {e}")
    
    def check_mobile_clear(self):
        """Check for clear signal from mobile"""
        try:
            response = requests.get(f"{self.server_url}/check_clear", timeout=1.0)
            if response.status_code == 200:
                data = response.json()
                if data.get('clear'):
                    print("📱 Clear signal received, clearing display")
                    # Clear the laptop display
                    self.latest_mobile_digit = None
                    self.activations = None
                    self.predictions = None
                    self.predicted_digit = None
                    self.confidence = 0
                    self.animation_time = 0
                    self.animation_phase = 0
                    self.processing_digit = False  # Reset processing flag
                    return True
        except Exception as e:
            print(f"Error checking clear signal: {e}")
        return False
    
    def check_mobile_input(self):
        """Check for new mobile digit input"""
        try:
            while not self.mobile_digit_queue.empty():
                digit_array = self.mobile_digit_queue.get_nowait()
                if not self.processing_digit:  # Only process if not already processing
                    self.latest_mobile_digit = digit_array
                    self.processing_digit = True
                    print(f"📱 Received new digit from mobile - Shape: {digit_array.shape}, Min: {digit_array.min():.2f}, Max: {digit_array.max():.2f}")
                    return True
        except:
            pass
        return False
        
    def draw_neuron_with_label(self, x, y, activation, radius, layer_type, neuron_id=None):
        # Color based on activation and layer type
        if layer_type == "input":
            intensity = min(255, max(0, int((activation + 2) * 64)))  # Normalize from [-2, 2] to [0, 255]
            color = (intensity, intensity, intensity)
        elif layer_type == "output":
            # Gradient from red to green based on activation
            intensity = min(255, int(activation * 255))
            color = (255 - intensity, intensity, 0)
        else:  # hidden layers
            intensity = min(255, max(0, int(activation * 255)))
            color = (intensity // 2, intensity // 2, 255 - intensity // 2)
        
        # Draw glow effect for high activations
        if activation > 0.3:
            glow_radius = radius + int(activation * 12)
            for i in range(3):
                alpha = 30 - i * 10
                # Ensure coordinates are valid integers
                center_x, center_y = int(x), int(y)
                current_radius = max(1, glow_radius - i * 3)
                pygame.draw.circle(self.screen, color, (center_x, center_y), current_radius, 1)
        
        # Draw neuron
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, self.WHITE, (x, y), radius, 2)
        
        # Add activation value for important neurons
        if layer_type == "output" or (layer_type == "hidden1" and neuron_id < 8):
            if activation > 0.1:
                text = self.font_tiny.render(f"{activation:.2f}", True, self.WHITE)
                text_rect = text.get_rect(center=(x + radius + 25, y))
                self.screen.blit(text, text_rect)
    
    def draw_educational_connections(self, layer1_pos, layer2_pos, size1, size2, activations1, activations2, progress=1.0, layer_name=""):
        if activations1 is None or activations2 is None:
            return
        
        # Determine connection color based on layer_name
        if layer_name == "input_to_hidden1":
            base_color = (0, 150, 255)  # BRIGHTER BLUE for Input → Hidden Layer 1
        elif layer_name == "hidden1_to_hidden2":
            base_color = (255, 255, 100)  # BRIGHTER YELLOW for Hidden Layer 1 → Hidden Layer 2
        elif layer_name == "hidden2_to_output":
            base_color = (100, 255, 100)  # BRIGHTER GREEN for Hidden Layer 2 → Output
        else:
            base_color = (200, 200, 200)  # Brighter default gray
        
        # Draw more connections for better visualization
        max_connections = 100  # Increased for more impressive animation
        step1 = max(1, size1 // 10)  # More connections from input layer
        step2 = max(1, size2 // 6)   # More connections between hidden layers
        
        for i in range(0, min(size1, len(activations1)), step1):
            for j in range(0, min(size2, len(activations2)), step2):
                x1, y1 = layer1_pos
                x2, y2 = layer2_pos
                
                spacing1 = 400 / max(size1, 1)
                spacing2 = 400 / max(size2, 1)
                
                neuron1_y = y1 + i * spacing1
                neuron2_y = y2 + j * spacing2
                
                # Connection strength based on activations
                # Use normalized activations for better visibility
                norm_act1 = min(1.0, abs(activations1[i]) * 10)  # Scale up for visibility
                norm_act2 = min(1.0, abs(activations2[j]) * 10)  # Scale up for visibility
                strength = norm_act1 * norm_act2 * progress
                
                # Always draw connections with some visibility
                if strength > 0.01 or progress > 0.1:  # Lower threshold and show during animation
                    alpha = min(255, max(50, int(strength * 255)))  # Minimum alpha for visibility
                    
                    # Line thickness reflects influence strength
                    thickness = max(2, min(6, int(strength * 6)))  # Thicker lines
                    
                    # Apply color with alpha
                    color = tuple(int(c * alpha / 255) for c in base_color)
                    
                    # Animated connection
                    if progress < 1.0:
                        end_x = x1 + (x2 - x1) * progress
                        end_y = neuron1_y + (neuron2_y - neuron1_y) * progress
                        pygame.draw.line(self.screen, color, (x1, neuron1_y), (end_x, end_y), thickness)
                    else:
                        pygame.draw.line(self.screen, color, (x1, neuron1_y), (x2, neuron2_y), thickness)
    
    def draw_input_layer_educational(self):
        # Draw title and explanation
        title = self.font_medium.render("Input Layer", True, self.WHITE)
        self.screen.blit(title, (self.input_x - 30, self.input_y - 60))
        
        if self.show_explanations:
            exp_lines = self.explanations["input"].split('\n')
            for i, line in enumerate(exp_lines):
                exp_text = self.font_tiny.render(line, True, self.CYAN)
                self.screen.blit(exp_text, (self.input_x - 20, self.input_y + 420 + i * 20))
        
        if self.activations is not None:
            # Display as 28x28 grayscale grid (colored pixels)
            input_activations = self.activations.reshape(28, 28)
            cell_size = 10
            
            for i in range(28):
                for j in range(28):
                    x = self.input_x + j * cell_size
                    y = self.input_y + i * cell_size
                    # Normalize from [0, 1] to [0, 255] (new preprocessing)
                    intensity = min(255, max(0, int(input_activations[i, j] * 255)))
                    color = (intensity, intensity, intensity)
                    pygame.draw.rect(self.screen, color, (x, y, cell_size - 1, cell_size - 1))
            
            # Add grid lines for clarity
            for i in range(29):
                pygame.draw.line(self.screen, self.GRAY, 
                               (self.input_x, self.input_y + i * cell_size),
                               (self.input_x + 28 * cell_size, self.input_y + i * cell_size), 1)
                pygame.draw.line(self.screen, self.GRAY,
                               (self.input_x + i * cell_size, self.input_y),
                               (self.input_x + i * cell_size, self.input_y + 28 * cell_size), 1)
            
            # Draw label
            label = self.font_small.render("28×28 Input Grid", True, self.GRAY)
            self.screen.blit(label, (self.input_x + 100, self.input_y + 300))
    
    def draw_hidden_layer_educational(self, x, y, activations, label, max_neurons, layer_type):
        # Draw title and explanation
        title = self.font_medium.render(label, True, self.WHITE)
        self.screen.blit(title, (x - 40, y - 60))
        
        if self.show_explanations and layer_type in self.explanations:
            exp_lines = self.explanations[layer_type].split('\n')
            for i, line in enumerate(exp_lines):
                exp_text = self.font_tiny.render(line, True, self.CYAN)
                self.screen.blit(exp_text, (x - 30, y + 420 + i * 20))
        
        if activations is not None:
            display_neurons = min(len(activations), max_neurons)
            spacing = 400 / max(display_neurons, 1)
            
            for i in range(display_neurons):
                neuron_x = x
                neuron_y = y + i * spacing
                activation = activations[i] if i < len(activations) else 0
                
                # Apply ReLU activation for visualization (max(0, activation))
                display_activation = max(0, activation)
                
                self.draw_neuron_with_label(neuron_x, neuron_y, display_activation, radius=8, 
                                          layer_type=layer_type, neuron_id=i)
    
    def draw_output_layer_educational(self):
        # Draw title and explanation
        title = self.font_medium.render("Output Layer", True, self.WHITE)
        self.screen.blit(title, (self.output_x - 40, self.output_y - 60))
        
        if self.show_explanations:
            exp_lines = self.explanations["output"].split('\n')
            for i, line in enumerate(exp_lines):
                exp_text = self.font_tiny.render(line, True, self.CYAN)
                self.screen.blit(exp_text, (self.output_x - 30, self.output_y + 420 + i * 20))
        
        if self.predictions is not None:
            spacing = 40
            bar_width = 100
            bar_height = 30
            
            for i in range(10):
                x = self.output_x
                y = self.output_y + i * spacing
                
                # Draw background
                pygame.draw.rect(self.screen, self.GRAY, (x + 30, y, bar_width, bar_height))
                
                # Draw prediction bar
                activation_width = int(self.predictions[i] * bar_width)
                if activation_width > 0:
                    # Color gradient
                    color_intensity = int(self.predictions[i] * 255)
                    color = (255 - color_intensity, color_intensity, 0)
                    pygame.draw.rect(self.screen, color, (x + 30, y, activation_width, bar_height))
                
                # Draw border
                pygame.draw.rect(self.screen, self.WHITE, (x + 30, y, bar_width, bar_height), 2)
                
                # Draw digit label
                digit_text = self.font_medium.render(str(i), True, self.WHITE)
                self.screen.blit(digit_text, (x, y + 3))
                
                # Draw percentage
                percent_text = self.font_small.render(f"{self.predictions[i]*100:.1f}%", True, self.WHITE)
                self.screen.blit(percent_text, (x + bar_width + 40, y + 5))
                
                # Highlight predicted digit
                if i == self.predicted_digit:
                    pygame.draw.rect(self.screen, self.YELLOW, (x - 15, y - 5, bar_width + 55, bar_height + 10), 3)
                    
                    # Add "PREDICTED" label
                    pred_text = self.font_tiny.render("PREDICTED", True, self.YELLOW)
                    self.screen.blit(pred_text, (x + bar_width + 40, y - 15))
    
    def preprocess_image(self):
        """Use mobile digit directly instead of canvas preprocessing"""
        if self.latest_mobile_digit is not None:
            # Mobile digit is already preprocessed by the server
            digit = self.latest_mobile_digit
            
            # Convert to tensor with correct dtype (float32)
            digit_tensor = torch.from_numpy(digit).float().unsqueeze(0).unsqueeze(0)
            
            return digit_tensor
        else:
            # Return empty tensor if no mobile input
            return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    
    def run_inference(self):
        if not self.model_ready:
            return
        
        # Prevent too frequent predictions
        current_time = time.time()
        if current_time - self.last_prediction_time < 1.0:
            return
        
        self.last_prediction_time = current_time
        
        # Preprocess image
        input_tensor = self.preprocess_image().to(self.device)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == "CNN":
                output, fc1_out, conv2_out = self.model(input_tensor)
            else:  # MLP
                output, hidden1, hidden2 = self.model(input_tensor)
                fc1_out = hidden1
                conv2_out = hidden2
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            
            # Store activations for visualization
            self.activations = input_tensor.cpu().numpy().flatten()
            self.predictions = probabilities.cpu().numpy().flatten()
            
            if self.model_type == "CNN":
                self.hidden1_activations = fc1_out.cpu().numpy().flatten()
                self.hidden2_activations = conv2_out.cpu().numpy().flatten()
            else:  # MLP
                self.hidden1_activations = hidden1.cpu().numpy().flatten()
                self.hidden2_activations = hidden2.cpu().numpy().flatten()
            
            # Get prediction
            self.predicted_digit = np.argmax(self.predictions)
            self.confidence = self.predictions[self.predicted_digit]
            
            # Start animation
            self.animation_time = time.time()
            self.animation_phase = 0
    
    def draw_prediction_display(self):
        if self.predicted_digit is not None:
            # Draw prediction box
            box_x, box_y = 600, 650
            box_width, box_height = 400, 150
            
            # Animated background
            if self.animation_phase > 0.8:
                pygame.draw.rect(self.screen, self.BLUE, (box_x, box_y, box_width, box_height))
            else:
                pygame.draw.rect(self.screen, (0, 50, 150), (box_x, box_y, box_width, box_height))
            
            pygame.draw.rect(self.screen, self.WHITE, (box_x, box_y, box_width, box_height), 4)
            
            # Draw predicted digit
            digit_text = self.font_large.render(str(self.predicted_digit), True, self.WHITE)
            text_rect = digit_text.get_rect(center=(box_x + box_width // 2, box_y + 50))
            self.screen.blit(digit_text, text_rect)
            
            # Draw confidence
            confidence_text = self.font_medium.render(f"Confidence: {self.confidence*100:.1f}%", True, self.WHITE)
            text_rect = confidence_text.get_rect(center=(box_x + box_width // 2, box_y + 100))
            self.screen.blit(confidence_text, text_rect)
            
            # Add explanation
            if self.confidence > 0.8:
                exp_text = self.font_small.render("High confidence - clear prediction!", True, self.GREEN)
            elif self.confidence > 0.5:
                exp_text = self.font_small.render("Moderate confidence - reasonably clear", True, self.YELLOW)
            else:
                exp_text = self.font_small.render("Low confidence - try drawing clearer", True, self.ORANGE)
            
            text_rect = exp_text.get_rect(center=(box_x + box_width // 2, box_y + 130))
            self.screen.blit(exp_text, text_rect)
    
    def draw_ui_elements(self):
        # Draw title
        title_text = self.font_title.render("MNIST Neural Network - Mobile Input Visualization", True, self.WHITE)
        title_rect = title_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_text, title_rect)
        
        # Draw mobile input display label
        mobile_text = self.font_medium.render("Mobile Input", True, self.WHITE)
        self.screen.blit(mobile_text, (self.mobile_display_x, self.mobile_display_y - 30))
        
        # Draw connection status
        if self.mobile_connected:
            status_text = "📱 Mobile: Connected"
            color = self.GREEN
        else:
            status_text = "📱 Mobile: Waiting for connection..."
            color = self.YELLOW
        
        status = self.font_small.render(status_text, True, color)
        self.screen.blit(status, (self.mobile_display_x, self.mobile_display_y + 300))
        
        # Draw instructions
        instructions = [
            "📱 Draw on mobile device",
            "SPACE: Force Predict", 
            "E: Toggle Explanations",
            "ESC: Exit"
        ]
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, self.WHITE)
            self.screen.blit(text, (50, 50 + i * 25))
        
        # Draw model status
        if self.model_ready:
            model_status = "✅ Model: Ready"
            model_color = self.GREEN
        else:
            model_status = "❌ Model: Not loaded"
            model_color = self.RED
        
        model_text = self.font_small.render(model_status, True, model_color)
        self.screen.blit(model_text, (50, 180))
        
        # Draw server info
        server_text = self.font_tiny.render(f"Server: {self.server_url}", True, self.CYAN)
        self.screen.blit(server_text, (50, 210))
        
        # Draw FPS and device info
        fps_text = self.font_small.render(f"FPS: {self.clock.get_fps():.1f}", True, self.WHITE)
        self.screen.blit(fps_text, (self.width - 120, 20))
        
        device_text = self.font_small.render(f"Device: {self.device}", True, self.WHITE)
        self.screen.blit(device_text, (self.width - 200, 50))
        
        # Draw explanation toggle status
        exp_status = "ON" if self.show_explanations else "OFF"
        exp_text = self.font_small.render(f"Explanations: {exp_status}", True, self.CYAN)
        self.screen.blit(exp_text, (self.width - 200, 80))
        
        # Draw connection explanation if animating
        if self.animation_phase > 0 and self.show_explanations:
            conn_exp = self.explanations["connections"].split('\n')
            for i, line in enumerate(conn_exp):
                exp_text = self.font_tiny.render(line, True, self.CYAN)
                self.screen.blit(exp_text, (self.width - 300, 120 + i * 18))
    
    def run(self):
        running = True
        
        while running:
            # Calculate animation progress
            if self.animation_time:
                elapsed = time.time() - self.animation_time
                if elapsed < 6.0:  # Slower animation for better visibility
                    self.animation_phase = elapsed / 6.0
                else:
                    self.animation_phase = 1.0
            
            # Check for mobile input
            if self.check_mobile_input():
                # Automatically run inference when new mobile digit arrives
                self.run_inference()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:  # Force prediction
                        self.run_inference()
                    elif event.key == pygame.K_e:  # Toggle explanations
                        self.show_explanations = not self.show_explanations
            
            # Clear screen
            self.screen.fill(self.BLACK)
            
            # Draw mobile input display
            if self.latest_mobile_digit is not None:
                # Display the processed mobile digit
                digit_display = self.latest_mobile_digit.copy()
                # Normalize for display (from [0, 1] to [0, 255])
                digit_display = (digit_display * 255).astype(np.uint8)
                digit_display = np.clip(digit_display, 0, 255)
                
                # Convert to pygame surface
                digit_surface = pygame.surfarray.make_surface(digit_display)
                digit_surface = pygame.transform.scale(digit_surface, (self.mobile_display_size, self.mobile_display_size))
                self.screen.blit(digit_surface, (self.mobile_display_x, self.mobile_display_y))
                
                # Add label
                label = self.font_small.render("Processed Input", True, self.WHITE)
                self.screen.blit(label, (self.mobile_display_x, self.mobile_display_y - 30))
            else:
                # Draw placeholder
                pygame.draw.rect(self.screen, self.WHITE, 
                               (self.mobile_display_x, self.mobile_display_y, 
                                self.mobile_display_size, self.mobile_display_size))
                pygame.draw.rect(self.screen, self.GRAY, 
                               (self.mobile_display_x, self.mobile_display_y, 
                                self.mobile_display_size, self.mobile_display_size), 3)
                
                # Draw waiting text
                wait_text = self.font_small.render("Waiting for mobile input...", True, self.GRAY)
                wait_rect = wait_text.get_rect(center=(self.mobile_display_x + self.mobile_display_size // 2,
                                                  self.mobile_display_y + self.mobile_display_size // 2))
                self.screen.blit(wait_text, wait_rect)
                
                # Add label
                label = self.font_small.render("Mobile Input", True, self.WHITE)
                self.screen.blit(label, (self.mobile_display_x, self.mobile_display_y - 30))
            
            pygame.draw.rect(self.screen, self.GRAY, 
                           (self.mobile_display_x, self.mobile_display_y, 
                            self.mobile_display_size, self.mobile_display_size), 3)
            
            # Draw neural network layers with educational content
            self.draw_input_layer_educational()
            self.draw_hidden_layer_educational(self.hidden1_x, self.hidden1_y, 
                                              self.hidden1_activations if hasattr(self, 'hidden1_activations') else None,
                                              "Hidden Layer 1", max_neurons=32, layer_type="hidden1")
            self.draw_hidden_layer_educational(self.hidden2_x, self.hidden2_y,
                                              self.hidden2_activations if hasattr(self, 'hidden2_activations') else None,
                                              "Hidden Layer 2", max_neurons=16, layer_type="hidden2")
            self.draw_output_layer_educational()
            
            # Draw animated connections with proper timing
            if self.animation_phase > 0:
                # Step 1: Input to Hidden Layer 1 (0-2s)
                progress1 = min(1.0, self.animation_phase * 3.0)  # Complete by 0.33
                
                # Always draw Step 1 once it starts
                if self.animation_phase > 0:
                    # Create multiple connection points from the grid for better visual sync
                    grid_points = []
                    for row in range(0, 28, 4):  # Sample every 4th row
                        for col in range(0, 28, 4):  # Sample every 4th column
                            grid_x = self.input_x + col * 10 + 5  # Center of grid cell
                            grid_y = self.input_y + row * 10 + 5  # Center of grid cell
                            grid_points.append((grid_x, grid_y))
                    
                    # Draw connections from multiple grid points
                    for grid_x, grid_y in grid_points[:49]:  # Limit to 49 points (7x7)
                        self.draw_educational_connections(
                                (grid_x, grid_y),  # Individual grid point
                                (self.hidden1_x, self.hidden1_y),
                                1,  # Single point
                                128,  # Hidden layer 1 size
                                np.array([1.0]),  # Single activation
                                self.hidden1_activations if hasattr(self, 'hidden1_activations') else np.zeros(128),
                                progress=progress1,
                                layer_name="input_to_hidden1"
                            )
                
                # Step 2: Hidden Layer 1 to Hidden Layer 2 (2s-4s) - Starts after Step 1 begins
                if self.animation_phase > 0.33:
                    progress2 = min(1.0, (self.animation_phase - 0.33) * 3.0)  # Start at 0.33, complete by 0.67
                    self.draw_educational_connections(
                            (self.hidden1_x, self.hidden1_y),
                            (self.hidden2_x, self.hidden2_y),
                            128,  # Hidden layer 1 size (matches CNN fc1)
                            64,   # Hidden layer 2 size (matches CNN conv2 features)
                            self.hidden1_activations if hasattr(self, 'hidden1_activations') else np.zeros(128),
                            self.hidden2_activations if hasattr(self, 'hidden2_activations') else np.zeros(64),
                            progress=progress2,
                            layer_name="hidden1_to_hidden2"
                        )
                
                # Step 3: Hidden Layer 2 to Output (4s-6s) - Starts after Step 2 begins
                if self.animation_phase > 0.67:
                    progress3 = min(1.0, (self.animation_phase - 0.67) * 3.0)  # Start at 0.67, complete by 1.0
                    self.draw_educational_connections(
                            (self.hidden2_x, self.hidden2_y),
                            (self.output_x, self.output_y),
                            64,   # Hidden layer 2 size (matches CNN conv2 features)
                            10,   # Output layer size
                            self.hidden2_activations if hasattr(self, 'hidden2_activations') else np.zeros(64),
                            self.predictions if self.predictions is not None else np.zeros(10),
                            progress=progress3,
                            layer_name="hidden2_to_output"
                        )
                
                # Show output digit and top-3 contributors only after all animations complete
                if self.animation_phase >= 1.0:
                    # Highlight top-3 contributing hidden layer 2 neurons (NO thick lines)
                    if hasattr(self, 'hidden2_activations') and self.predictions is not None:
                        # Contribution proxy: activation × confidence of predicted digit
                        predicted_digit = np.argmax(self.predictions)
                        confidence = self.predictions[predicted_digit]
                        
                        contributions = np.abs(self.hidden2_activations) * confidence
                        
                        top_k = 3
                        top_indices = np.argsort(contributions)[-top_k:]
                        
                        hidden2_spacing = 400 / 64  # Updated to match actual layer size
                        
                        for idx in top_indices:
                            y = self.hidden2_y + idx * hidden2_spacing
                            y = max(self.hidden2_y, min(self.hidden2_y + 400, y))
                            
                            pulse = abs(math.sin(time.time() * 2.5))
                            green = int(180 + 75 * pulse)
                            
                            glow_color = (0, green, 0)
                            
                            # Soft glow rings
                            for r in range(12, 20, 3):
                                pygame.draw.circle(
                                    self.screen,
                                    glow_color,
                                    (int(self.hidden2_x), int(y)),
                                    r,
                                    2
                                )
                            
                            # Core highlight
                            pygame.draw.circle(
                                self.screen,
                                (0, 255, 0),
                                (int(self.hidden2_x), int(y)),
                                8
                            )
                
                # Draw UI elements
            self.draw_ui_elements()
            
            # Draw prediction display
            self.draw_prediction_display()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        pygame.quit()

if __name__ == "__main__":
    # Get server URL from command line or use default
    server_url = "http://localhost:5000"
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--server="):
            server_url = arg.split("=")[1]
            break
    
    print(f"🚀 Starting MNIST Mobile Visualization")
    print(f"📡 Server URL: {server_url}")
    print(f"📱 Mobile interface: {server_url}/")
    print(f"💡 On mobile device, open the URL above to start drawing")
    
    visualizer = EducationalNeuralNetworkVisualizer(server_url=server_url)
    visualizer.run()
