import serial
import time
import numpy as np

class ArduinoController:
    def __init__(self, port='COM7', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        
    def connect(self):
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.connected = False
            print("Disconnected from Arduino")
    
    def send_predictions(self, predictions):
        """Send prediction probabilities to Arduino for LED visualization"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            # Create a simple protocol: send 10 values (0-255) for digits 0-9
            # Scale predictions to 0-255 range
            scaled_predictions = [int(p * 255) for p in predictions]
            
            # Send as comma-separated values followed by newline
            data_string = ','.join(map(str, scaled_predictions)) + '\n'
            self.serial_connection.write(data_string.encode())
            
            # Wait for Arduino to process
            time.sleep(0.1)
            
            return True
        except Exception as e:
            print(f"Error sending data to Arduino: {e}")
            return False
    
    def send_digit_highlight(self, digit):
        """Send a single digit to highlight on Arduino"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            # Send command to highlight specific digit
            command = f"H{digit}\n"
            self.serial_connection.write(command.encode())
            time.sleep(0.05)
            return True
        except Exception as e:
            print(f"Error sending highlight command: {e}")
            return False

# Example Arduino code (for reference):
"""
// Arduino code for LED visualization
const int ledPins[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Pins for digits 0-9

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 10; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    
    if (data.startsWith("H")) {
      // Highlight mode - single digit
      int digit = data.substring(1).toInt();
      for (int i = 0; i < 10; i++) {
        digitalWrite(ledPins[i], i == digit ? HIGH : LOW);
      }
    } else {
      // Probability mode - all LEDs with varying brightness
      int values[10];
      int index = 0;
      char *token = strtok(&data[0], ",");
      
      while (token != NULL && index < 10) {
        values[index] = atoi(token);
        token = strtok(NULL, ",");
        index++;
      }
      
      for (int i = 0; i < 10; i++) {
        analogWrite(ledPins[i], values[i]);
      }
    }
  }
}
"""
