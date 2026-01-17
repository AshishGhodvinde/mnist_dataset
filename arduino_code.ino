// Arduino code for MNIST LED visualization
// Upload this to your Arduino before running the Python visualization

const int ledPins[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Pins for digits 0-9

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 10; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
  
  // Test LEDs - blink all once
  for (int i = 0; i < 10; i++) {
    digitalWrite(ledPins[i], HIGH);
    delay(100);
    digitalWrite(ledPins[i], LOW);
  }
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    
    if (data.startsWith("H")) {
      // Highlight mode - single digit
      int digit = data.substring(1).toInt();
      if (digit >= 0 && digit <= 9) {
        for (int i = 0; i < 10; i++) {
          digitalWrite(ledPins[i], i == digit ? HIGH : LOW);
        }
      }
    } else {
      // Probability mode - all LEDs with varying brightness
      int values[10];
      int index = 0;
      
      // Parse comma-separated values
      char *token = strtok(&data[0], ",");
      while (token != NULL && index < 10) {
        values[index] = atoi(token);
        token = strtok(NULL, ",");
        index++;
      }
      
      // Set LED brightness (0-255)
      for (int i = 0; i < 10; i++) {
        analogWrite(ledPins[i], values[i]);
      }
    }
  }
}
