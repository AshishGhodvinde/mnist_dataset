@echo off
echo ========================================
echo   MNIST Mobile Visualization
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting mobile server...
start "Mobile Server" python mobile_server.py

echo Waiting for server to start...
timeout /t 3

echo.
echo Starting visualization...
echo.
echo Mobile interface will be available at: http://YOUR_LAPTOP_IP:5000
echo Replace YOUR_LAPTOP_IP with your laptop's local IP address
echo.
python educational_visualization.py

pause
