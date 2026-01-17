@echo off
echo MNIST Neural Network Visualization - Exhibition Setup
echo ====================================================

echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo Training neural network model...
python train_model.py
if errorlevel 1 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To run the visualization:
echo   python educational_visualization.py
echo.
echo To run with Arduino support:
echo   python educational_visualization.py --arduino
echo.
echo Press any key to start the visualization...
pause

python educational_visualization.py
