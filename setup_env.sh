#!/bin/bash
# Setup Python virtual environment for Image Matching Project
# Linux/Mac version

echo "============================================================"
echo "Image Matching Project - Environment Setup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.7+ first"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n) " recreate
    if [ "$recreate" = "y" ]; then
        echo "Removing old virtual environment..."
        rm -rf venv
    else
        echo "Keeping existing environment."
        skip_create=1
    fi
fi

# Create virtual environment
if [ -z "$skip_create" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        echo "Make sure you have python3-venv installed"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        echo "  CentOS/RHEL: sudo yum install python3-venv"
        exit 1
    fi
    
    echo "Virtual environment created successfully!"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo ""
echo "============================================================"
echo "Installing packages..."
echo "============================================================"
echo ""
echo "Choose installation type:"
echo "  1. Traditional only (lightweight, ~100MB)"
echo "  2. TensorFlow (full features, ~500MB)"
echo "  3. Both (recommended, ~500MB)"
echo ""
read -p "Enter choice (1/2/3): " install_choice

case $install_choice in
    1)
        echo "Installing traditional packages only..."
        pip install -r requirements.txt
        ;;
    2)
        echo "Installing TensorFlow packages..."
        pip install -r requirements_tf.txt
        ;;
    *)
        echo "Installing all packages..."
        pip install -r requirements.txt
        pip install -r requirements_tf.txt
        ;;
esac

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some packages failed to install"
    echo "Try running: pip install -r requirements_tf.txt"
fi

# Verify installation
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"
echo ""
python verify_offline.py

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Virtual environment created in: venv/"
echo ""
echo "To use the project:"
echo "  1. Activate environment: source activate_env.sh"
echo "  2. Run your scripts: python image_matcher.py image1.jpg image2.jpg"
echo "  3. Deactivate when done: deactivate"
echo ""
echo "Quick start:"
echo "  source activate_env.sh"
echo "  python image_matcher.py image1.jpg image2.jpg"
echo ""

