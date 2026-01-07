#!/bin/bash
# Quick activation script for Linux/Mac

if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: bash setup_env.sh"
    return 1
fi

echo "Activating Image Matching environment..."
source venv/bin/activate

echo ""
echo "============================================================"
echo "Image Matching Environment Active"
echo "============================================================"
echo ""
echo "You can now run:"
echo "  python image_matcher.py image1.jpg image2.jpg"
echo "  python tf_image_matcher.py image1.jpg image2.jpg"
echo "  python test_tf_matcher.py"
echo ""
echo "To deactivate: deactivate"
echo ""

