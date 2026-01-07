# Offline Setup Guide

Complete guide for running the image matching project **completely offline** without internet connection.

## 📦 What You Need to Download

### 1. Python Packages (One-time Download)
All packages from `requirements.txt` and `requirements_tf.txt`

### 2. Pretrained TensorFlow Models (~200MB)
- EfficientNetB0 (~29 MB)
- ResNet50 (~98 MB)
- VGG16 (~58 MB)
- MobileNetV2 (~14 MB)

## 🔧 Step-by-Step Offline Setup

### Step 1: Download Everything (While Online)

```bash
# 1. Install Python packages and download them
pip download -r requirements_tf.txt -d offline_packages/

# 2. Download pretrained models
python download_models_offline.py

# 3. Verify models were downloaded
python download_models_offline.py --verify
```

### Step 2: Package for Offline Use

Create a folder with everything needed:

```
ImageMatching_Offline/
├── offline_packages/          # Python packages (from pip download)
├── pretrained_models/          # TensorFlow models
│   ├── efficientnetb0/
│   ├── efficientnetb0.h5
│   ├── resnet50/
│   ├── resnet50.h5
│   ├── vgg16/
│   ├── vgg16.h5
│   ├── mobilenetv2/
│   └── mobilenetv2.h5
├── *.py                       # All Python scripts
├── requirements_tf.txt
└── README.md
```

### Step 3: Transfer to Offline Machine

1. Copy the entire `ImageMatching_Offline/` folder to offline machine
2. Use USB drive, network transfer, or any available method

### Step 4: Install on Offline Machine

```bash
# Navigate to the folder
cd ImageMatching_Offline

# Install packages from offline directory
pip install --no-index --find-links=offline_packages/ -r requirements_tf.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### Step 5: Use Offline

```bash
# Use the offline version
python tf_image_matcher_offline.py image1.jpg image2.jpg

# Or use traditional method (no models needed)
python image_matcher.py image1.jpg image2.jpg
```

## 🚀 Quick Commands

### Online Machine (One-time Setup)

```bash
# Download all Python packages
pip download tensorflow==2.4.0 opencv-python==4.5.5.64 numpy==1.19.5 matplotlib==3.3.4 scikit-image==0.17.2 h5py==2.10.0 pillow==8.4.0 tqdm==4.64.1 -d offline_packages/

# Download pretrained models
python download_models_offline.py
```

### Offline Machine (Every Use)

```bash
# Install once
pip install --no-index --find-links=offline_packages/ tensorflow opencv-python numpy matplotlib

# Use anytime
python tf_image_matcher_offline.py camera1.jpg camera2.jpg
```

## 📁 Files Needed for Offline Use

### Essential Files (Must Have)

```
✓ tf_image_matcher_offline.py   # Offline deep learning matcher
✓ image_matcher.py               # Traditional matcher (no models needed)
✓ pretrained_models/             # Downloaded model weights
✓ offline_packages/              # Python packages
✓ requirements_tf.txt            # Package list
```

### Optional Files

```
○ train_regressor.py            # For training (needs lots of data)
○ test_tf_matcher.py            # Testing script
○ example_usage.py              # Examples
○ compare_feature_methods.py    # Comparison tool
```

## 🎯 Usage Examples

### Traditional Method (No Models Needed) ⚡

```bash
# Fastest, works offline immediately
python image_matcher.py image1.jpg image2.jpg
```

**Pros:**
- No pretrained models needed
- Works immediately
- Small file size

**Cons:**
- Lower accuracy
- Less robust to lighting changes

### Deep Learning Method (Offline) 🎯

```bash
# Best accuracy, needs pretrained models
python tf_image_matcher_offline.py image1.jpg image2.jpg
```

**Pros:**
- High accuracy
- Robust to lighting/color differences
- Best for dual-camera setup

**Cons:**
- Needs pretrained models (~200MB)
- Requires TensorFlow

## 🔍 Verify Offline Setup

Create `verify_offline.py`:

```python
import os
import sys

def verify_offline_setup():
    """Verify that everything is ready for offline use"""
    
    print("Verifying offline setup...")
    print("=" * 60)
    
    # Check Python packages
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    print("\nPython Packages:")
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
    
    # Check pretrained models
    print("\nPretrained Models:")
    models_dir = 'pretrained_models'
    if os.path.exists(models_dir):
        models = ['efficientnetb0.h5', 'resnet50.h5', 'vgg16.h5', 'mobilenetv2.h5']
        for model in models:
            path = os.path.join(models_dir, model)
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✓ {model} ({size:.1f} MB)")
            else:
                print(f"  ✗ {model} - MISSING")
    else:
        print(f"  ✗ {models_dir}/ directory not found")
    
    # Check scripts
    print("\nEssential Scripts:")
    scripts = [
        'tf_image_matcher_offline.py',
        'image_matcher.py',
        'download_models_offline.py'
    ]
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - MISSING")
    
    print("\n" + "=" * 60)
    print("Verification complete!")

if __name__ == "__main__":
    verify_offline_setup()
```

Run verification:
```bash
python verify_offline.py
```

## 💾 Disk Space Requirements

| Component | Size | Required |
|-----------|------|----------|
| Python packages | ~500 MB | Yes |
| Pretrained models | ~200 MB | For deep learning |
| Project scripts | ~1 MB | Yes |
| **Total** | **~700 MB** | |

## 🛠️ Troubleshooting

### "No module named 'tensorflow'"

```bash
# Install from offline packages
pip install --no-index --find-links=offline_packages/ tensorflow
```

### "Model file not found"

```bash
# Re-download models (while online)
python download_models_offline.py
```

### "Cannot connect to internet"

This is expected! The offline version shouldn't need internet.
If you see this error, make sure you're using:
- `tf_image_matcher_offline.py` (not `tf_image_matcher.py`)
- Models are in `pretrained_models/` directory

### Use traditional method as fallback

```bash
# Works without any pretrained models
python image_matcher.py image1.jpg image2.jpg
```

## 📦 Complete Offline Package Script

Create `create_offline_package.py`:

```python
"""Create complete offline package"""
import os
import shutil
import subprocess

def create_offline_package():
    print("Creating offline package...")
    
    # Create package directory
    package_dir = 'ImageMatching_Offline'
    os.makedirs(package_dir, exist_ok=True)
    
    # Download packages
    print("\n1. Downloading Python packages...")
    subprocess.run([
        'pip', 'download',
        '-r', 'requirements_tf.txt',
        '-d', os.path.join(package_dir, 'offline_packages')
    ])
    
    # Download models
    print("\n2. Downloading pretrained models...")
    subprocess.run(['python', 'download_models_offline.py'])
    
    # Copy files
    print("\n3. Copying project files...")
    files_to_copy = [
        'tf_image_matcher_offline.py',
        'image_matcher.py',
        'requirements_tf.txt',
        'OFFLINE_SETUP.md',
        'README.md'
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, package_dir)
    
    # Copy models
    if os.path.exists('pretrained_models'):
        shutil.copytree('pretrained_models', 
                       os.path.join(package_dir, 'pretrained_models'),
                       dirs_exist_ok=True)
    
    print(f"\n✓ Offline package created: {package_dir}/")
    print("\nTransfer this folder to offline machine")

if __name__ == "__main__":
    create_offline_package()
```

Run:
```bash
python create_offline_package.py
```

## ✅ Checklist for Offline Use

**Before going offline:**
- [ ] Downloaded all Python packages (`pip download`)
- [ ] Downloaded pretrained models (`download_models_offline.py`)
- [ ] Verified models (`download_models_offline.py --verify`)
- [ ] Tested scripts work with downloaded models
- [ ] Packaged everything into one folder
- [ ] Created backup copy

**On offline machine:**
- [ ] Copied folder to offline machine
- [ ] Installed packages from offline_packages/
- [ ] Verified installation (`verify_offline.py`)
- [ ] Tested with sample images
- [ ] Ready to use!

## 🎯 Which Method to Use Offline?

### For Maximum Convenience
```bash
python image_matcher.py image1.jpg image2.jpg
```
- No setup needed
- Works immediately
- Good for simple cases

### For Maximum Accuracy
```bash
python tf_image_matcher_offline.py image1.jpg image2.jpg
```
- Requires pretrained models
- Best accuracy
- Recommended for dual-camera setup

---

**Both methods work completely offline after initial setup!** 🚀

