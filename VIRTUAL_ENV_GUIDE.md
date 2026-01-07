# Virtual Environment Guide

Complete guide for using the self-contained Python environment for this project.

## 🎯 What is a Virtual Environment?

A virtual environment is an **isolated Python installation** that:
- ✅ Keeps project dependencies separate from system Python
- ✅ Prevents version conflicts with other projects
- ✅ Makes the project portable and reproducible
- ✅ Easy to delete and recreate

## 🚀 Quick Start

### Windows

```bash
# 1. Setup (one time)
setup_env.bat

# 2. Use the project
activate_env.bat
python image_matcher.py image1.jpg image2.jpg

# Or use quick run scripts
run_traditional.bat image1.jpg image2.jpg
run_tensorflow.bat image1.jpg image2.jpg
```

### Linux/Mac

```bash
# 1. Setup (one time)
bash setup_env.sh

# 2. Use the project
source activate_env.sh
python image_matcher.py image1.jpg image2.jpg

# 3. Deactivate when done
deactivate
```

## 📁 Virtual Environment Structure

After setup, you'll have:

```
ImageMatching/
├── venv/                          # Virtual environment folder
│   ├── Scripts/                   # (Windows)
│   ├── bin/                       # (Linux/Mac)
│   ├── Lib/                       # Installed packages
│   └── pyvenv.cfg                 # Config file
├── setup_env.bat                  # Setup script (Windows)
├── setup_env.sh                   # Setup script (Linux/Mac)
├── activate_env.bat               # Quick activate (Windows)
├── activate_env.sh                # Quick activate (Linux/Mac)
├── run_traditional.bat            # Run with auto-activation
├── run_tensorflow.bat             # Run TensorFlow with auto-activation
└── [project files...]
```

## 🔧 Detailed Setup Instructions

### Step 1: Initial Setup

**Windows:**
```bash
setup_env.bat
```

**Linux/Mac:**
```bash
bash setup_env.sh
# or
chmod +x setup_env.sh
./setup_env.sh
```

This will:
1. Create virtual environment in `venv/` folder
2. Activate the environment
3. Install all required packages
4. Verify the installation

### Step 2: Choose Installation Type

During setup, you'll be asked:

```
Choose installation type:
  1. Traditional only (lightweight, ~100MB)
  2. TensorFlow (full features, ~500MB)
  3. Both (recommended, ~500MB)
```

**Recommendations:**
- **Option 1**: If you only need basic matching (faster setup)
- **Option 2**: If you only need deep learning matching
- **Option 3**: Best choice - gives you all options

### Step 3: Activation

Every time you want to use the project:

**Windows:**
```bash
activate_env.bat
```

**Linux/Mac:**
```bash
source activate_env.sh
```

**You'll see:**
```
============================================================
Image Matching Environment Active
============================================================

You can now run:
  python image_matcher.py image1.jpg image2.jpg
  python tf_image_matcher.py image1.jpg image2.jpg

To deactivate: deactivate
```

### Step 4: Use the Project

Once activated, run any Python script:

```bash
# Traditional matching
python image_matcher.py camera1.jpg camera2.jpg

# TensorFlow matching
python tf_image_matcher.py camera1.jpg camera2.jpg hybrid

# Test the matcher
python test_tf_matcher.py

# Train a model
python train_regressor.py --epochs 30
```

### Step 5: Deactivate

When you're done:

```bash
deactivate
```

## 🎮 Quick Run Scripts (No Manual Activation)

For convenience, use these scripts that **automatically activate** the environment:

### Windows

```bash
# Traditional matching
run_traditional.bat image1.jpg image2.jpg

# TensorFlow matching
run_tensorflow.bat image1.jpg image2.jpg
```

### Linux/Mac (create similar scripts)

```bash
#!/bin/bash
source venv/bin/activate
python image_matcher.py $1 $2
```

## 💡 Common Commands

### Check What's Installed

```bash
# Activate environment first
activate_env.bat  # (Windows)
source activate_env.sh  # (Linux/Mac)

# List installed packages
pip list

# Check specific package
pip show tensorflow
pip show opencv-python
```

### Install Additional Packages

```bash
# Activate environment
activate_env.bat

# Install package
pip install some-package

# Save to requirements
pip freeze > requirements_custom.txt
```

### Update Packages

```bash
# Activate environment
activate_env.bat

# Update specific package
pip install --upgrade tensorflow

# Update all packages
pip install --upgrade -r requirements_tf.txt
```

### Recreate Environment

If something goes wrong:

**Windows:**
```bash
# Delete old environment
rmdir /s /q venv

# Create new one
setup_env.bat
```

**Linux/Mac:**
```bash
# Delete old environment
rm -rf venv

# Create new one
bash setup_env.sh
```

## 🌐 Offline Usage with Virtual Environment

### Prepare Offline Package

**While online:**
```bash
# 1. Setup environment
setup_env.bat

# 2. Download pretrained models
activate_env.bat
python download_models_offline.py

# 3. Download packages for offline install
pip download -r requirements_tf.txt -d offline_packages/
```

### Use Offline

**On offline machine:**
```bash
# 1. Copy entire project folder (including venv/)
# 2. Just activate and use!
activate_env.bat
python tf_image_matcher_offline.py image1.jpg image2.jpg
```

Or recreate environment offline:
```bash
# Install from offline packages
python -m venv venv
venv\Scripts\activate
pip install --no-index --find-links=offline_packages/ -r requirements_tf.txt
```

## 📊 Virtual Environment Benefits

| Aspect | Without venv | With venv |
|--------|-------------|-----------|
| **Isolation** | ❌ Affects system Python | ✅ Isolated |
| **Dependencies** | ❌ Global conflicts | ✅ Project-specific |
| **Portability** | ❌ Hard to reproduce | ✅ Easy to share |
| **Cleanup** | ❌ Hard to remove | ✅ Just delete venv/ |
| **Multiple Projects** | ❌ Version conflicts | ✅ Each has own env |

## 🔍 Troubleshooting

### "venv not found"

**Windows:**
```bash
python -m venv venv
```

**Linux:**
```bash
# Install venv module
sudo apt install python3-venv  # Ubuntu/Debian
sudo yum install python3-venv  # CentOS/RHEL

# Create environment
python3 -m venv venv
```

### "Command not found" after activation

Make sure you activated correctly:

**Windows:**
```bash
call venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### "Module not found" error

```bash
# Activate environment first!
activate_env.bat

# Then verify installation
python verify_offline.py

# Reinstall if needed
pip install -r requirements_tf.txt
```

### Slow installation

```bash
# Use faster mirrors
pip install -r requirements_tf.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or install one by one
pip install opencv-python
pip install numpy
pip install tensorflow
```

## 🎯 Best Practices

### 1. Always Activate Before Use

```bash
# ✓ Good
activate_env.bat
python image_matcher.py image1.jpg image2.jpg

# ✗ Bad (might use wrong Python)
python image_matcher.py image1.jpg image2.jpg
```

### 2. Keep venv/ in .gitignore

The `venv/` folder should NOT be committed to git:

```gitignore
# .gitignore
venv/
__pycache__/
*.pyc
```

### 3. Use Requirements Files

Always maintain `requirements.txt`:

```bash
# After installing new packages
pip freeze > requirements_current.txt
```

### 4. Test in Clean Environment

Before sharing:

```bash
# Delete environment
rm -rf venv

# Recreate from requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements_tf.txt

# Test
python image_matcher.py test1.jpg test2.jpg
```

## 📦 Portable Virtual Environment

To make the environment truly portable:

### Option 1: Include venv/ folder

```bash
# Create package
ImageMatching_Complete/
├── venv/              # Include entire virtual environment
├── *.py              # Python scripts
└── README.md

# On another machine with same OS/architecture:
cd ImageMatching_Complete
activate_env.bat  # Just activate and use!
```

**Pros:** Works immediately on target machine
**Cons:** Large size (~1GB), OS-specific

### Option 2: Use requirements.txt

```bash
# Create package
ImageMatching_Portable/
├── requirements_tf.txt
├── *.py
└── setup_env.bat

# On target machine:
setup_env.bat  # Creates and installs everything
```

**Pros:** Small size, cross-platform
**Cons:** Needs internet or offline_packages/

## 🚀 Advanced: Multiple Environments

For different use cases:

```bash
# Create different environments
python -m venv venv_traditional    # Traditional only
python -m venv venv_tensorflow     # TensorFlow only
python -m venv venv_dev            # Development with extra tools

# Use specific environment
venv_traditional\Scripts\activate  # Use traditional
venv_tensorflow\Scripts\activate   # Use TensorFlow
```

## ✅ Verification Checklist

After setup, verify:

```bash
# 1. Activate environment
activate_env.bat

# 2. Check Python location (should be in venv/)
where python              # Windows
which python              # Linux/Mac

# 3. Verify packages
python verify_offline.py

# 4. Test functionality
python test_tf_matcher.py
```

Expected output:
```
✓ All packages installed correctly
✓ Virtual environment active
✓ Ready to use!
```

---

**You now have a self-contained, portable Python environment!** 🎉

Use `activate_env.bat` every time you want to work on this project.

