"""
Verify that everything is ready for offline use
"""

import os
import sys


def verify_offline_setup():
    """Verify that everything is ready for offline use"""
    
    print("=" * 60)
    print("Verifying Offline Setup")
    print("=" * 60)
    
    all_ok = True
    
    # Check Python packages
    print("\n1. Python Packages:")
    print("-" * 60)
    
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'skimage': 'scikit-image',
        'PIL': 'Pillow',
        'tqdm': 'tqdm'
    }
    
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:20s} (version: {version})")
        except ImportError:
            print(f"  ✗ {name:20s} - MISSING")
            all_ok = False
    
    # Check pretrained models
    print("\n2. Pretrained Models:")
    print("-" * 60)
    
    models_dir = 'pretrained_models'
    if os.path.exists(models_dir):
        models = ['efficientnetb0.h5', 'resnet50.h5', 'vgg16.h5', 'mobilenetv2.h5']
        for model in models:
            path = os.path.join(models_dir, model)
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✓ {model:25s} ({size:6.1f} MB)")
            else:
                print(f"  ⚠ {model:25s} - Not found (optional)")
    else:
        print(f"  ⚠ {models_dir}/ directory not found")
        print(f"    Run 'python download_models_offline.py' to download")
        print(f"    Or use traditional method (image_matcher.py) which doesn't need models")
    
    # Check scripts
    print("\n3. Essential Scripts:")
    print("-" * 60)
    
    scripts = {
        'tf_image_matcher_offline.py': 'Deep learning matcher (offline)',
        'image_matcher.py': 'Traditional matcher (always works)',
        'download_models_offline.py': 'Model downloader',
        'requirements_tf.txt': 'Package requirements'
    }
    
    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"  ✓ {script:30s} - {description}")
        else:
            print(f"  ✗ {script:30s} - MISSING")
            all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if all_ok:
        print("✓ All essential components are present!")
        print("\nYou can use:")
        print("  python image_matcher.py image1.jpg image2.jpg")
        
        if os.path.exists(models_dir):
            print("  python tf_image_matcher_offline.py image1.jpg image2.jpg")
    else:
        print("⚠ Some components are missing")
        print("\nTo fix:")
        print("  1. Install packages: pip install -r requirements_tf.txt")
        print("  2. Download models: python download_models_offline.py")
    
    # Offline readiness
    print("\n" + "=" * 60)
    print("Offline Readiness:")
    print("=" * 60)
    
    has_packages = all(package in sys.modules or __import__(package, fromlist=['']) for package in ['cv2', 'numpy'])
    has_models = os.path.exists(models_dir) and os.path.exists(os.path.join(models_dir, 'efficientnetb0.h5'))
    
    if has_packages:
        print("✓ Can run traditional matcher OFFLINE (image_matcher.py)")
    else:
        print("✗ Need to install packages first")
    
    if has_packages and has_models:
        print("✓ Can run deep learning matcher OFFLINE (tf_image_matcher_offline.py)")
    elif has_packages:
        print("⚠ Deep learning needs pretrained models (download_models_offline.py)")
    else:
        print("✗ Need packages and models for deep learning matcher")
    
    print("\n" + "=" * 60)
    
    return all_ok


if __name__ == "__main__":
    success = verify_offline_setup()
    sys.exit(0 if success else 1)

