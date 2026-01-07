"""
Download all necessary pretrained models for offline usage
Run this script ONCE while online, then you can use the project offline
"""

import tensorflow as tf
from tensorflow import keras
import os
import sys


def download_model(model_name, model_class, input_shape=(512, 512, 3)):
    """
    Download and save a pretrained model
    
    Args:
        model_name: Name for saving
        model_class: Keras model class
        input_shape: Input shape
    """
    print(f"\nDownloading {model_name}...")
    print("-" * 60)
    
    try:
        # Create models directory
        models_dir = 'pretrained_models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Download model
        print(f"Fetching {model_name} with ImageNet weights...")
        model = model_class(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        # Save model
        save_path = os.path.join(models_dir, f'{model_name}.h5')
        model.save(save_path)
        
        # Also save in Keras format (more portable)
        saved_model_path = os.path.join(models_dir, f'{model_name}.keras')
        model.save(saved_model_path)
        
        print(f"[OK] Downloaded and saved to:")
        print(f"  - {save_path}")
        print(f"  - {saved_model_path}")
        
        # Get model size
        if os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Failed to download {model_name}: {e}")
        return False


def download_all_models():
    """Download all models needed for the project"""
    print("=" * 60)
    print("Downloading Pretrained Models for Offline Usage")
    print("=" * 60)
    print("\nThis will download ~200MB of model weights")
    print("Make sure you have a stable internet connection")
    print()
    
    models_to_download = [
        ('efficientnetb0', tf.keras.applications.EfficientNetB0),
        ('resnet50', tf.keras.applications.ResNet50),
        ('vgg16', tf.keras.applications.VGG16),
        ('mobilenetv2', tf.keras.applications.MobileNetV2),
    ]
    
    results = {}
    
    for model_name, model_class in models_to_download:
        success = download_model(model_name, model_class)
        results[model_name] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for model_name, success in results.items():
        status = "[OK] Success" if success else "[FAILED] Failed"
        print(f"{model_name:20s}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nDownloaded {successful}/{total} models successfully")
    
    if successful == total:
        print("\n[OK] All models downloaded!")
        print("You can now use the project OFFLINE")
        print("\nModels saved in: pretrained_models/")
    else:
        print("\n[WARNING] Some models failed to download")
        print("Check your internet connection and try again")
    
    return results


def verify_models():
    """Verify that all models can be loaded"""
    print("\n" + "=" * 60)
    print("Verifying Models...")
    print("=" * 60)
    
    models_dir = 'pretrained_models'
    
    if not os.path.exists(models_dir):
        print("✗ Models directory not found!")
        print("Run download first")
        return False
    
    models = ['efficientnetb0', 'resnet50', 'vgg16', 'mobilenetv2']
    
    all_ok = True
    for model_name in models:
        h5_path = os.path.join(models_dir, f'{model_name}.h5')
        saved_model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(h5_path):
            try:
                model = keras.models.load_model(h5_path)
                print(f"[OK] {model_name:20s}: OK")
            except Exception as e:
                print(f"[FAILED] {model_name:20s}: Failed to load - {e}")
                all_ok = False
        else:
            print(f"[FAILED] {model_name:20s}: Not found")
            all_ok = False
    
    if all_ok:
        print("\n[OK] All models verified and ready for offline use!")
    else:
        print("\n[WARNING] Some models have issues")
    
    return all_ok


def main():
    """Main download script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download pretrained models for offline usage')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing models instead of downloading')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_models()
    else:
        print("This will download pretrained TensorFlow models (~200MB)")
        print("Continue? (y/n): ", end='')
        
        # Auto-yes for non-interactive environments
        if sys.stdin.isatty():
            response = input().lower()
            if response != 'y':
                print("Cancelled")
                return
        else:
            print("y (non-interactive mode)")
        
        results = download_all_models()
        
        # Verify after download
        if sum(results.values()) > 0:
            verify_models()


if __name__ == "__main__":
    main()

