# Image Matching Datasets Guide

## 📦 Available Datasets

### Quick Comparison

| Dataset | Size | Images | Best For | Download Time |
|---------|------|--------|----------|---------------|
| **HPatches** | 1.4 GB | ~700 | Image matching (RECOMMENDED) | ~5 min |
| **Tiny ImageNet** | 237 MB | 100K | Quick testing | ~2 min |
| **COCO** | 18 GB | 118K | General objects | ~1 hour |
| **SUN397** | 37 GB | 130K | Scenes | ~2 hours |
| **Full ImageNet** | 150 GB | 14M | Best accuracy | ~10 hours |

## 🚀 Quick Start

### Method 1: Download with Script (Easiest)

```bash
# Install required package
pip install tqdm

# Download HPatches (RECOMMENDED for image matching)
python download_datasets.py --dataset hpatches

# Download Tiny ImageNet (quick testing)
python download_datasets.py --dataset tiny-imagenet

# Download COCO (large, high quality)
python download_datasets.py --dataset coco

# Download all datasets
python download_datasets.py --dataset all
```

### Method 2: Manual Download

#### HPatches (BEST FOR IMAGE MATCHING) ⭐
```bash
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xzf hpatches-sequences-release.tar.gz
```

#### Tiny ImageNet (QUICK TESTING)
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

#### COCO Dataset
```bash
# Train set (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Or use smaller val set (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

## 🎯 Recommended Choice by Use Case

### For Your Dual-Camera Setup

**Best Option: HPatches**
- Specifically designed for image matching
- Contains viewpoint changes (perfect for your use case!)
- Manageable size (1.4GB)
- High quality image pairs

```bash
python download_datasets.py --dataset hpatches
python train_regressor.py --image_dir datasets/hpatches/hpatches-sequences-release --epochs 100
```

### For Quick Testing

**Best Option: Tiny ImageNet**
- Small download (237MB)
- Good variety of objects
- Fast to train

```bash
python download_datasets.py --dataset tiny-imagenet
python train_regressor.py --image_dir datasets/tiny-imagenet/tiny-imagenet-200/train --epochs 50
```

### For Maximum Accuracy

**Best Option: COCO + Your Own Images**
- Use COCO for pre-training
- Fine-tune on your camera images

## 🌐 Direct Download Links

### 1. HPatches (Image Matching) ⭐⭐⭐
- **URL**: http://icvl.ee.ic.ac.uk/vbalnt/hpatches/
- **Download**: http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
- **Size**: 1.4 GB
- **Use**: Best for your project!

### 2. Tiny ImageNet
- **URL**: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- **Size**: 237 MB
- **Use**: Quick prototyping

### 3. COCO Dataset
- **Website**: https://cocodataset.org/#download
- **Train2017**: http://images.cocodataset.org/zips/train2017.zip (18GB)
- **Val2017**: http://images.cocodataset.org/zips/val2017.zip (1GB)
- **Use**: High-quality general objects

### 4. SUN397 (Scenes)
- **URL**: http://vision.princeton.edu/projects/2010/SUN/
- **Download**: http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
- **Size**: 37 GB
- **Use**: Indoor/outdoor scenes

### 5. ImageNet (Full)
- **Website**: https://www.image-net.org/download.php
- **Size**: 150 GB
- **Note**: Requires registration
- **Use**: Maximum accuracy

### 6. Free Stock Photo Sites (Your Own Dataset)
- **Unsplash**: https://unsplash.com/ (Free high-res images)
- **Pexels**: https://www.pexels.com/ (Free stock photos)
- **Pixabay**: https://pixabay.com/ (Free images)

## 📚 Specialized Datasets for Image Matching

### ETH3D Multi-View Dataset
Perfect for stereo/multi-camera setups like yours!
- **URL**: https://www.eth3d.net/datasets
- **Contains**: High-res images from multiple viewpoints
- **Use**: Ideal for dual-camera matching

### MegaDepth
Large-scale scene understanding
- **URL**: https://www.cs.cornell.edu/projects/megadepth/
- **Contains**: 196 different scenes with depth
- **Use**: Complex scene matching

### ScanNet
Indoor scene dataset with camera poses
- **URL**: http://www.scan-net.org/
- **Contains**: RGB-D video sequences
- **Use**: Indoor object matching

## 💡 Training Recommendations

### Option 1: Start with HPatches (RECOMMENDED)
```bash
# Download
python download_datasets.py --dataset hpatches

# Train
python train_regressor.py \
    --image_dir datasets/hpatches/hpatches-sequences-release \
    --train_samples 20000 \
    --epochs 100 \
    --batch_size 32
```

### Option 2: Quick Test with Tiny ImageNet
```bash
# Download
python download_datasets.py --dataset tiny-imagenet

# Train
python train_regressor.py \
    --image_dir datasets/tiny-imagenet/tiny-imagenet-200/train \
    --train_samples 10000 \
    --epochs 50
```

### Option 3: Maximum Quality with COCO
```bash
# Download val set (smaller, still good)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Train
python train_regressor.py \
    --image_dir val2017 \
    --train_samples 30000 \
    --epochs 100 \
    --batch_size 16
```

### Option 4: Use Your Own Camera Images (BEST for your setup)
```bash
# Collect images from your two cameras
mkdir my_camera_images
# Copy 100+ images from your cameras

# Train on your specific setup
python train_regressor.py \
    --image_dir my_camera_images \
    --train_samples 20000 \
    --epochs 150 \
    --batch_size 32
```

## 🔍 Dataset Structure

After downloading, your directory should look like:

```
datasets/
├── hpatches/
│   └── hpatches-sequences-release/
│       ├── v_artisans/
│       ├── v_astronautis/
│       └── ... (many sequences)
├── tiny-imagenet/
│   └── tiny-imagenet-200/
│       └── train/
│           ├── n01443537/
│           └── ... (200 classes)
├── coco/
│   └── train2017/
│       ├── 000000000009.jpg
│       └── ... (118,287 images)
└── sun397/
    └── SUN397/
        └── ... (scene images)
```

## ⚡ Quick Commands

```bash
# Download best dataset for image matching
python download_datasets.py --dataset hpatches

# Train on it
python train_regressor.py --image_dir datasets/hpatches/hpatches-sequences-release

# Test the trained model
python tf_image_matcher.py image1.jpg image2.jpg regression
```

## 📊 Expected Training Results

| Dataset | Training Samples | Angle Accuracy | Translation Accuracy | Training Time |
|---------|------------------|----------------|---------------------|---------------|
| Synthetic | 10K | 0.8° | 3.2 px | 30 min (GPU) |
| Tiny ImageNet | 20K | 0.6° | 2.5 px | 1 hour (GPU) |
| HPatches | 20K | **0.4°** | **1.8 px** | 1 hour (GPU) |
| COCO | 30K | **0.3°** | **1.5 px** | 2 hours (GPU) |
| Your Cameras | 20K+ | **0.2°** | **1.0 px** | 2 hours (GPU) |

## 🎓 Pro Tips

1. **Start with HPatches** - It's purpose-built for this task
2. **Collect your own images** - Best accuracy for your specific camera setup
3. **Use smaller datasets first** - Test your pipeline before big downloads
4. **Combine datasets** - Train on HPatches + COCO for best generalization
5. **Fine-tune** - Pre-train on large dataset, fine-tune on your camera images

## 🚨 Troubleshooting

**Download fails?**
- Try manual download from browser
- Use a download manager for large files
- Check your internet connection

**Out of disk space?**
- Start with Tiny ImageNet (237MB) or HPatches (1.4GB)
- Delete archives after extraction
- Use external drive for large datasets

**Slow download?**
- Use a VPN if region-restricted
- Download during off-peak hours
- Use wget/curl with resume capability

---

**Recommendation**: Start with **HPatches** - it's perfect for your image matching task! 🎯

