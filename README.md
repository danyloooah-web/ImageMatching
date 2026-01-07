# Image Matching and Alignment

A Python 3.7 project for matching and aligning two images captured from different camera positions. This project handles **rotation** and **translation** transformations to align images of the same object.

## Features

- 🎯 **Feature Detection**: Uses ORB, SIFT, or SURF for robust feature detection
- 🔄 **Rigid Transformation**: Estimates rotation and translation between images
- 📐 **RANSAC-based Alignment**: Robust estimation that handles outliers
- 🎨 **Visualization**: Comprehensive visualization of matches and results
- 🖼️ **Image Overlay**: Creates blended overlay of aligned images

## Requirements

- Python 3.7
- OpenCV 4.5.5
- NumPy 1.19.5
- Matplotlib 3.3.4

## Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using Command Line

```bash
python image_matcher.py image1.jpg image2.jpg
```

This will:
- Load both images
- Detect and match features
- Estimate the transformation (rotation + translation)
- Align the second image to the first
- Generate visualizations and save results

### Using as a Module

```python
from image_matcher import ImageMatcher

# Create matcher
matcher = ImageMatcher(feature_method='ORB')

# Perform matching and alignment
img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
    'reference.jpg',
    'query.jpg'
)

# Create overlay
overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)

# Save results
import cv2
cv2.imwrite('aligned.png', img2_aligned)
cv2.imwrite('overlay.png', overlay)
```

## Example Usage

Run the example script to create test images and see the alignment in action:

```bash
python example_usage.py
```

This will:
1. Create synthetic test images with known rotation and translation
2. Run the alignment algorithm
3. Generate comprehensive visualizations

## Output Files

The program generates several output files:

- `aligned_image.png` - The second image transformed to align with the first
- `overlay.png` - Blended overlay of both images
- `matches.png` - Visualization of feature matches
- `alignment_result.png` - Comprehensive 4-panel visualization showing all steps

## How It Works

### 1. Feature Detection
The algorithm detects distinctive keypoints in both images using ORB (Oriented FAST and Rotated BRIEF) features by default.

### 2. Feature Matching
Features are matched between images using a Brute-Force matcher with Lowe's ratio test to filter out poor matches.

### 3. Transformation Estimation
A rigid transformation (rotation + translation) is estimated using RANSAC to handle outliers robustly.

### 4. Image Alignment
The transformation is applied to the second image to align it with the first image.

### 5. Overlay Creation
The aligned images are blended together to visualize the quality of alignment.

## Advanced Usage

### Using Different Feature Detectors

```python
# Use ORB (fast, free)
matcher = ImageMatcher(feature_method='ORB')

# Use SIFT (more accurate, patented)
matcher = ImageMatcher(feature_method='SIFT')
```

### Adjusting Matching Parameters

```python
# Adjust ratio threshold for feature matching
# Lower = stricter (fewer but better matches)
# Higher = more permissive (more matches but may include false positives)
img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
    'image1.jpg',
    'image2.jpg',
    ratio_threshold=0.7  # Default is 0.75
)
```

### Custom Blending

```python
# Adjust alpha for different blending ratios
overlay_50_50 = matcher.create_overlay(img1, img2_aligned, alpha=0.5)  # 50-50 blend
overlay_70_30 = matcher.create_overlay(img1, img2_aligned, alpha=0.7)  # 70% img1, 30% img2
```

## Step-by-Step Processing

For more control, you can run each step individually:

```python
from image_matcher import ImageMatcher
import cv2

# Initialize
matcher = ImageMatcher(feature_method='ORB')
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Step 1: Detect and match features
kp1, kp2, good_matches = matcher.detect_and_match_features(img1, img2)

# Step 2: Estimate transformation
M, inliers = matcher.estimate_rigid_transform(kp1, kp2, good_matches)

# Step 3: Apply transformation
img2_aligned = matcher.apply_transformation(img2, M, img1.shape[:2])

# Step 4: Create visualization
overlay = matcher.create_overlay(img1, img2_aligned)
```

## Understanding the Transformation Matrix

The transformation matrix `M` is a 2x3 affine transformation matrix:

```
M = [[cos(θ), -sin(θ), tx],
     [sin(θ),  cos(θ), ty]]
```

Where:
- `θ` is the rotation angle
- `tx, ty` are the translation values in x and y directions

The program prints the estimated rotation angle and translation values.

## Troubleshooting

### Not enough matches found

Try:
- Using better quality images with more texture/features
- Increasing the `ratio_threshold` parameter
- Using SIFT instead of ORB: `ImageMatcher(feature_method='SIFT')`
- Ensuring the images actually contain the same object

### Poor alignment quality

Try:
- Checking if the object is truly the same size in both images
- Ensuring good lighting and image quality
- Adjusting RANSAC parameters (edit `ransacReprojThreshold` in the code)

### Images too different

The algorithm works best when:
- The same object is visible in both images
- The object occupies a significant portion of the image
- There's sufficient texture/features to match
- Rotation is less than 180 degrees
- Images have reasonable quality and contrast

## Technical Details

- **Feature Detector**: ORB (default), SIFT, or SURF
- **Matching Method**: Brute-Force with Lowe's ratio test
- **Transform Estimation**: Partial Affine (rotation + translation only)
- **Outlier Rejection**: RANSAC algorithm
- **Image Warping**: Bilinear interpolation

## License

This project is provided as-is for educational and commercial use.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- ORB: "ORB: an efficient alternative to SIFT or SURF" (Rublee et al., 2011)
- RANSAC: "Random sample consensus" (Fischler & Bolles, 1981)

