"""
Offline version of Deep Learning Image Matcher
Uses locally saved pretrained models instead of downloading from internet
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from typing import Tuple, Optional
import os


class OfflineDeepFeatureExtractor:
    """
    Extract deep features using locally saved pretrained models
    """
    
    def __init__(self, backbone='efficientnet', input_shape=(512, 512, 3), models_dir='pretrained_models'):
        """
        Initialize feature extractor with offline models
        
        Args:
            backbone: 'efficientnet', 'resnet50', 'vgg16', 'mobilenet'
            input_shape: Input image shape
            models_dir: Directory containing pretrained models
        """
        self.input_shape = input_shape
        self.backbone_name = backbone
        self.models_dir = models_dir
        self.model = self._load_offline_model(backbone)
        
    def _load_offline_model(self, backbone):
        """Load pretrained model from local files"""
        model_filename_map = {
            'efficientnet': 'efficientnetb0',
            'resnet50': 'resnet50',
            'vgg16': 'vgg16',
            'mobilenet': 'mobilenetv2'
        }
        
        model_filename = model_filename_map.get(backbone, 'efficientnetb0')
        
        # Try to load from saved model format first
        saved_model_path = os.path.join(self.models_dir, model_filename)
        h5_path = os.path.join(self.models_dir, f'{model_filename}.h5')
        
        if os.path.exists(saved_model_path):
            print(f"Loading {backbone} from {saved_model_path}")
            try:
                return keras.models.load_model(saved_model_path)
            except Exception as e:
                print(f"Failed to load SavedModel format: {e}")
        
        if os.path.exists(h5_path):
            print(f"Loading {backbone} from {h5_path}")
            try:
                return keras.models.load_model(h5_path)
            except Exception as e:
                print(f"Failed to load H5 format: {e}")
        
        # Fallback: try to load online (will work if internet available)
        print(f"⚠ Model not found locally, attempting to download...")
        print(f"Run 'python download_models_offline.py' first for offline usage")
        
        if backbone == 'efficientnet':
            return tf.keras.applications.EfficientNetB0(
                include_top=False, weights='imagenet', input_shape=self.input_shape
            )
        elif backbone == 'resnet50':
            return tf.keras.applications.ResNet50(
                include_top=False, weights='imagenet', input_shape=self.input_shape
            )
        elif backbone == 'vgg16':
            return tf.keras.applications.VGG16(
                include_top=False, weights='imagenet', input_shape=self.input_shape
            )
        elif backbone == 'mobilenet':
            return tf.keras.applications.MobileNetV2(
                include_top=False, weights='imagenet', input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def extract_features(self, image):
        """Extract dense features from image"""
        return self.model(image, training=False)


class OfflineDeepImageMatcher:
    """
    Complete offline deep learning-based image matcher
    Uses locally saved pretrained models
    """
    
    def __init__(self, input_shape=(512, 512, 3), models_dir='pretrained_models'):
        """
        Initialize offline matcher
        
        Args:
            input_shape: Input image shape
            models_dir: Directory containing pretrained models
        """
        self.input_shape = input_shape
        self.models_dir = models_dir
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"⚠ WARNING: Models directory '{models_dir}' not found!")
            print("Run 'python download_models_offline.py' to download models for offline use")
        
        self.feature_extractor = OfflineDeepFeatureExtractor(
            'efficientnet', 
            input_shape, 
            models_dir
        )
    
    def match_images(self, img1, img2):
        """
        Match and align two images using deep features
        
        Args:
            img1: Reference image
            img2: Query image
            
        Returns:
            transformation_matrix (2x3)
        """
        # Resize images
        img1_resized = cv2.resize(img1, self.input_shape[:2])
        img2_resized = cv2.resize(img2, self.input_shape[:2])
        
        # Preprocess
        img1_norm = tf.keras.applications.efficientnet.preprocess_input(
            img1_resized.astype(np.float32)
        )
        img2_norm = tf.keras.applications.efficientnet.preprocess_input(
            img2_resized.astype(np.float32)
        )
        
        # Add batch dimension
        img1_batch = np.expand_dims(img1_norm, axis=0)
        img2_batch = np.expand_dims(img2_norm, axis=0)
        
        # Extract features
        feat1 = self.feature_extractor.extract_features(img1_batch)[0]
        feat2 = self.feature_extractor.extract_features(img2_batch)[0]
        
        # Estimate transform from features
        M = self._estimate_transform_from_features(feat1, feat2, img1.shape, img2.shape)
        
        return M
    
    def _estimate_transform_from_features(self, feat1, feat2, orig_shape1, orig_shape2):
        """Estimate transformation from deep features"""
        feat1_np = feat1.numpy()
        feat2_np = feat2.numpy()
        
        h, w = feat1_np.shape[:2]
        
        # Convert to grayscale feature maps
        feat1_gray = np.mean(feat1_np, axis=-1)
        feat2_gray = np.mean(feat2_np, axis=-1)
        
        # Normalize to uint8
        feat1_uint8 = ((feat1_gray - feat1_gray.min()) / (feat1_gray.max() - feat1_gray.min()) * 255).astype(np.uint8)
        feat2_uint8 = ((feat2_gray - feat2_gray.min()) / (feat2_gray.max() - feat2_gray.min()) * 255).astype(np.uint8)
        
        # Detect keypoints on feature maps
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(feat1_uint8, None)
        kp2, des2 = orb.detectAndCompute(feat2_uint8, None)
        
        if len(kp1) < 4 or len(kp2) < 4:
            # Identity transform if not enough keypoints
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Filter matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # Get matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Scale points back to original image coordinates
        scale_x1 = orig_shape1[1] / w
        scale_y1 = orig_shape1[0] / h
        scale_x2 = orig_shape2[1] / w
        scale_y2 = orig_shape2[0] / h
        
        pts1[:, 0] *= scale_x1
        pts1[:, 1] *= scale_y1
        pts2[:, 0] *= scale_x2
        pts2[:, 1] *= scale_y2
        
        # Estimate rigid transform
        M, _ = cv2.estimateAffinePartial2D(
            pts2, pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if M is None:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        return M
    
    def align_images(self, img1_path, img2_path, save_output=True):
        """
        Complete pipeline to align two images
        
        Args:
            img1_path: Path to reference image
            img2_path: Path to query image
            save_output: Whether to save output images
            
        Returns:
            img1, img2_aligned, transformation_matrix
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Failed to load images")
        
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # Match and estimate transform
        print("Matching images using deep features (offline mode)...")
        M = self.match_images(img1, img2)
        
        # Extract parameters
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        
        print(f"Estimated rotation: {angle:.2f} degrees")
        print(f"Estimated translation: ({tx:.2f}, {ty:.2f}) pixels")
        
        # Apply transformation
        img2_aligned = cv2.warpAffine(
            img2, M,
            (img1.shape[1], img1.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        if save_output:
            cv2.imwrite('offline_aligned_image.png', img2_aligned)
            overlay = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
            cv2.imwrite('offline_overlay.png', overlay)
            print("Saved output images (offline_*.png)")
        
        return img1, img2_aligned, M


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python tf_image_matcher_offline.py <image1_path> <image2_path>")
        print("Example: python tf_image_matcher_offline.py ref.jpg query.jpg")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    print("=" * 60)
    print("Deep Learning Image Matching (OFFLINE MODE)")
    print("=" * 60)
    
    # Create matcher
    matcher = OfflineDeepImageMatcher()
    
    try:
        # Align images
        img1, img2_aligned, M = matcher.align_images(img1_path, img2_path)
        
        print("\n" + "=" * 60)
        print("Transformation Matrix:")
        print(M)
        print("=" * 60)
        print("\nAlignment completed successfully!")
        print("Check output files: offline_aligned_image.png, offline_overlay.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

