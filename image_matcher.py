"""
Image Matcher - Match and align two images captured from different cameras
Handles rotation and translation transformations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class ImageMatcher:
    """
    Class for matching and aligning two images of the same object
    captured from different camera positions.
    """
    
    def __init__(self, feature_method='ORB'):
        """
        Initialize the ImageMatcher
        
        Args:
            feature_method: Feature detection method ('ORB', 'SIFT', 'SURF')
        """
        self.feature_method = feature_method
        self.detector = self._get_detector()
        
    def _get_detector(self):
        """Get the feature detector based on the selected method"""
        if self.feature_method == 'ORB':
            return cv2.ORB_create(nfeatures=5000)
        elif self.feature_method == 'SIFT':
            return cv2.SIFT_create()
        elif self.feature_method == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Unknown feature method: {self.feature_method}")
    
    def detect_and_match_features(self, img1, img2, ratio_threshold=0.75):
        """
        Detect features in both images and match them
        
        Args:
            img1: First image (reference)
            img2: Second image (to be aligned)
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            keypoints1, keypoints2, good_matches
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        print(f"Detected {len(kp1)} keypoints in image 1")
        print(f"Detected {len(kp2)} keypoints in image 2")
        
        # Match features using BFMatcher
        if self.feature_method == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        return kp1, kp2, good_matches
    
    def estimate_rigid_transform(self, kp1, kp2, matches):
        """
        Estimate rigid transformation (rotation + translation) between images
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: Good matches between keypoints
            
        Returns:
            transformation_matrix: 2x3 affine transformation matrix
            inliers_mask: Mask indicating inlier matches
        """
        if len(matches) < 4:
            raise ValueError("Not enough matches to estimate transformation")
        
        # Extract matched keypoint coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate affine transformation using RANSAC
        # For rigid transform (rotation + translation), we use partial affine
        M, inliers = cv2.estimateAffinePartial2D(
            pts2, pts1, 
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if M is None:
            raise ValueError("Failed to estimate transformation")
        
        num_inliers = np.sum(inliers)
        print(f"Transformation estimated with {num_inliers}/{len(matches)} inliers")
        
        # Extract rotation angle
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        print(f"Estimated rotation: {angle:.2f} degrees")
        print(f"Estimated translation: ({M[0, 2]:.2f}, {M[1, 2]:.2f}) pixels")
        
        return M, inliers
    
    def apply_transformation(self, img, M, output_shape):
        """
        Apply transformation to image
        
        Args:
            img: Input image
            M: 2x3 transformation matrix
            output_shape: (height, width) of output image
            
        Returns:
            Transformed image
        """
        transformed = cv2.warpAffine(
            img, M, 
            (output_shape[1], output_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return transformed
    
    def match_and_align(self, img1_path, img2_path, ratio_threshold=0.75):
        """
        Complete pipeline: load images, match, and align
        
        Args:
            img1_path: Path to reference image
            img2_path: Path to image to be aligned
            ratio_threshold: Feature matching ratio threshold
            
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
        
        # Detect and match features
        kp1, kp2, good_matches = self.detect_and_match_features(
            img1, img2, ratio_threshold
        )
        
        if len(good_matches) < 4:
            raise ValueError("Not enough good matches found")
        
        # Estimate transformation
        M, inliers = self.estimate_rigid_transform(kp1, kp2, good_matches)
        
        # Apply transformation to align img2 to img1
        img2_aligned = self.apply_transformation(img2, M, img1.shape[:2])
        
        return img1, img2, img2_aligned, M, kp1, kp2, good_matches, inliers
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, inliers, 
                         save_path='matches.png'):
        """
        Visualize feature matches between images
        
        Args:
            img1: Reference image
            img2: Query image
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: List of matches
            inliers: Mask of inlier matches
            save_path: Path to save visualization
        """
        # Filter to show only inlier matches
        inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
        
        # Draw matches
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )
        
        # Save and display
        cv2.imwrite(save_path, match_img)
        print(f"Saved match visualization to {save_path}")
        
        return match_img
    
    def create_overlay(self, img1, img2_aligned, alpha=0.5):
        """
        Create overlay of two images
        
        Args:
            img1: Reference image
            img2_aligned: Aligned image
            alpha: Blending factor (0-1)
            
        Returns:
            Blended image
        """
        overlay = cv2.addWeighted(img1, alpha, img2_aligned, 1 - alpha, 0)
        return overlay
    
    def visualize_results(self, img1, img2, img2_aligned, overlay, 
                         save_path='alignment_result.png'):
        """
        Create comprehensive visualization of alignment results
        
        Args:
            img1: Reference image
            img2: Original second image
            img2_aligned: Aligned second image
            overlay: Overlay of img1 and img2_aligned
            save_path: Path to save visualization
        """
        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2_aligned_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title('Reference Image (Image 1)', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].set_title('Original Image 2', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(img2_aligned_rgb)
        axes[1, 0].set_title('Aligned Image 2', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay_rgb)
        axes[1, 1].set_title('Overlay (Blended Result)', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved result visualization to {save_path}")
        plt.close()


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python image_matcher.py <image1_path> <image2_path>")
        print("Example: python image_matcher.py ref.jpg query.jpg")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    print("=" * 60)
    print("Image Matching and Alignment")
    print("=" * 60)
    
    # Create matcher
    matcher = ImageMatcher(feature_method='ORB')
    
    try:
        # Perform matching and alignment
        img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
            img1_path, img2_path
        )
        
        # Visualize matches
        matcher.visualize_matches(img1, img2, kp1, kp2, matches, inliers)
        
        # Create overlay
        overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)
        
        # Save aligned image
        cv2.imwrite('aligned_image.png', img2_aligned)
        print("Saved aligned image to aligned_image.png")
        
        # Save overlay
        cv2.imwrite('overlay.png', overlay)
        print("Saved overlay to overlay.png")
        
        # Create comprehensive visualization
        matcher.visualize_results(img1, img2, img2_aligned, overlay)
        
        print("\n" + "=" * 60)
        print("Transformation Matrix:")
        print(M)
        print("=" * 60)
        print("\nAlignment completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

