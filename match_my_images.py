"""
Simple script to match your dual-camera images
Usage: python match_my_images.py camera1.jpg camera2.jpg
"""

import sys
import os
import cv2
from image_matcher import ImageMatcher


def main():
    print("=" * 60)
    print("Dual-Camera Image Matching")
    print("=" * 60)
    print()
    
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python match_my_images.py <camera1_image> <camera2_image>")
        print()
        print("Example:")
        print("  python match_my_images.py camera1.jpg camera2.jpg")
        print()
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"Error: File not found: {img1_path}")
        sys.exit(1)
    
    if not os.path.exists(img2_path):
        print(f"Error: File not found: {img2_path}")
        sys.exit(1)
    
    print(f"Camera 1 Image: {img1_path}")
    print(f"Camera 2 Image: {img2_path}")
    print()
    
    # Create matcher
    print("Initializing matcher...")
    matcher = ImageMatcher(feature_method='ORB')
    
    try:
        print("Matching images...")
        print("-" * 60)
        
        # Perform matching
        img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
            img1_path,
            img2_path
        )
        
        print()
        print("-" * 60)
        print("Results:")
        print("-" * 60)
        
        # Extract transformation parameters
        import numpy as np
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        
        print(f"Rotation: {angle:.2f} degrees")
        print(f"Translation: ({tx:.2f}, {ty:.2f}) pixels")
        print(f"Matches found: {len(matches)}")
        print(f"Inliers: {int(np.sum(inliers))}")
        print(f"Match quality: {int(np.sum(inliers))/len(matches)*100:.1f}%")
        
        # Create visualizations
        print()
        print("Creating visualizations...")
        
        # Save aligned image
        cv2.imwrite('my_aligned_image.png', img2_aligned)
        print("  [OK] my_aligned_image.png")
        
        # Create overlay
        overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)
        cv2.imwrite('my_overlay.png', overlay)
        print("  [OK] my_overlay.png")
        
        # Visualize matches
        matcher.visualize_matches(img1, img2, kp1, kp2, matches, inliers, 
                                 save_path='my_matches.png')
        print("  [OK] my_matches.png")
        
        # Create result visualization
        matcher.visualize_results(img1, img2, img2_aligned, overlay,
                                 save_path='my_result.png')
        print("  [OK] my_result.png")
        
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print()
        print("Output files saved with 'my_' prefix:")
        print("  1. my_aligned_image.png  - Aligned camera 2 image")
        print("  2. my_overlay.png        - Blended overlay")
        print("  3. my_matches.png        - Feature matches")
        print("  4. my_result.png         - Complete comparison")
        print()
        print("Open 'my_result.png' to see the complete visualization!")
        
    except Exception as e:
        print()
        print("=" * 60)
        print("Error occurred!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

