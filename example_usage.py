"""
Example usage of ImageMatcher class
Demonstrates how to use the image matching and alignment functionality
"""

import cv2
import numpy as np
from image_matcher import ImageMatcher


def example_basic_usage():
    """Basic usage example"""
    print("Example 1: Basic Image Matching and Alignment")
    print("-" * 60)
    
    # Create matcher
    matcher = ImageMatcher(feature_method='ORB')
    
    # Paths to your images
    img1_path = 'image1.jpg'  # Reference image
    img2_path = 'image2.jpg'  # Image to align
    
    try:
        # Perform matching and alignment
        img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
            img1_path, img2_path
        )
        
        # Create overlay
        overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)
        
        # Save results
        cv2.imwrite('aligned_result.png', img2_aligned)
        cv2.imwrite('overlay_result.png', overlay)
        
        # Visualize
        matcher.visualize_matches(img1, img2, kp1, kp2, matches, inliers)
        matcher.visualize_results(img1, img2, img2_aligned, overlay)
        
        print("Success! Check the output images.")
        
    except Exception as e:
        print(f"Error: {e}")


def example_step_by_step():
    """Step-by-step example with more control"""
    print("\nExample 2: Step-by-Step Processing")
    print("-" * 60)
    
    # Initialize matcher
    matcher = ImageMatcher(feature_method='ORB')
    
    # Load images
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return
    
    # Step 1: Detect and match features
    kp1, kp2, good_matches = matcher.detect_and_match_features(img1, img2)
    
    # Step 2: Estimate transformation
    M, inliers = matcher.estimate_rigid_transform(kp1, kp2, good_matches)
    
    # Step 3: Apply transformation
    img2_aligned = matcher.apply_transformation(img2, M, img1.shape[:2])
    
    # Step 4: Create overlay
    overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)
    
    # Save results
    cv2.imwrite('step_by_step_aligned.png', img2_aligned)
    cv2.imwrite('step_by_step_overlay.png', overlay)
    
    print("Step-by-step processing completed!")


def example_create_test_images():
    """Create synthetic test images with known rotation and translation"""
    print("\nExample 3: Creating Test Images")
    print("-" * 60)
    
    # Create a test image with some patterns
    img_size = (600, 800, 3)
    img1 = np.zeros(img_size, dtype=np.uint8)
    
    # Add some geometric shapes
    cv2.rectangle(img1, (100, 100), (300, 200), (255, 0, 0), -1)
    cv2.circle(img1, (400, 300), 80, (0, 255, 0), -1)
    cv2.rectangle(img1, (500, 400), (700, 500), (0, 0, 255), -1)
    cv2.putText(img1, 'TEST', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 255, 255), 3)
    
    # Save reference image
    cv2.imwrite('test_image1.jpg', img1)
    
    # Create rotated and translated version
    # Get image center
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    
    # Rotation angle (degrees)
    angle = 15
    
    # Get rotation matrix
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Add translation
    M_rot[0, 2] += 50  # x translation
    M_rot[1, 2] += 30  # y translation
    
    # Apply transformation
    img2 = cv2.warpAffine(img1, M_rot, (img1.shape[1], img1.shape[0]),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(50, 50, 50))
    
    # Save transformed image
    cv2.imwrite('test_image2.jpg', img2)
    
    print(f"Created test images with:")
    print(f"  Rotation: {angle} degrees")
    print(f"  Translation: (50, 30) pixels")
    print("\nNow testing alignment...")
    
    # Now test the matcher on these images
    matcher = ImageMatcher(feature_method='ORB')
    
    try:
        img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
            'test_image1.jpg', 'test_image2.jpg'
        )
        
        overlay = matcher.create_overlay(img1, img2_aligned, alpha=0.5)
        
        cv2.imwrite('test_aligned.png', img2_aligned)
        cv2.imwrite('test_overlay.png', overlay)
        
        matcher.visualize_matches(img1, img2, kp1, kp2, matches, inliers, 
                                 save_path='test_matches.png')
        matcher.visualize_results(img1, img2, img2_aligned, overlay,
                                 save_path='test_result.png')
        
        print("\nTest completed! Check test_*.png files")
        
    except Exception as e:
        print(f"Error during test: {e}")


def example_different_blending():
    """Example showing different blending ratios"""
    print("\nExample 4: Different Blending Ratios")
    print("-" * 60)
    
    matcher = ImageMatcher(feature_method='ORB')
    
    try:
        img1, img2, img2_aligned, M, _, _, _, _ = matcher.match_and_align(
            'image1.jpg', 'image2.jpg'
        )
        
        # Try different alpha values
        alphas = [0.3, 0.5, 0.7]
        
        for alpha in alphas:
            overlay = matcher.create_overlay(img1, img2_aligned, alpha=alpha)
            filename = f'overlay_alpha_{int(alpha*100)}.png'
            cv2.imwrite(filename, overlay)
            print(f"Saved overlay with alpha={alpha} to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Image Matcher - Example Usage")
    print("=" * 60)
    
    # Run the test image example (works without real images)
    example_create_test_images()
    
    # Uncomment these if you have your own images:
    # example_basic_usage()
    # example_step_by_step()
    # example_different_blending()

