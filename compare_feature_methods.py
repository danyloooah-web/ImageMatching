"""
Compare ORB on raw images vs ORB on deep features
Demonstrates why deep features first is better
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_matcher import ImageMatcher as TraditionalMatcher
from tf_image_matcher import DeepImageMatcher
import time


def create_challenging_test_images():
    """
    Create test images with challenging conditions:
    - Lighting changes
    - Color variations
    - Shadows
    """
    # Create base image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(600):
        img[i, :] = [i // 3, 50, 100]
    
    # Add shapes
    cv2.rectangle(img, (100, 100), (300, 250), (255, 100, 100), -1)
    cv2.circle(img, (500, 200), 80, (100, 255, 100), -1)
    cv2.rectangle(img, (550, 350), (750, 500), (100, 100, 255), -1)
    
    # Add texture
    for _ in range(30):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        r = np.random.randint(5, 25)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img, (x, y), r, color, -1)
    
    # Add text
    cv2.putText(img, 'REFERENCE', (250, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)
    
    # Save reference
    cv2.imwrite('compare_ref.jpg', img)
    
    # Create transformed version with lighting change
    center = (img.shape[1] // 2, img.shape[0] // 2)
    angle = 25
    tx, ty = 60, -40
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    img_transformed = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                     borderMode=cv2.BORDER_REFLECT)
    
    # Apply lighting change (this makes it challenging!)
    img_transformed = cv2.convertScaleAbs(img_transformed, alpha=0.7, beta=30)
    
    # Add color shift
    img_transformed[:, :, 2] = cv2.convertScaleAbs(img_transformed[:, :, 2], alpha=1.2)
    
    # Save query
    cv2.imwrite('compare_query.jpg', img_transformed)
    
    print(f"Created test images with:")
    print(f"  Ground truth: angle={angle}°, tx={tx}, ty={ty}")
    print(f"  Lighting change: -30% brightness, +30 offset")
    print(f"  Color shift: +20% red channel")
    
    return angle, tx, ty


def test_traditional_orb():
    """Test traditional ORB on raw images"""
    print("\n" + "=" * 60)
    print("Method 1: Traditional ORB (ORB First)")
    print("=" * 60)
    
    matcher = TraditionalMatcher(feature_method='ORB')
    
    try:
        start_time = time.time()
        img1, img2, img2_aligned, M, kp1, kp2, matches, inliers = matcher.match_and_align(
            'compare_ref.jpg',
            'compare_query.jpg'
        )
        elapsed = time.time() - start_time
        
        # Extract parameters
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        
        print(f"✓ Success!")
        print(f"  Detected: angle={angle:.2f}°, tx={tx:.2f}, ty={ty:.2f}")
        print(f"  Matches: {len(matches)} total, {np.sum(inliers)} inliers")
        print(f"  Time: {elapsed:.3f}s")
        
        # Save result
        cv2.imwrite('compare_traditional_aligned.jpg', img2_aligned)
        overlay = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
        cv2.imwrite('compare_traditional_overlay.jpg', overlay)
        
        return {
            'success': True,
            'angle': angle,
            'tx': tx,
            'ty': ty,
            'matches': len(matches),
            'inliers': int(np.sum(inliers)),
            'time': elapsed
        }
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return {'success': False, 'error': str(e)}


def test_deep_features_hybrid():
    """Test deep features + ORB (hybrid method)"""
    print("\n" + "=" * 60)
    print("Method 2: Deep Features First (Hybrid)")
    print("=" * 60)
    
    matcher = DeepImageMatcher(method='hybrid')
    
    try:
        start_time = time.time()
        img1, img2_aligned, M = matcher.align_images(
            'compare_ref.jpg',
            'compare_query.jpg',
            save_output=False
        )
        elapsed = time.time() - start_time
        
        # Extract parameters
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        
        print(f"✓ Success!")
        print(f"  Detected: angle={angle:.2f}°, tx={tx:.2f}, ty={ty:.2f}")
        print(f"  Time: {elapsed:.3f}s")
        
        # Save result
        cv2.imwrite('compare_hybrid_aligned.jpg', img2_aligned)
        overlay = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
        cv2.imwrite('compare_hybrid_overlay.jpg', overlay)
        
        return {
            'success': True,
            'angle': angle,
            'tx': tx,
            'ty': ty,
            'time': elapsed
        }
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return {'success': False, 'error': str(e)}


def visualize_comparison(ground_truth, trad_result, hybrid_result):
    """Create comprehensive comparison visualization"""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    gt_angle, gt_tx, gt_ty = ground_truth
    
    print("\nGround Truth:")
    print(f"  Angle: {gt_angle}°")
    print(f"  Translation: ({gt_tx}, {gt_ty})")
    
    if trad_result['success']:
        print("\nTraditional ORB:")
        print(f"  Angle: {trad_result['angle']:.2f}° (error: {abs(trad_result['angle'] - gt_angle):.2f}°)")
        print(f"  Translation: ({trad_result['tx']:.1f}, {trad_result['ty']:.1f})")
        print(f"    Error: ({abs(trad_result['tx'] - gt_tx):.1f}, {abs(trad_result['ty'] - gt_ty):.1f}) pixels")
        print(f"  Matches: {trad_result['matches']} ({trad_result['inliers']} inliers)")
        print(f"  Time: {trad_result['time']:.3f}s")
    else:
        print("\nTraditional ORB: FAILED")
    
    if hybrid_result['success']:
        print("\nDeep Features + ORB (Hybrid):")
        print(f"  Angle: {hybrid_result['angle']:.2f}° (error: {abs(hybrid_result['angle'] - gt_angle):.2f}°)")
        print(f"  Translation: ({hybrid_result['tx']:.1f}, {hybrid_result['ty']:.1f})")
        print(f"    Error: ({abs(hybrid_result['tx'] - gt_tx):.1f}, {abs(hybrid_result['ty'] - gt_ty):.1f}) pixels")
        print(f"  Time: {hybrid_result['time']:.3f}s")
    else:
        print("\nDeep Features + ORB: FAILED")
    
    # Create comparison figure
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Load images
        ref = cv2.imread('compare_ref.jpg')
        query = cv2.imread('compare_query.jpg')
        
        # Row 1: Reference, Query, empty
        axes[0, 0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Reference Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Query Image\n(with lighting & color changes)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Add comparison text
        comparison_text = f"Ground Truth:\n  Rotation: {gt_angle}°\n  Translation: ({gt_tx}, {gt_ty})\n\n"
        
        if trad_result['success']:
            angle_err = abs(trad_result['angle'] - gt_angle)
            tx_err = abs(trad_result['tx'] - gt_tx)
            ty_err = abs(trad_result['ty'] - gt_ty)
            comparison_text += f"Traditional ORB:\n"
            comparison_text += f"  Angle Error: {angle_err:.2f}°\n"
            comparison_text += f"  Translation Error: ({tx_err:.1f}, {ty_err:.1f})\n"
            comparison_text += f"  Time: {trad_result['time']:.2f}s\n\n"
        else:
            comparison_text += "Traditional ORB: FAILED\n\n"
        
        if hybrid_result['success']:
            angle_err = abs(hybrid_result['angle'] - gt_angle)
            tx_err = abs(hybrid_result['tx'] - gt_tx)
            ty_err = abs(hybrid_result['ty'] - gt_ty)
            comparison_text += f"Deep Features + ORB:\n"
            comparison_text += f"  Angle Error: {angle_err:.2f}°\n"
            comparison_text += f"  Translation Error: ({tx_err:.1f}, {ty_err:.1f})\n"
            comparison_text += f"  Time: {hybrid_result['time']:.2f}s"
        
        axes[0, 2].text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        axes[0, 2].axis('off')
        
        # Row 2: Results
        if trad_result['success']:
            trad_overlay = cv2.imread('compare_traditional_overlay.jpg')
            axes[1, 0].imshow(cv2.cvtColor(trad_overlay, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Traditional ORB Result', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'FAILED', ha='center', va='center', fontsize=20, color='red')
            axes[1, 0].set_title('Traditional ORB Result', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        if hybrid_result['success']:
            hybrid_overlay = cv2.imread('compare_hybrid_overlay.jpg')
            axes[1, 1].imshow(cv2.cvtColor(hybrid_overlay, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('Deep Features + ORB Result', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'FAILED', ha='center', va='center', fontsize=20, color='red')
            axes[1, 1].set_title('Deep Features + ORB Result', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Winner annotation
        winner_text = ""
        if trad_result['success'] and hybrid_result['success']:
            trad_total_err = abs(trad_result['angle'] - gt_angle) + abs(trad_result['tx'] - gt_tx) + abs(trad_result['ty'] - gt_ty)
            hybrid_total_err = abs(hybrid_result['angle'] - gt_angle) + abs(hybrid_result['tx'] - gt_tx) + abs(hybrid_result['ty'] - gt_ty)
            
            if hybrid_total_err < trad_total_err:
                winner_text = "🏆 WINNER:\nDeep Features + ORB\n\n"
                winner_text += f"Better Accuracy!\n"
                winner_text += f"{((trad_total_err - hybrid_total_err) / trad_total_err * 100):.1f}% improvement"
            elif trad_total_err < hybrid_total_err:
                winner_text = "🏆 WINNER:\nTraditional ORB\n\n"
                winner_text += f"{trad_result['time'] / hybrid_result['time']:.1f}x faster"
            else:
                winner_text = "Tie!"
        elif hybrid_result['success']:
            winner_text = "🏆 WINNER:\nDeep Features + ORB\n\n(Traditional failed)"
        elif trad_result['success']:
            winner_text = "🏆 WINNER:\nTraditional ORB\n\n(Hybrid failed)"
        
        axes[1, 2].text(0.5, 0.5, winner_text, fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('feature_method_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Comparison visualization saved to: feature_method_comparison.png")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    """Main comparison"""
    print("=" * 60)
    print("Feature Extraction Method Comparison")
    print("ORB First vs Deep Features First")
    print("=" * 60)
    
    # Create challenging test images
    print("\nCreating challenging test images...")
    ground_truth = create_challenging_test_images()
    
    # Test traditional method
    trad_result = test_traditional_orb()
    
    # Test hybrid method
    hybrid_result = test_deep_features_hybrid()
    
    # Compare results
    visualize_comparison(ground_truth, trad_result, hybrid_result)
    
    print("\n" + "=" * 60)
    print("✓ Comparison complete!")
    print("Check the output files:")
    print("  - feature_method_comparison.png (comprehensive comparison)")
    print("  - compare_traditional_overlay.jpg")
    print("  - compare_hybrid_overlay.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()

