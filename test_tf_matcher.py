"""
Test script for deep learning image matcher
Creates test images and evaluates performance
"""

import numpy as np
import cv2
from tf_image_matcher import DeepImageMatcher
import time
import os


def create_test_image_pair():
    """
    Create a test image pair with known transformation
    """
    # Create base image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Add various patterns for better feature detection
    # Background gradient
    for i in range(600):
        img[i, :] = [i // 3, 50, 100]
    
    # Geometric shapes
    cv2.rectangle(img, (100, 100), (300, 250), (255, 100, 100), -1)
    cv2.rectangle(img, (110, 110), (290, 240), (200, 50, 50), 3)
    
    cv2.circle(img, (500, 200), 80, (100, 255, 100), -1)
    cv2.circle(img, (500, 200), 60, (50, 200, 50), 3)
    
    cv2.rectangle(img, (550, 350), (750, 500), (100, 100, 255), -1)
    
    # Add some texture with circles
    for _ in range(20):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        r = np.random.randint(5, 20)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img, (x, y), r, color, -1)
    
    # Add text
    cv2.putText(img, 'TEST IMAGE', (250, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)
    cv2.putText(img, 'PATTERN', (300, 450), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 0), 2)
    
    # Add grid pattern
    for i in range(0, 800, 40):
        cv2.line(img, (i, 0), (i, 600), (150, 150, 150), 1)
    for i in range(0, 600, 40):
        cv2.line(img, (0, i), (800, i), (150, 150, 150), 1)
    
    # Save reference image
    cv2.imwrite('test_ref_deep.jpg', img)
    
    # Apply known transformation
    center = (img.shape[1] // 2, img.shape[0] // 2)
    
    # Test different transformations
    test_cases = [
        {'angle': 15, 'tx': 50, 'ty': 30, 'name': 'small'},
        {'angle': 45, 'tx': 100, 'ty': -50, 'name': 'medium'},
        {'angle': -30, 'tx': -80, 'ty': 60, 'name': 'large'},
        {'angle': 90, 'tx': 20, 'ty': -20, 'name': 'rotate90'},
    ]
    
    results = []
    
    for test_case in test_cases:
        angle = test_case['angle']
        tx = test_case['tx']
        ty = test_case['ty']
        name = test_case['name']
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        img_transformed = cv2.warpAffine(
            img, M,
            (img.shape[1], img.shape[0]),
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Save transformed image
        filename = f'test_query_deep_{name}.jpg'
        cv2.imwrite(filename, img_transformed)
        
        results.append({
            'query_file': filename,
            'true_angle': angle,
            'true_tx': tx,
            'true_ty': ty
        })
    
    print("Created test image pairs:")
    print("  Reference: test_ref_deep.jpg")
    for r in results:
        print(f"  Query: {r['query_file']} (angle={r['true_angle']}°, tx={r['true_tx']}, ty={r['true_ty']})")
    
    return 'test_ref_deep.jpg', results


def evaluate_matcher(ref_file, test_cases, method='hybrid'):
    """
    Evaluate the matcher on test cases
    """
    print("\n" + "=" * 60)
    print(f"Evaluating Deep Learning Matcher (method: {method})")
    print("=" * 60)
    
    # Create matcher
    matcher = DeepImageMatcher(method=method)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['query_file']}")
        print("-" * 60)
        print(f"Ground Truth: angle={test_case['true_angle']}°, "
              f"tx={test_case['true_tx']}, ty={test_case['true_ty']}")
        
        try:
            # Time the matching
            start_time = time.time()
            
            img1, img2_aligned, M = matcher.align_images(
                ref_file,
                test_case['query_file'],
                save_output=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract estimated parameters
            est_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            est_tx = M[0, 2]
            est_ty = M[1, 2]
            
            # Calculate errors
            angle_error = abs(est_angle - test_case['true_angle'])
            # Handle angle wrap-around
            if angle_error > 180:
                angle_error = 360 - angle_error
            
            tx_error = abs(est_tx - test_case['true_tx'])
            ty_error = abs(est_ty - test_case['true_ty'])
            
            print(f"Estimated: angle={est_angle:.2f}°, tx={est_tx:.2f}, ty={est_ty:.2f}")
            print(f"Errors: angle={angle_error:.2f}°, tx={tx_error:.2f}, ty={ty_error:.2f}")
            print(f"Time: {elapsed_time:.3f}s")
            
            # Calculate alignment quality (MSE between aligned images)
            mse = np.mean((img1.astype(float) - img2_aligned.astype(float)) ** 2)
            print(f"Alignment MSE: {mse:.2f}")
            
            results.append({
                'test_case': test_case['query_file'],
                'angle_error': angle_error,
                'tx_error': tx_error,
                'ty_error': ty_error,
                'time': elapsed_time,
                'mse': mse,
                'success': True
            })
            
            # Save this test result
            cv2.imwrite(f'result_{method}_{i+1}_aligned.png', img2_aligned)
            overlay = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
            cv2.imwrite(f'result_{method}_{i+1}_overlay.png', overlay)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'test_case': test_case['query_file'],
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        avg_angle_error = np.mean([r['angle_error'] for r in successful])
        avg_tx_error = np.mean([r['tx_error'] for r in successful])
        avg_ty_error = np.mean([r['ty_error'] for r in successful])
        avg_time = np.mean([r['time'] for r in successful])
        avg_mse = np.mean([r['mse'] for r in successful])
        
        print(f"Success Rate: {len(successful)}/{len(results)}")
        print(f"Average Angle Error: {avg_angle_error:.2f}°")
        print(f"Average Translation Error: tx={avg_tx_error:.2f}, ty={avg_ty_error:.2f}")
        print(f"Average Time: {avg_time:.3f}s")
        print(f"Average Alignment MSE: {avg_mse:.2f}")
    else:
        print("All tests failed!")
    
    return results


def main():
    """Main test script"""
    print("=" * 60)
    print("Deep Learning Image Matcher - Performance Test")
    print("=" * 60)
    
    # Create test images
    print("\nCreating test images...")
    ref_file, test_cases = create_test_image_pair()
    
    # Test hybrid method (doesn't need training)
    results_hybrid = evaluate_matcher(ref_file, test_cases, method='hybrid')
    
    # If you have a trained model, uncomment this:
    # print("\n\nTesting with regression model...")
    # if os.path.exists('trained_transform_model.h5'):
    #     results_regression = evaluate_matcher(ref_file, test_cases, method='regression')
    # else:
    #     print("No trained model found. Run train_regressor.py first.")
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("Check result_*.png files for visual results")
    print("=" * 60)


if __name__ == "__main__":
    main()

