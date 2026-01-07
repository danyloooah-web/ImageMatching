"""
Training script for the transformation regression network
Generates synthetic training data and trains the model
"""

import tensorflow as tf
import numpy as np
import cv2
from tf_image_matcher import TransformationRegressor
from tqdm import tqdm
import os


class SyntheticDataGenerator:
    """
    Generate synthetic training data by applying random transformations
    """
    
    def __init__(self, image_paths, input_shape=(512, 512, 3), 
                 max_rotation=180, max_translation=0.2):
        """
        Initialize data generator
        
        Args:
            image_paths: List of paths to images for generating pairs
            input_shape: Input shape for the network
            max_rotation: Maximum rotation angle in degrees
            max_translation: Maximum translation as fraction of image size
        """
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.images = self._load_images()
        
    def _load_images(self):
        """Load all images"""
        images = []
        print(f"Loading {len(self.image_paths)} images...")
        for path in tqdm(self.image_paths):
            img = cv2.imread(path)
            if img is not None:
                # Resize to input shape
                img_resized = cv2.resize(img, self.input_shape[:2])
                images.append(img_resized)
        print(f"Loaded {len(images)} images successfully")
        return images
    
    def generate_pair(self):
        """
        Generate a training pair with random transformation
        
        Returns:
            img1, img2_transformed, [angle_rad, tx_norm, ty_norm]
        """
        # Select random image
        img1 = self.images[np.random.randint(len(self.images))].copy()
        
        # Generate random transformation parameters
        angle = np.random.uniform(-self.max_rotation, self.max_rotation)
        h, w = img1.shape[:2]
        tx = np.random.uniform(-self.max_translation * w, self.max_translation * w)
        ty = np.random.uniform(-self.max_translation * h, self.max_translation * h)
        
        # Apply transformation
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        
        img2 = cv2.warpAffine(
            img1, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Normalize images
        img1_norm = img1.astype(np.float32) / 255.0
        img2_norm = img2.astype(np.float32) / 255.0
        
        # Normalize transformation parameters
        angle_rad = np.radians(angle)
        tx_norm = tx / w
        ty_norm = ty / h
        
        params = np.array([angle_rad, tx_norm, ty_norm], dtype=np.float32)
        
        return img1_norm, img2_norm, params
    
    def generate_batch(self, batch_size):
        """Generate a batch of training pairs"""
        batch_img1 = []
        batch_img2 = []
        batch_params = []
        
        for _ in range(batch_size):
            img1, img2, params = self.generate_pair()
            batch_img1.append(img1)
            batch_img2.append(img2)
            batch_params.append(params)
        
        return (np.array(batch_img1), np.array(batch_img2)), np.array(batch_params)
    
    def create_dataset(self, num_samples, batch_size):
        """
        Create a tf.data.Dataset
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size
            
        Returns:
            tf.data.Dataset
        """
        def generator():
            for _ in range(num_samples):
                img1, img2, params = self.generate_pair()
                yield (img1, img2), params
        
        output_signature = (
            (tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
             tf.TensorSpec(shape=self.input_shape, dtype=tf.float32)),
            tf.TensorSpec(shape=(3,), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def collect_images_from_directory(directory, extensions=['.jpg', '.jpeg', '.png']):
    """Collect all image paths from a directory"""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def create_synthetic_images(num_images=100, output_dir='synthetic_train_images'):
    """
    Create synthetic images for training if no real images available
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} synthetic training images...")
    
    for i in tqdm(range(num_images)):
        # Create random image with various patterns
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Random background
        bg_color = np.random.randint(0, 100, 3)
        img[:] = bg_color
        
        # Add random shapes
        num_shapes = np.random.randint(5, 15)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rectangle', 'circle', 'line', 'text'])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            if shape_type == 'rectangle':
                pt1 = (np.random.randint(0, 450), np.random.randint(0, 450))
                pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
                cv2.rectangle(img, pt1, pt2, color, -1)
            
            elif shape_type == 'circle':
                center = (np.random.randint(50, 462), np.random.randint(50, 462))
                radius = np.random.randint(10, 50)
                cv2.circle(img, center, radius, color, -1)
            
            elif shape_type == 'line':
                pt1 = (np.random.randint(0, 512), np.random.randint(0, 512))
                pt2 = (np.random.randint(0, 512), np.random.randint(0, 512))
                cv2.line(img, pt1, pt2, color, np.random.randint(2, 10))
            
            elif shape_type == 'text':
                pt = (np.random.randint(10, 400), np.random.randint(30, 482))
                text = chr(np.random.randint(65, 90))  # Random letter
                cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                           np.random.uniform(0.5, 2.0), color, 2)
        
        # Save
        cv2.imwrite(os.path.join(output_dir, f'synthetic_{i:04d}.jpg'), img)
    
    print(f"Created synthetic images in {output_dir}/")
    return output_dir


def train_model(image_directory=None, 
                train_samples=10000,
                val_samples=1000,
                batch_size=32,
                epochs=50,
                model_save_path='trained_transform_model.h5'):
    """
    Train the transformation regression model
    
    Args:
        image_directory: Directory containing training images
        train_samples: Number of training samples to generate
        val_samples: Number of validation samples
        batch_size: Batch size
        epochs: Number of epochs
        model_save_path: Path to save trained model
    """
    print("=" * 60)
    print("Training Transformation Regression Network")
    print("=" * 60)
    
    # Get image paths
    if image_directory is None or not os.path.exists(image_directory):
        print("No image directory provided or not found.")
        print("Creating synthetic training images...")
        image_directory = create_synthetic_images(num_images=200)
    
    image_paths = collect_images_from_directory(image_directory)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_directory}")
    
    print(f"Found {len(image_paths)} images for training")
    
    # Create data generators
    print("\nCreating training data generator...")
    train_gen = SyntheticDataGenerator(
        image_paths,
        input_shape=(512, 512, 3),
        max_rotation=180,
        max_translation=0.2
    )
    
    print("Creating validation data generator...")
    val_gen = SyntheticDataGenerator(
        image_paths,
        input_shape=(512, 512, 3),
        max_rotation=180,
        max_translation=0.2
    )
    
    # Create datasets
    print(f"\nGenerating {train_samples} training samples...")
    train_dataset = train_gen.create_dataset(train_samples, batch_size)
    
    print(f"Generating {val_samples} validation samples...")
    val_dataset = val_gen.create_dataset(val_samples, batch_size)
    
    # Create and compile model
    print("\nBuilding model...")
    regressor = TransformationRegressor(input_shape=(512, 512, 3))
    regressor.compile_model(learning_rate=1e-4)
    
    print("\nModel Summary:")
    regressor.model.summary()
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = regressor.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save final model
    regressor.save(model_save_path)
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Training and Validation MAE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("Training history plot saved to training_history.png")
        
    except Exception as e:
        print(f"Could not plot training history: {e}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train transformation regression network')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing training images')
    parser.add_argument('--train_samples', type=int, default=10000,
                       help='Number of training samples (default: 10000)')
    parser.add_argument('--val_samples', type=int, default=1000,
                       help='Number of validation samples (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--output', type=str, default='trained_transform_model.h5',
                       help='Output model path (default: trained_transform_model.h5)')
    
    args = parser.parse_args()
    
    # Train
    train_model(
        image_directory=args.image_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_save_path=args.output
    )


if __name__ == "__main__":
    main()

