"""
Test model predictions on random images from the dataset
Displays 10 random images with model predictions
"""

import os
from pathlib import Path
import torch
import functools
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Fix for PyTorch 2.6+ weights_only=True default
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO


# Configuration
DATASET_DIR = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/dataset")
MODEL_PATH = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/model.pt")
OUTPUT_DIR = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/prediction_results")


def get_all_images():
    """Get all image paths from train, valid, and test directories (class folders)"""
    image_paths = []
    
    for split in ['train', 'valid', 'test']:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            # Get images from class folders
            for class_folder in split_dir.iterdir():
                if class_folder.is_dir():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        image_paths.extend(list(class_folder.glob(ext)))
    
    return image_paths


def classify_image_from_detections(model, image_path):
    """
    Classify an image using classification model.
    Returns: (predicted_class, top5_dict, all_dict, raw_results)
    """
    # Import the classification function from main
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import classify_image
    
    predicted_class, top5_dict, all_dict = classify_image(model, image_path)
    results = model(str(image_path), verbose=False)
    
    return predicted_class, top5_dict, all_dict, results


def draw_classification(image, predicted_class, top5_dict, all_dict, ax):
    """Draw image with classification label"""
    # Load image
    img_array = np.array(image)
    ax.imshow(img_array)
    ax.axis('off')
    
    # Display classification result
    if predicted_class:
        # Format confidence information
        conf_text = f"Predicted: {predicted_class.upper()}\n"
        
        if top5_dict:
            conf_text += f"Confidence: {top5_dict.get(predicted_class, 0.0):.2f}\n"
            other_classes = [f"{k}:{v:.2f}" for k, v in top5_dict.items() if k != predicted_class]
            if other_classes:
                conf_text += "Other: " + ", ".join(other_classes[:3])
        
        # Draw classification label
        ax.text(
            0.5, 0.02, conf_text,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=11, color='white', weight='bold',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
            family='monospace'
        )
    else:
        # No classification
        ax.text(
            0.5, 0.5, 'No classification',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=14, color='red', weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )


def test_random_images(num_images=10, split=None):
    """
    Test model on random images and display classification results
    
    Args:
        num_images: Number of images to test (default: 10)
        split: Which split to use ('train', 'valid', 'test', or None for all)
    """
    print("="*60)
    print("MODEL CLASSIFICATION TEST")
    print("="*60)
    
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get image paths
    if split:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            print(f"❌ Error: Split directory not found: {split_dir}")
            return
        image_paths = []
        # Get images from class folders
        for class_folder in split_dir.iterdir():
            if class_folder.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_paths.extend(list(class_folder.glob(ext)))
    else:
        image_paths = get_all_images()
    
    if len(image_paths) == 0:
        print("❌ Error: No images found in dataset")
        return
    
    # Select random images
    num_images = min(num_images, len(image_paths))
    selected_images = random.sample(image_paths, num_images)
    
    print(f"\n📸 Found {len(image_paths)} images")
    print(f"🎲 Selected {num_images} random images for testing")
    print(f"🤖 Loading model: {MODEL_PATH}")
    
    # Load model
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("✅ Model loaded successfully\n")
    
    # Run classification on images
    all_results = []
    for i, img_path in enumerate(selected_images, 1):
        print(f"Processing image {i}/{num_images}: {img_path.name}")
        try:
            predicted_class, top5_dict, all_dict, raw_results = classify_image_from_detections(model, img_path)
            all_results.append((img_path, predicted_class, top5_dict, all_dict, raw_results))
        except Exception as e:
            print(f"  ⚠️  Error processing {img_path.name}: {e}")
            all_results.append((img_path, None, {}, {}, None))
    
    # Display results
    print("\n" + "="*60)
    print("GENERATING CLASSIFICATION VISUALIZATION")
    print("="*60)
    
    # Create figure with subplots
    cols = 5
    rows = (num_images + cols - 1) // cols  # Ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle('Image Classification Results', fontsize=16, fontweight='bold')
    
    # Flatten axes if needed
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, result_data in enumerate(all_results):
        ax = axes[idx]
        
        if len(result_data) == 5:
            img_path, predicted_class, top5_dict, all_dict, raw_results = result_data
        else:
            # Fallback for old format
            img_path, predicted_class = result_data[0], result_data[1] if len(result_data) > 1 else None
            top5_dict, all_dict = {}, {}
            raw_results = None
        
        try:
            # Load image
            image = Image.open(img_path)
            
            # Draw classification
            draw_classification(image, predicted_class, top5_dict, all_dict, ax)
            
            # Add image path as title (filename only)
            ax.set_title(img_path.name, fontsize=8, pad=2)
            
            # Print summary to console
            if predicted_class:
                conf = top5_dict.get(predicted_class, 0.0)
                print(f"  {img_path.name}: {predicted_class.upper()} (confidence: {conf:.2f})")
            else:
                print(f"  {img_path.name}: No classification")
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            ax.set_title(img_path.name, fontsize=8)
            print(f"  {img_path.name}: Error - {e}")
    
    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    output_file = OUTPUT_DIR / f"predictions_{split or 'all'}_{num_images}images.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\n✅ Saved prediction visualization to:")
    print(f"   {output_file}")
    print(f"\n📁 You can open this file to view the predictions.")
    
    # Also save individual images with classifications
    print("\n💾 Saving individual classification images...")
    for idx, result_data in enumerate(all_results, 1):
        try:
            if len(result_data) == 5:
                img_path, predicted_class, top5_dict, all_dict, raw_results = result_data
            else:
                img_path, predicted_class = result_data[0], result_data[1] if len(result_data) > 1 else None
                top5_dict, all_dict = {}, {}
            
            image = Image.open(img_path)
            
            # Create figure for single image
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            draw_classification(image, predicted_class, top5_dict, all_dict, ax)
            ax.set_title(f"{img_path.name}", fontsize=12, fontweight='bold')
            
            # Save individual image
            output_file_single = OUTPUT_DIR / f"classification_{idx:02d}_{img_path.stem}.png"
            plt.savefig(output_file_single, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Could not save individual image for {result_data[0].name}: {e}")
    
    print(f"\n✅ Individual images saved to: {OUTPUT_DIR}")


def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
    num_images = 10
    split = None
    
    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("\nUsage: python test_predictions.py [options]")
            print("\nOptions:")
            print("  --num N        Number of images to test (default: 10)")
            print("  --split SPLIT  Use specific split: train, valid, or test")
            print("  --help, -h     Show this help message")
            print("\nExample:")
            print("  python test_predictions.py --num 5 --split test")
            return
        
        if '--num' in sys.argv:
            idx = sys.argv.index('--num')
            if idx + 1 < len(sys.argv):
                try:
                    num_images = int(sys.argv[idx + 1])
                except ValueError:
                    print("❌ Invalid number for --num")
                    return
        
        if '--split' in sys.argv:
            idx = sys.argv.index('--split')
            if idx + 1 < len(sys.argv):
                split = sys.argv[idx + 1]
                if split not in ['train', 'valid', 'test']:
                    print(f"❌ Invalid split: {split}. Use 'train', 'valid', or 'test'")
                    return
    
    test_random_images(num_images=num_images, split=split)


if __name__ == "__main__":
    main()
