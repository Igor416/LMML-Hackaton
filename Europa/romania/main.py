"""
YOLOv8 Classification Model Training
Train a classification model to recognize soft drink brands
"""

import os
from pathlib import Path
import torch
import functools

# Fix for PyTorch 2.6+ weights_only=True default
# Monkeypatch torch.load to use weights_only=False by default for ultralytics compatibility
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, default to False for compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
import shutil


# Configuration
DATASET_DIR = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/dataset")
OUTPUT_DIR = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/output")
MODEL_PATH = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/model.pt")
MIN_ACCURACY = 90.0  # Minimum accuracy required to pass

# Class mapping: folder names to output strings
CLASS_MAPPING = {
    'Coca-cola': 'cola',
    'Fanta': 'fanta',
    'Sprite': 'sprite'
}

# Reverse mapping for getting class name from index
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def classify_image(model, image_path):
    """
    Classify an image using YOLOv8 classification model.
    
    Returns:
        tuple: (predicted_class, top5_predictions_dict, all_predictions_dict)
    """
    # Run classification (not detection)
    results = model(str(image_path), verbose=False)
    
    if not results or len(results) == 0:
        return None, {}, {}
    
    result = results[0]
    
    # Get top prediction
    if hasattr(result, 'names') and hasattr(result, 'probs'):
        # Get class index with highest probability
        top1_idx = result.probs.top1
        top1_class_name = result.names[top1_idx]
        
        # Map to output class
        predicted_class = CLASS_MAPPING.get(top1_class_name, 'cola')
        
        # Get top 5 predictions
        top5 = result.probs.top5
        top5_conf = result.probs.top5conf
        
        top5_dict = {}
        all_dict = {}
        
        for idx, conf in enumerate(result.probs.data):
            folder_name = result.names[idx]
            mapped_class = CLASS_MAPPING.get(folder_name, 'cola')
            all_dict[folder_name] = float(conf)
            
            if idx in top5:
                top5_dict[mapped_class] = float(conf)
        
        return predicted_class, top5_dict, all_dict
    else:
        return None, {}, {}


def classify_image_simple(model, image_path):
    """
    Classify an image and return only the predicted class as a string.
    
    Args:
        model: YOLO classification model instance
        image_path: Path to image file
    
    Returns:
        str: 'cola', 'fanta', 'sprite', or None if no prediction
    """
    predicted_class, _, _ = classify_image(model, image_path)
    return predicted_class


def get_ground_truth_class(image_path):
    """
    Get ground truth class from image path (folder structure).
    Returns the mapped class name.
    """
    # Extract class folder name from path
    # Path format: .../train/Coca-cola/image.jpg or .../valid/Fanta/image.jpg
    path_parts = Path(image_path).parts
    
    for part in reversed(path_parts):
        if part in CLASS_MAPPING:
            return CLASS_MAPPING[part]
    
    return None


def calculate_classification_accuracy(model, dataset_split='valid'):
    """
    Calculate classification accuracy by comparing image-level predictions
    to ground truth from folder structure.
    """
    from collections import defaultdict
    
    split_dir = DATASET_DIR / dataset_split
    
    if not split_dir.exists():
        return None, {}
    
    # Get all images from class folders
    image_paths = []
    for class_folder in split_dir.iterdir():
        if class_folder.is_dir() and class_folder.name in CLASS_MAPPING:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_paths.extend(list(class_folder.glob(ext)))
    
    if len(image_paths) == 0:
        return None, {}
    
    correct = 0
    total = 0
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print(f"\nEvaluating {len(image_paths)} images from {dataset_split} set...")
    
    for img_path in image_paths:
        # Get ground truth from folder name
        true_class = get_ground_truth_class(img_path)
        
        if true_class is None:
            continue  # Skip if no ground truth
        
        # Get prediction
        predicted_class, _, _ = classify_image(model, img_path)
        
        if predicted_class is None:
            continue  # Skip if no prediction
        
        total += 1
        class_stats[true_class]['total'] += 1
        
        if predicted_class == true_class:
            correct += 1
            class_stats[true_class]['correct'] += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return accuracy, dict(class_stats)


def calculate_accuracy(results):
    """
    Calculate accuracy from validation results.
    For classification, we calculate image-level accuracy.
    """
    # This function is kept for backward compatibility
    # Actual classification accuracy is calculated in validate_model
    return None


def train_detection_model():
    """
    Train a YOLOv8 classification model on the custom dataset.
    """
    print("="*60)
    print("TRAINING YOLOV8 CLASSIFICATION MODEL")
    print("="*60)
    
    # Check if dataset directory exists
    if not DATASET_DIR.exists():
        print(f"\nDataset directory not found: {DATASET_DIR}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    train_dir = DATASET_DIR / 'train'
    if not train_dir.exists():
        print(f"\nDataset training directory not found: {train_dir}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Initialize YOLOv8 classifier
    print("\nInitializing YOLOv8 classification model...")
    model = YOLO("yolov8n-cls.pt")  # Start with pre-trained classification model
    
    # Train the model
    print("\nStarting training...")
    print("Training classification model on folder-based dataset...")
    print(f"Dataset path: {DATASET_DIR}")
    print(f"Training folder: {train_dir}")
    
    results = model.train(
        data=str(DATASET_DIR),    # Path to parent directory containing 'train' folder
        epochs=10,               # Number of training epochs
        imgsz=224,                # Image size for classification
        batch=16,                 # Batch size
        project=str(OUTPUT_DIR),  # Project directory
        name="classification",     # Experiment name
        exist_ok=True,            # Overwrite if exists
        patience=20,              # Early stopping patience
        save=True,                # Save checkpoints
        plots=True,               # Generate training plots
        verbose=True,             # Verbose output
    )
    
    # Save the best model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        shutil.copy(best_model_path, MODEL_PATH)
        print(f"\n✓ Model saved to: {MODEL_PATH}")
        print(f"✓ Best model from: {best_model_path}")
    else:
        # Try alternative location
        best_model_path = Path(results.save_dir) / "best.pt"
        if best_model_path.exists():
            shutil.copy(best_model_path, MODEL_PATH)
            print(f"\n✓ Model saved to: {MODEL_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Print results summary
    if hasattr(results, 'results_dict'):
        print("\nTraining Results:")
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return results


def validate_model(model_path=MODEL_PATH):
    """
    Validate the trained model on the validation set and calculate classification accuracy
    """
    print("="*60)
    print("VALIDATING MODEL (CLASSIFICATION)")
    print("="*60)
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None, None
    
    # Load the trained model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Calculate classification accuracy
    print("\nCalculating classification accuracy...")
    accuracy, class_stats = calculate_classification_accuracy(model, dataset_split='valid')
    
    print("\n" + "="*60)
    print("CLASSIFICATION VALIDATION RESULTS")
    print("="*60)
    
    if accuracy is not None:
        print(f"\nOverall Classification Accuracy: {accuracy:.2f}%")
        
        if class_stats:
            print("\nPer-Class Statistics:")
            for class_name, stats in class_stats.items():
                class_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"  {class_name}: {stats['correct']}/{stats['total']} = {class_acc:.2f}%")
        
        print("\n" + "="*60)
        
        if accuracy >= MIN_ACCURACY:
            print(f"✅ SUCCESS: Accuracy ({accuracy:.2f}%) meets the minimum requirement ({MIN_ACCURACY}%)!")
            print("   You have PASSED the threshold!")
        else:
            print(f"❌ FAILED: Accuracy ({accuracy:.2f}%) is below the minimum requirement ({MIN_ACCURACY}%)")
            print("   You need to improve the model to pass.")
            print("\n   Suggestions:")
            print("   - Train for more epochs")
            print("   - Increase image size")
            print("   - Add more training data")
            print("   - Try a larger model (yolov8m.pt or yolov8l.pt)")
    else:
        print("\n⚠️  Could not calculate classification accuracy.")
        print("   Make sure validation images and labels exist.")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return accuracy, class_stats


def classify(image_path, model_path=None):
    """
    Simple classification function that returns only the class string.
    
    Args:
        image_path: Path to image file (str or Path)
        model_path: Optional path to model file (defaults to MODEL_PATH)
    
    Returns:
        str: 'cola', 'fanta', or 'sprite'
    
    Example:
        >>> result = classify("path/to/image.jpg")
        >>> print(result)  # 'cola', 'fanta', or 'sprite'
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(str(model_path))
    result = classify_image_simple(model, image_path)
    
    if result is None:
        raise ValueError("No classification could be determined from the image")
    
    return result


def main():
    """
    Main entry point
    """
    import sys
    
    print("="*60)
    print("YOLOV8 SOFT DRINK CLASSIFICATION")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--validate" or sys.argv[1] == "-v":
            validate_model()
        elif sys.argv[1] == "--train" or sys.argv[1] == "-t":
            train_detection_model()
        elif sys.argv[1] == "--classify" or sys.argv[1] == "-c":
            # Simple classification mode - just output the class string
            if len(sys.argv) < 3:
                print("Usage: python main.py --classify <image_path>")
                return
            try:
                result = classify(sys.argv[2])
                print(result)  # Output only the string
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print_help()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print_help()
    else:
        # Default: try to train
        if MODEL_PATH.exists():
            print(f"Model already exists: {MODEL_PATH}")
            response = input("Do you want to retrain? (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Use --validate to validate existing model.")
                return
        
        train_detection_model()


def print_help():
    """Print help information"""
    print("\nUsage: python main.py [option]")
    print("\nOptions:")
    print("  -t, --train         Train the model")
    print("  -v, --validate      Validate the model and check accuracy")
    print("  -c, --classify PATH Classify an image (outputs: 'cola', 'fanta', or 'sprite')")
    print("  -h, --help          Show this help message")
    print("\nExample:")
    print("  python main.py -t                    # Train the model")
    print("  python main.py -v                    # Validate the model and check if accuracy >= 90%")
    print("  python main.py -c path/to/image.jpg  # Classify an image (outputs only the class string)")


if __name__ == "__main__":
    main()

