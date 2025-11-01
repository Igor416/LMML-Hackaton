"""
YOLOv8 Object Detection Model Training
Train an object detection model to detect soft drink brands: cola, fanta, and sprite
"""

import os
from pathlib import Path
from ultralytics import YOLO
import shutil


# Configuration
DATASET_YAML = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/dataset/data.yaml")
OUTPUT_DIR = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/output")
MODEL_PATH = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/model.pt")


def train_detection_model():
    """
    Train a YOLOv8 object detection model on the custom dataset
    """
    print("="*60)
    print("TRAINING YOLOV8 OBJECT DETECTION MODEL")
    print("="*60)
    
    # Check if dataset YAML exists
    if not DATASET_YAML.exists():
        print(f"\nDataset YAML not found: {DATASET_YAML}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Initialize YOLOv8 detector
    print("\nInitializing YOLOv8 object detection model...")
    model = YOLO("yolov8n.pt")  # Start with pre-trained nano model
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=str(DATASET_YAML),  # Path to dataset YAML
        epochs=10,               # Number of training epochs
        imgsz=640,                # Image size
        batch=16,                 # Batch size
        project=str(OUTPUT_DIR),  # Project directory
        name="detection",         # Experiment name
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
    Validate the trained model on the validation set
    """
    print("="*60)
    print("VALIDATING MODEL")
    print("="*60)
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    # Load the trained model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Validate using the dataset YAML
    print("\nRunning validation...")
    results = model.val(data=str(DATASET_YAML))
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return results


def main():
    """
    Main entry point
    """
    import sys
    
    print("="*60)
    print("YOLOV8 SOFT DRINK OBJECT DETECTION")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--validate" or sys.argv[1] == "-v":
            validate_model()
        elif sys.argv[1] == "--train" or sys.argv[1] == "-t":
            train_detection_model()
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
    print("  -t, --train      Train the model")
    print("  -v, --validate   Validate the model")
    print("  -h, --help       Show this help message")
    print("\nExample:")
    print("  python main.py -t      # Train the model")
    print("  python main.py -v      # Validate the model")


if __name__ == "__main__":
    main()

