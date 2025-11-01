"""
Standalone script to calculate accuracy from model.pt
Caches results to avoid recalculating if model hasn't changed
"""

import os
from pathlib import Path
import torch
import functools
import json
import hashlib

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
DATASET_YAML = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/hidden/dataset/data.yaml")
MODEL_PATH = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/model.pt")
CACHE_FILE = Path("D:/Coding/Python/Practice/LMML-Hackaton/Europa/romania/.accuracy_cache.json")
MIN_ACCURACY = 90.0


def get_file_hash(file_path):
    """Calculate MD5 hash of file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_cached_accuracy():
    """Load cached accuracy if model hasn't changed"""
    if not CACHE_FILE.exists():
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        # Check if cached model path matches
        if cache.get('model_path') != str(MODEL_PATH):
            return None
        
        # Check if model file still exists
        if not MODEL_PATH.exists():
            return None
        
        # Check if model file has changed (by hash)
        current_hash = get_file_hash(MODEL_PATH)
        if cache.get('model_hash') == current_hash:
            return cache.get('accuracy'), cache.get('metrics', {})
        
        return None
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def save_cached_accuracy(accuracy, metrics):
    """Save accuracy results to cache"""
    model_hash = get_file_hash(MODEL_PATH) if MODEL_PATH.exists() else None
    
    cache = {
        'model_path': str(MODEL_PATH),
        'model_hash': model_hash,
        'accuracy': accuracy,
        'metrics': metrics,
    }
    
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass  # Fail silently if can't write cache


def calculate_accuracy(model):
    """
    Calculate classification accuracy from model validation.
    """
    try:
        from main import calculate_classification_accuracy
        
        # Calculate classification accuracy
        accuracy, class_stats = calculate_classification_accuracy(model, dataset_split='valid')
        
        # Convert class_stats to metrics_dict format
        metrics_dict = {'classification_accuracy': accuracy / 100.0 if accuracy else 0.0}
        if class_stats:
            for class_name, stats in class_stats.items():
                class_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                metrics_dict[f'{class_name}_accuracy'] = class_acc / 100.0
                metrics_dict[f'{class_name}_total'] = stats['total']
                metrics_dict[f'{class_name}_correct'] = stats['correct']
        
        return accuracy, metrics_dict
    except Exception as e:
        print(f"Error calculating classification accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


def get_accuracy(force_recalculate=False):
    """
    Get accuracy from model.pt, using cache if available.
    
    Args:
        force_recalculate: If True, recalculate even if cache exists
    
    Returns:
        tuple: (accuracy, metrics_dict)
    """
    # Check cache first
    if not force_recalculate:
        cached = load_cached_accuracy()
        if cached is not None:
            accuracy, metrics = cached
            print("📦 Using cached accuracy results (model hasn't changed)")
            return accuracy, metrics
    
    # Validate model file exists
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        return None, {}
    
    print("🔄 Calculating classification accuracy (this may take a moment)...")
    print(f"   Model: {MODEL_PATH}")
    
    # Load model
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, {}
    
    # Calculate classification accuracy
    try:
        accuracy, metrics = calculate_accuracy(model)
        
        # Cache results
        if accuracy is not None:
            save_cached_accuracy(accuracy, metrics)
        
        return accuracy, metrics
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


def main():
    """Main entry point"""
    import sys
    
    force_recalculate = '--force' in sys.argv or '-f' in sys.argv
    
    print("="*60)
    print("MODEL ACCURACY CHECKER")
    print("="*60)
    
    accuracy, metrics = get_accuracy(force_recalculate=force_recalculate)
    
    if accuracy is None:
        print("\n⚠️  Could not calculate accuracy.")
        if metrics:
            print("\nAvailable metrics (raw):")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)
    
    # Show all metrics
    if metrics:
        print("\nAll Metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Show main accuracy
    print("\n" + "="*60)
    print(f"CLASSIFICATION ACCURACY: {accuracy:.2f}%")
    print("="*60)
    
    # Check if passed
    if accuracy >= MIN_ACCURACY:
        print(f"\n✅ SUCCESS: Accuracy ({accuracy:.2f}%) meets the minimum requirement ({MIN_ACCURACY}%)!")
        print("   You have PASSED the threshold!")
    else:
        print(f"\n❌ FAILED: Accuracy ({accuracy:.2f}%) is below the minimum requirement ({MIN_ACCURACY}%)")
        print("   You need to improve the model to pass.")
        print(f"\n   Missing: {MIN_ACCURACY - accuracy:.2f}% to pass")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
