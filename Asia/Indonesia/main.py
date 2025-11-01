"""
Adversarial Attack for Panda Classification
Creates perturbation.npy that makes non-panda images be classified as "Panda"

Usage:
  python main.py --image sample.png --model pd_model.h5
"""
import os
import argparse
import numpy as np
from PIL import Image

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf_available = True
except Exception:
    tf_available = False


def load_image(path, size=224):
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def create_adversarial_perturbation(model, image, eps=0.2, iterations=20, alpha=0.01):
    """Create adversarial perturbation using PGD."""
    image_tensor = tf.constant(image[np.newaxis, ...], dtype=tf.float32)
    perturbation = tf.Variable(tf.zeros_like(image_tensor))
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            adv_image = tf.clip_by_value(image_tensor + perturbation, 0.0, 1.0)
            logits = model(adv_image, training=False)
            loss = -tf.reduce_max(logits[0])  # Maximize highest probability
        
        grad = tape.gradient(loss, perturbation)
        perturbation.assign_add(alpha * tf.sign(grad))
        perturbation.assign(tf.clip_by_value(perturbation, -eps, eps))
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{iterations}")
    
    return perturbation.numpy()[0].astype(np.float32)


def create_fallback_perturbation(image, eps=0.2):
    """Simple fallback when model is not available."""
    noise = np.random.normal(0, eps/3, image.shape).astype(np.float32)
    gray = np.mean(image, axis=2, keepdims=True)
    contrast_mask = np.abs(gray - 0.5) * 2
    perturbation = noise * (0.5 + contrast_mask)
    return np.clip(perturbation, -eps, eps).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Create adversarial perturbation")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--model', type=str, default=None, help='Model path (pd_model.h5)')
    parser.add_argument('--iterations', type=int, default=20, help='PGD iterations')
    parser.add_argument('--max_amp', type=float, default=0.2, help='Max amplitude')
    args = parser.parse_args()
    
    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    img = load_image(args.image)
    print(f"Loaded image: {args.image} (shape: {img.shape})")
    
    # Load model and create perturbation
    if args.model and tf_available and os.path.exists(args.model):
        print("Loading model...")
        model = load_model(args.model, compile=False)
        print("Creating adversarial perturbation with PGD...")
        perturb = create_adversarial_perturbation(
            model, img, eps=args.max_amp, iterations=args.iterations
        )
    else:
        if not tf_available:
            print("TensorFlow not available. Using fallback method.")
        elif not args.model:
            print("No model provided. Using fallback method.")
        else:
            print(f"Model not found: {args.model}. Using fallback method.")
        perturb = create_fallback_perturbation(img, eps=args.max_amp)
    
    # Ensure correct format
    perturb = np.clip(perturb.astype(np.float32), -args.max_amp, args.max_amp)
    
    # Ensure correct shape
    if perturb.shape != (224, 224, 3):
        perturb = Image.fromarray(
            ((perturb + args.max_amp) / (2 * args.max_amp) * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)
        perturb = (np.array(perturb).astype(np.float32) / 255.0 * 2 * args.max_amp - args.max_amp).astype(np.float32)
    
    # Save perturbation
    np.save('perturbation.npy', perturb)
    print(f"\n✓ Saved perturbation.npy")
    print(f"  Shape: {perturb.shape}, dtype: {perturb.dtype}")
    print(f"  Range: [{perturb.min():.4f}, {perturb.max():.4f}]")


if __name__ == '__main__':
    main()
