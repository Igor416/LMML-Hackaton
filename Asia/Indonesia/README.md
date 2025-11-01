# Adversarial Attack for Panda Classification

This script crafts an adversarial perturbation that causes a classifier to predict "Panda" with high confidence when applied to a non-panda image.

## Usage

### Basic Usage (with model file)

```bash
python main.py --image sample.png --model pd_model.h5
```

This will:
1. Load the input image
2. Load the Keras model (if available)
3. Craft an adversarial perturbation using PGD (Projected Gradient Descent)
4. Save `perturbation.npy` in the current directory

### Without Model File

If you don't have the model file locally, the script will use a fallback method (less effective):

```bash
python main.py --image sample.png
```

### Parameters

- `--image`: **Required**. Path to input image (PNG/JPG)
- `--model`: Path to Keras model file (`pd_model.h5`). Optional but highly recommended for best results
- `--max_amp`: Maximum perturbation amplitude (default: 0.2, range: [-0.2, 0.2])
- `--iterations`: Number of PGD iterations (default: 20, increase for better results)
- `--alpha`: Step size per iteration (default: 0.01)
- `--size`: Image size (default: 224)
- `--out_dir`: Optional output directory for visualization images

### Examples

**High-quality attack (more iterations):**
```bash
python main.py --image sample.png --model pd_model.h5 --iterations 50 --alpha 0.015
```

**Quick test:**
```bash
python main.py --image sample.png --model pd_model.h5 --iterations 10
```

**With visualization output:**
```bash
python main.py --image sample.png --model pd_model.h5 --out_dir results/
```

## Output

The script generates `perturbation.npy` with the following properties:
- **Shape**: (224, 224, 3)
- **Dtype**: float32
- **Value range**: [-0.2, 0.2]
- **File format**: NumPy binary format (use `np.load('perturbation.npy')` to load)

## How It Works

The script uses **PGD (Projected Gradient Descent)** adversarial attack:

1. **Loads the model** (if provided)
2. **Iteratively updates perturbation** to maximize panda class probability:
   - Computes gradients of the loss with respect to the perturbation
   - Updates perturbation in the direction that increases panda probability
   - Clips perturbation to stay within [-0.2, 0.2] bounds
3. **Saves the final perturbation** as `perturbation.npy`

### Attack Algorithm

- **PGD (Projected Gradient Descent)**: Iterative gradient-based attack
- **Targeted**: Maximizes panda class probability (or max logit if panda class unknown)
- **Projection**: Constrains perturbation to L∞ ball of radius 0.2

## Requirements

```bash
pip install numpy pillow tensorflow
```

Or without TensorFlow (less effective):

```bash
pip install numpy pillow
```

Note: TensorFlow requires Python 3.11 or earlier (not 3.14+)

## Troubleshooting

**"TensorFlow not available"**: 
- Install TensorFlow with `pip install tensorflow` 
- Or use Python 3.11/3.12 (TensorFlow doesn't support 3.14+ yet)

**Low confidence predictions**:
- Increase `--iterations` (e.g., 50 or 100)
- Increase `--alpha` slightly (e.g., 0.015)
- Ensure you're using the correct model file

**Model loading fails**:
- Verify the model file path is correct
- Check that the model is a Keras `.h5` file
- The script will use fallback method if model can't be loaded

