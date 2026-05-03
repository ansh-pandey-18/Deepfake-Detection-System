"""
Preprocessing, validation, TTA, and model recommendation for deepfake detection.

This module enhances prediction accuracy without retraining models by:
- Standardizing preprocessing per model's training pipeline
- Applying Test-Time Augmentation (TTA) for more stable predictions
- Validating inputs to prevent corrupt/invalid data from degrading results
- Recommending the best model based on image characteristics
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# Model configuration — target sizes and normalization
# ============================================================
MODEL_CONFIG = {
    'CNN': {
        'target_size': (256, 256),
        'normalize': True,       # /255.0
        'norm_factor': 255.0,
    },
    'Efficientnet': {
        'target_size': (300, 300),
        'normalize': False,      # raw pixel values (0-255)
        'norm_factor': 1.0,
    },
    'Efficientnet Art': {
        'target_size': (224, 224),
        'normalize': False,      # raw pixel values (0-255)
        'norm_factor': 1.0,
    },
}


# ============================================================
# Input Validation
# ============================================================
def validate_image(image):
    """
    Validate and sanitize an input image.

    Checks:
        - Image can be opened and decoded
        - Converts grayscale / RGBA to RGB
        - Verifies minimum size (at least 32x32)

    Args:
        image: File-like object or file path.

    Returns:
        Tuple (is_valid: bool, pil_image_or_None, error_message_or_None)
    """
    try:
        img = Image.open(image)
        img.load()  # force full decode to catch corruption
    except Exception as e:
        return False, None, f"Could not open image: {str(e)}"

    # Check minimum size
    if img.width < 32 or img.height < 32:
        return False, None, f"Image too small ({img.width}x{img.height}). Minimum is 32x32."

    # Convert to RGB if needed
    if img.mode == 'L':  # grayscale
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        # Composite onto white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    return True, img, None


def _pil_to_fileobj(pil_img):
    """Convert a PIL Image to a BytesIO file-like object."""
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# ============================================================
# Standardized Preprocessing
# ============================================================
def preprocess_image(pil_img, model_name):
    """
    Preprocess a PIL image for a specific model.

    Applies the exact resize and normalization that matches
    each model's training pipeline.

    Args:
        pil_img: PIL Image in RGB mode.
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.

    Returns:
        Numpy array of shape (1, H, W, 3), ready for model.predict().
    """
    config = MODEL_CONFIG[model_name]
    target_size = config['target_size']

    # Resize using high-quality resampling
    img_resized = pil_img.resize(target_size, Image.LANCZOS)
    img_arr = img_to_array(img_resized)

    # Normalize per model's training pipeline
    if config['normalize']:
        img_arr = img_arr / config['norm_factor']

    # Add batch dimension
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


# ============================================================
# Test-Time Augmentation (TTA)
# ============================================================
def _generate_augmented_versions(pil_img):
    """
    Generate augmented variants of an image for TTA.

    Produces 4 versions:
        1. Original
        2. Horizontal flip
        3. Brightness adjusted (+15%)
        4. Contrast adjusted (+15%)

    Args:
        pil_img: PIL Image in RGB mode.

    Returns:
        List of 4 PIL Images.
    """
    augmented = [pil_img]

    # Horizontal flip
    augmented.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))

    # Brightness adjustment (+15%)
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    augmented.append(brightness_enhancer.enhance(1.15))

    # Contrast adjustment (+15%)
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    augmented.append(contrast_enhancer.enhance(1.15))

    return augmented


def predict_with_tta(image, model_name, run_model_fn):
    """
    Run prediction with Test-Time Augmentation for improved accuracy.

    Pipeline:
        1. Validate input image
        2. Generate 4 augmented versions
        3. Preprocess each for the selected model
        4. Run inference on each
        5. Average sigmoid outputs
        6. Return calibrated prediction

    Args:
        image: File-like object or file path.
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.
        run_model_fn: The raw model inference function (run_cnn, run_effNet, etc.)
            that accepts a preprocessed numpy array and returns prediction.

    Returns:
        Dict with:
            - 'prediction': numpy array [[averaged_confidence]]
            - 'label': 'REAL' or 'AI Generated'
            - 'confidence': float
            - 'tta_predictions': list of individual predictions
            - 'is_valid': bool
            - 'error': str or None
    """
    # Validate
    # Reset file pointer if it's a file-like object
    if hasattr(image, 'seek'):
        image.seek(0)

    is_valid, pil_img, error = validate_image(image)
    if not is_valid:
        return {
            'prediction': np.array([[0.5]]),
            'label': 'Unknown',
            'confidence': 0.5,
            'tta_predictions': [],
            'is_valid': False,
            'error': error
        }

    # Generate augmented versions
    augmented_images = _generate_augmented_versions(pil_img)

    # Preprocess and predict each
    tta_predictions = []
    for aug_img in augmented_images:
        preprocessed = preprocess_image(aug_img, model_name)
        pred = run_model_fn(preprocessed)
        tta_predictions.append(float(pred[0][0]))

    # Average predictions for calibrated result
    avg_confidence = np.mean(tta_predictions)

    label = "REAL" if avg_confidence >= 0.5 else "AI Generated"

    return {
        'prediction': np.array([[avg_confidence]]),
        'label': label,
        'confidence': float(avg_confidence),
        'tta_predictions': tta_predictions,
        'is_valid': True,
        'error': None
    }


# ============================================================
# Model Recommendation
# ============================================================
def recommend_model(image):
    """
    Analyze image characteristics and recommend the best model.

    Heuristics:
        - High color saturation + low edge density → artistic image → EfficientNet Art
        - High edge density + natural colors → photorealistic → EfficientNet
        - Otherwise → CNN (general purpose)

    Args:
        image: File-like object or file path.

    Returns:
        Dict with:
            - 'recommended': model name string
            - 'reason': human-readable explanation
            - 'scores': dict of analysis scores
    """
    if hasattr(image, 'seek'):
        image.seek(0)

    try:
        img = Image.open(image)
        img = img.convert('RGB')
    except Exception:
        return {
            'recommended': 'CNN',
            'reason': 'Could not analyze image, defaulting to CNN',
            'scores': {}
        }

    # Resize to a standard size for fast analysis
    analysis_img = img.resize((128, 128), Image.LANCZOS)
    img_arr = np.array(analysis_img, dtype=np.float32)

    # --- Color saturation analysis ---
    # Convert to HSV-like representation to measure saturation
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c
    # Average saturation (higher = more colorful/artistic)
    avg_saturation = np.mean(chroma)

    # --- Edge density analysis ---
    # Simple Sobel-like gradient magnitude
    gray = np.mean(img_arr, axis=2)
    # Horizontal and vertical gradients
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    avg_edge_density = (np.mean(gx) + np.mean(gy)) / 2.0

    # --- Color variance (uniformity) ---
    color_variance = np.std(img_arr)

    # --- Decision logic ---
    scores = {
        'saturation': float(avg_saturation),
        'edge_density': float(avg_edge_density),
        'color_variance': float(color_variance),
    }

    # Artistic images: high saturation or low edge density with high color variance
    if avg_saturation > 60 and avg_edge_density < 15:
        recommended = 'Efficientnet Art'
        reason = 'Image appears artistic (high color saturation, smooth textures)'
    elif avg_saturation > 50 and color_variance > 70:
        recommended = 'Efficientnet Art'
        reason = 'Image has artistic color patterns'
    # Photorealistic: moderate-high edge density, natural color distribution
    elif avg_edge_density > 10 and color_variance > 50:
        recommended = 'Efficientnet'
        reason = 'Image appears photorealistic (natural textures and colors)'
    else:
        recommended = 'CNN'
        reason = 'General-purpose model recommended for this image type'

    return {
        'recommended': recommended,
        'reason': reason,
        'scores': scores
    }
