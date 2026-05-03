"""
Grad-CAM explainability module for deepfake detection.

Generates class activation heatmaps to visualize which regions
of an image most influenced the model's prediction.
Supports CNN, EfficientNet, and EfficientNet Art models.

No model weights are modified — this is inference-time only.
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


# ============================================================
# CNN Model Builder (cached separately for Grad-CAM access)
# ============================================================
_cached_cnn_model = None


def get_cnn_model(weights_path):
    """
    Build and cache the CNN model for Grad-CAM.

    Uses the exact same architecture as run_cnn() in app.py,
    but keeps the model object in memory for gradient computation.

    Args:
        weights_path: Path to CNN weights file.

    Returns:
        Compiled Keras Sequential model with loaded weights.
    """
    global _cached_cnn_model
    if _cached_cnn_model is not None:
        return _cached_cnn_model

    model = Sequential()
    model.add(Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        input_shape=(256, 256, 3)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.09))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(weights_path)
    _cached_cnn_model = model
    return model


# ============================================================
# Fallback layer names per model type
# ============================================================
FALLBACK_LAYERS = {
    'CNN': None,               # auto-detect last Conv2D
    'Efficientnet': 'top_conv',
    'Efficientnet Art': 'top_conv',
}


# ============================================================
# Layer Detection
# ============================================================
def get_last_conv_layer(model):
    """
    Automatically find the last convolutional layer in a model.

    Handles both simple Sequential models and complex architectures
    like EfficientNet with nested layers.

    Args:
        model: Keras model.

    Returns:
        Name of the last Conv2D layer, or None if not found.
    """
    last_conv = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        # Handle nested models (e.g., EfficientNet wrapped in a functional model)
        elif hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv = sub_layer.name

    return last_conv


def _find_conv_layer_in_model(model, layer_name):
    """
    Find a layer by name, searching nested models if needed.

    Returns the layer object and the model that contains it.
    """
    # Direct lookup first
    try:
        layer = model.get_layer(layer_name)
        return layer, model
    except ValueError:
        pass

    # Search nested models
    for parent_layer in model.layers:
        if hasattr(parent_layer, 'layers'):
            try:
                layer = parent_layer.get_layer(layer_name)
                return layer, parent_layer
            except ValueError:
                continue

    return None, None


# ============================================================
# Grad-CAM Generation — Sub-model approach (robust)
# ============================================================
def _generate_gradcam_submodel(model, img_array, last_conv_layer_name):
    """
    Generate Grad-CAM using the sub-model approach.

    Creates a temporary model that outputs both the conv layer activations
    and the final prediction, then uses GradientTape to compute gradients.
    Handles nested models (like EfficientNet wrapped in Sequential) by 
    forward passing through outer layers.

    Args:
        model: Keras model with accessible .input and .output.
        img_array: Preprocessed input array of shape (1, H, W, 3).
        last_conv_layer_name: Name of the target conv layer.

    Returns:
        heatmap: 2D numpy array normalized to [0, 1], or None on failure.
    """
    try:
        # Find the layer and the model that owns it
        conv_layer, owner_model = _find_conv_layer_in_model(model, last_conv_layer_name)
        if conv_layer is None:
            print(f"Warning: Layer '{last_conv_layer_name}' not found in model")
            return None

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            if owner_model == model:
                # Direct sub-model extraction
                grad_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=[conv_layer.output, model.output]
                )
                conv_outputs, predictions = grad_model(img_tensor)
            else:
                # Nested model extraction: replace inner model with grad_model
                grad_model = tf.keras.Model(
                    inputs=owner_model.input,
                    outputs=[conv_layer.output, owner_model.output]
                )
                
                # Manual forward pass of the top-level model
                x = img_tensor
                conv_outputs = None
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    if layer == owner_model:
                        conv_outputs, x = grad_model(x)
                    else:
                        x = layer(x)
                predictions = x

            loss = predictions[:, 0]

        # Gradients of the output w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("Warning: Gradients are None — sub-model approach failed")
            return None

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv feature maps by the pooled gradients
        conv_output_val = conv_outputs[0]
        heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + safe normalize
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        print(f"Grad-CAM sub-model approach failed: {e}")
        return None


# ============================================================
# Grad-CAM Generation — Manual forward pass (fallback for Sequential)
# ============================================================
def _generate_gradcam_manual(model, img_array, last_conv_layer_name):
    """
    Generate Grad-CAM via manual layer-by-layer forward pass.

    Fallback for Sequential models in Keras 3.x where .input/.output
    may not be accessible. Iterates through layers manually.

    Args:
        model: Keras Sequential model.
        img_array: Preprocessed input array of shape (1, H, W, 3).
        last_conv_layer_name: Name of the target conv layer.

    Returns:
        heatmap: 2D numpy array normalized to [0, 1], or None on failure.
    """
    try:
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            current = img_tensor
            conv_output = None

            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                current = layer(current)
                if layer.name == last_conv_layer_name:
                    conv_output = current

            prediction = current
            loss = prediction[:, 0]

        if conv_output is None:
            print(f"Warning: Conv layer '{last_conv_layer_name}' not found during forward pass")
            return None

        grads = tape.gradient(loss, conv_output)

        if grads is None:
            print("Warning: Gradients are None — manual approach failed")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output_val = conv_output[0]
        heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + safe normalize
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        print(f"Grad-CAM manual forward pass failed: {e}")
        return None


# ============================================================
# Unified Grad-CAM with retry + fallback
# ============================================================
def generate_gradcam(model, img_array, last_conv_layer_name=None, model_name=None):
    """
    Generate a Grad-CAM heatmap with automatic retry and fallback logic.

    Strategy:
        1. Try sub-model approach (works for Functional models like EfficientNet)
        2. If that fails, try manual forward pass (works for Sequential CNN)
        3. If first layer detection fails, retry with model-specific fallback layer

    Args:
        model: Keras model.
        img_array: Preprocessed input array of shape (1, H, W, 3).
        last_conv_layer_name: Name of the target conv layer (auto-detected if None).
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art' (for fallback).

    Returns:
        Tuple of (heatmap, used_fallback):
            - heatmap: 2D numpy array normalized to [0,1], or None on failure
            - used_fallback: bool indicating if fallback layer was used
    """
    # Ensure input shape is (1, H, W, 3)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Step 1: detect conv layer
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    if last_conv_layer_name is None:
        print("Warning: No convolutional layer found for Grad-CAM")
        return None, False

    # Step 2: try sub-model approach first (reliable for Functional models)
    heatmap = _generate_gradcam_submodel(model, img_array, last_conv_layer_name)
    if heatmap is not None:
        return heatmap, False

    # Step 3: try manual forward pass (fallback for Sequential)
    heatmap = _generate_gradcam_manual(model, img_array, last_conv_layer_name)
    if heatmap is not None:
        return heatmap, False

    # Step 4: retry with model-specific fallback layer
    if model_name and model_name in FALLBACK_LAYERS:
        fallback_layer = FALLBACK_LAYERS[model_name]
        if fallback_layer and fallback_layer != last_conv_layer_name:
            print(f"Retrying Grad-CAM with fallback layer: {fallback_layer}")
            heatmap = _generate_gradcam_submodel(model, img_array, fallback_layer)
            if heatmap is not None:
                return heatmap, True
            heatmap = _generate_gradcam_manual(model, img_array, fallback_layer)
            if heatmap is not None:
                return heatmap, True

    return None, False


# ============================================================
# Heatmap Overlay
# ============================================================
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: PIL Image (RGB) or numpy array.
        heatmap: 2D numpy array from generate_gradcam(), normalized [0, 1].
        alpha: Transparency of the heatmap overlay (0 = invisible, 1 = opaque).

    Returns:
        PIL Image with the colored heatmap overlay.
    """
    if isinstance(original_img, Image.Image):
        original_arr = np.array(original_img)
    else:
        original_arr = original_img.copy()

    # Ensure RGB and uint8
    if original_arr.dtype != np.uint8:
        if original_arr.max() <= 1.0:
            original_arr = (original_arr * 255).astype(np.uint8)
        else:
            original_arr = original_arr.astype(np.uint8)

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_arr.shape[1], original_arr.shape[0]))

    # Apply JET colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_arr)

    return Image.fromarray(overlay)


# ============================================================
# High-Level API
# ============================================================
def get_gradcam_for_image(image, model_name, models_dict):
    """
    High-level function: generate Grad-CAM for an uploaded image.

    Handles preprocessing, model selection, heatmap generation,
    and overlay creation in one call. Includes automatic retry
    with fallback conv layer names per model type.

    Args:
        image: File-like object (uploaded image).
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.
        models_dict: Dict with keys:
            - 'cnn_weights_path': str
            - 'eff_net_model': loaded Keras model
            - 'eff_net_art_model': loaded Keras model

    Returns:
        Dict with:
            - 'heatmap': raw 2D heatmap array
            - 'overlay': PIL Image with heatmap overlay
            - 'original': PIL Image (original, resized to model input)
            - 'success': bool
            - 'error': str or None
            - 'used_fallback': bool
    """
    from preprocessing import preprocess_image, validate_image

    if hasattr(image, 'seek'):
        image.seek(0)

    # Validate
    is_valid, pil_img, error = validate_image(image)
    if not is_valid:
        return {'success': False, 'error': error, 'heatmap': None,
                'overlay': None, 'original': None, 'used_fallback': False}

    # Config per model
    config = {
        'CNN': {'target_size': (256, 256), 'normalize': True},
        'Efficientnet': {'target_size': (300, 300), 'normalize': False},
        'Efficientnet Art': {'target_size': (224, 224), 'normalize': False},
    }

    cfg = config[model_name]
    target_size = cfg['target_size']

    # Resize original for overlay
    original_resized = pil_img.resize(target_size, Image.LANCZOS)

    # Preprocess for the model
    img_arr = preprocess_image(pil_img, model_name)

    # Ensure input shape is (1, H, W, 3)
    if len(img_arr.shape) == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    # Get the model object
    try:
        if model_name == 'CNN':
            model = get_cnn_model(models_dict['cnn_weights_path'])
        elif model_name == 'Efficientnet':
            model = models_dict['eff_net_model']
        elif model_name == 'Efficientnet Art':
            model = models_dict['eff_net_art_model']
        else:
            return {'success': False, 'error': f'Unknown model: {model_name}',
                    'heatmap': None, 'overlay': None, 'original': None,
                    'used_fallback': False}
    except Exception as e:
        return {'success': False, 'error': f'Model access failed: {e}',
                'heatmap': None, 'overlay': None, 'original': None,
                'used_fallback': False}

    # Generate Grad-CAM with retry + fallback
    heatmap, used_fallback = generate_gradcam(
        model, img_arr, last_conv_layer_name=None, model_name=model_name
    )

    if heatmap is None:
        return {'success': False, 'error': 'Could not generate Grad-CAM heatmap',
                'heatmap': None, 'overlay': None, 'original': original_resized,
                'used_fallback': False}

    # Create overlay
    overlay_img = overlay_heatmap(original_resized, heatmap, alpha=0.4)

    return {
        'success': True,
        'error': None,
        'heatmap': heatmap,
        'overlay': overlay_img,
        'original': original_resized,
        'used_fallback': used_fallback
    }
