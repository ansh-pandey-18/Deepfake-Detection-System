"""
Video processing utilities for deepfake detection.

Provides frame extraction (with smart filtering), prediction aggregation,
rolling-window smoothing, and confidence-weighted final decisions.
These functions are separated from UI logic to allow future extensions
such as Grad-CAM overlays, timeline visualization, and confidence graphs.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# ============================================================
# Frame Quality Filters
# ============================================================
def is_blurry(frame_rgb, threshold=30.0):
    """
    Detect if a frame is too blurry for reliable prediction.

    Uses Laplacian variance: low variance = blurry image.
    Default threshold is conservative (30) to avoid false rejections.

    Args:
        frame_rgb: RGB numpy array (H, W, 3).
        threshold: Variance below this value is considered blurry.

    Returns:
        True if the frame is blurry, False otherwise.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_near_duplicate(frame_rgb, prev_frame_rgb, threshold=2.0):
    """
    Detect if a frame is nearly identical to the previous frame.

    Uses mean absolute pixel difference. Only flags true duplicates
    (>95% similarity) to avoid over-filtering.

    Args:
        frame_rgb: Current frame (RGB numpy array).
        prev_frame_rgb: Previous frame (RGB numpy array).
        threshold: Mean difference below this is a near-duplicate.
            Default 2.0 catches only >95% identical frames.

    Returns:
        True if frames are near-duplicates, False otherwise.
    """
    if prev_frame_rgb is None:
        return False

    # Resize both to same small size for fast comparison
    curr_small = cv2.resize(frame_rgb, (64, 64))
    prev_small = cv2.resize(prev_frame_rgb, (64, 64))

    mean_diff = np.mean(np.abs(curr_small.astype(float) - prev_small.astype(float)))
    return mean_diff < threshold


# ============================================================
# Frame Extraction
# ============================================================
def extract_frames(video_path, frame_interval=5):
    """
    Extract frames from a video at regular intervals (basic version).

    Args:
        video_path: Path to the video file on disk.
        frame_interval: Extract every nth frame (default: 5).

    Returns:
        List of dicts, each containing:
            - 'frame': RGB numpy array (H, W, 3)
            - 'index': original frame number in the video
            - 'timestamp': time position in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback default

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps
            frames.append({
                'frame': rgb_frame,
                'index': frame_idx,
                'timestamp': timestamp
            })
        frame_idx += 1

    cap.release()
    return frames


def extract_frames_smart(video_path, frame_interval=5, blur_threshold=30.0, dup_threshold=2.0,
                         min_frames=5):
    """
    Extract frames with smart quality filtering and automatic fallback.

    Skips blurry and near-duplicate frames. If filtering removes too many
    frames (fewer than min_frames remain), automatically falls back to
    basic extraction with no filtering to guarantee usable output.

    Args:
        video_path: Path to the video file on disk.
        frame_interval: Extract every nth frame.
        blur_threshold: Laplacian variance threshold for blur detection.
        dup_threshold: Mean pixel difference threshold for duplicate detection.
        min_frames: Minimum frames required. Falls back to basic extraction
            if fewer than this many frames survive filtering.

    Returns:
        Tuple of (frames_list, skipped_blurry, skipped_duplicate, used_fallback)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames = []
    frame_idx = 0
    prev_frame = None
    skipped_blurry = 0
    skipped_duplicate = 0
    total_candidates = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            total_candidates += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Skip blurry frames
            if is_blurry(rgb_frame, threshold=blur_threshold):
                skipped_blurry += 1
                frame_idx += 1
                continue

            # Skip near-duplicate frames
            if is_near_duplicate(rgb_frame, prev_frame, threshold=dup_threshold):
                skipped_duplicate += 1
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            frames.append({
                'frame': rgb_frame,
                'index': frame_idx,
                'timestamp': timestamp
            })
            prev_frame = rgb_frame

        frame_idx += 1

    cap.release()

    # SAFETY FALLBACK: if filtering removed too many frames, retry without filtering
    if len(frames) < min_frames and total_candidates > 0:
        print(f"Smart extraction yielded only {len(frames)} frames "
              f"(skipped {skipped_blurry} blurry, {skipped_duplicate} duplicate). "
              f"Falling back to basic extraction.")
        fallback_frames = extract_frames(video_path, frame_interval=frame_interval)
        return fallback_frames, skipped_blurry, skipped_duplicate, True

    return frames, skipped_blurry, skipped_duplicate, False


# ============================================================
# Frame-to-FileObj Bridge (for reusing existing preprocessing)
# ============================================================
def frame_to_fileobj(frame_rgb):
    """
    Convert an RGB numpy array frame to a file-like BytesIO object.

    This bridge allows reuse of the existing Keras load_img-based
    preprocessing functions that expect a file path or file-like object.

    Args:
        frame_rgb: RGB numpy array of shape (H, W, 3).

    Returns:
        BytesIO object containing the frame encoded as PNG.
    """
    img = Image.fromarray(frame_rgb.astype('uint8'))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# ============================================================
# Prediction
# ============================================================
def predict_frame(frame_rgb, model_name, preprocess_functions):
    """
    Predict whether a single frame is real or AI-generated.

    Converts the numpy frame to a file-like object and passes it through
    the existing preprocessing + prediction pipeline (same functions used
    for image detection).

    Args:
        frame_rgb: RGB numpy array of shape (H, W, 3).
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.
        preprocess_functions: Dict mapping model_name to the
            corresponding preprocess-and-predict function.

    Returns:
        Dict with:
            - 'label': 'REAL' or 'AI Generated'
            - 'confidence': float sigmoid output
    """
    file_obj = frame_to_fileobj(frame_rgb)
    preprocess_fn = preprocess_functions[model_name]
    prediction = preprocess_fn(file_obj)
    confidence = float(prediction[0][0])
    label = "REAL" if confidence >= 0.5 else "AI Generated"
    return {
        'label': label,
        'confidence': confidence
    }


def predict_frame_with_tta(frame_rgb, model_name, run_model_fn):
    """
    Predict a single frame using Test-Time Augmentation.

    Uses the preprocessing module's TTA pipeline for more stable
    per-frame predictions.

    Args:
        frame_rgb: RGB numpy array of shape (H, W, 3).
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.
        run_model_fn: Raw model inference function.

    Returns:
        Dict with 'label', 'confidence', and TTA details.
    """
    from preprocessing import predict_with_tta
    file_obj = frame_to_fileobj(frame_rgb)
    result = predict_with_tta(file_obj, model_name, run_model_fn)
    return {
        'label': result['label'],
        'confidence': result['confidence']
    }


# ============================================================
# Prediction Smoothing
# ============================================================
def smooth_predictions(results, window=5):
    """
    Apply rolling-window smoothing to frame predictions.

    Averages confidence scores over a sliding window to reduce
    noise from frame-to-frame prediction jitter.

    Args:
        results: List of prediction dicts with 'confidence' key.
        window: Size of the rolling window.

    Returns:
        List of smoothed prediction dicts (same format, updated confidence + label).
    """
    if len(results) <= 1:
        return results

    confidences = [r['confidence'] for r in results]
    smoothed = []

    for i in range(len(confidences)):
        # Window: from max(0, i - window//2) to min(len, i + window//2 + 1)
        start = max(0, i - window // 2)
        end = min(len(confidences), i + window // 2 + 1)
        avg_conf = np.mean(confidences[start:end])

        smoothed_result = dict(results[i])  # copy original
        smoothed_result['raw_confidence'] = smoothed_result['confidence']
        smoothed_result['confidence'] = float(avg_conf)
        smoothed_result['label'] = "REAL" if avg_conf >= 0.5 else "AI Generated"
        smoothed.append(smoothed_result)

    return smoothed


# ============================================================
# Video-Level Prediction
# ============================================================
def predict_video(frames, model_name, preprocess_functions, progress_callback=None,
                  use_tta=False, run_model_fn=None, smoothing_window=5):
    """
    Aggregate frame-level predictions into a final video-level prediction.

    Enhanced with:
        - Optional TTA per frame
        - Rolling-window prediction smoothing
        - Confidence-weighted final decision (mean of confidences)

    Args:
        frames: List of frame dicts from extract_frames() or extract_frames_smart().
        model_name: One of 'CNN', 'Efficientnet', 'Efficientnet Art'.
        preprocess_functions: Dict mapping model_name to preprocess+predict function.
        progress_callback: Optional callable(current, total) for UI updates.
        use_tta: If True, use TTA for each frame prediction.
        run_model_fn: Required if use_tta=True. The raw model inference function.
        smoothing_window: Size of rolling window for smoothing (default: 5).

    Returns:
        Dict with:
            - final_label: 'REAL' or 'AI Generated'
            - total_frames: int
            - fake_count: int
            - real_count: int
            - fake_percentage: float
            - real_percentage: float
            - weighted_confidence: float (mean of all frame confidences)
            - frame_results: list of per-frame result dicts
    """
    results = []
    total = len(frames)

    for i, frame_data in enumerate(frames):
        try:
            if use_tta and run_model_fn is not None:
                result = predict_frame_with_tta(
                    frame_data['frame'], model_name, run_model_fn
                )
            else:
                result = predict_frame(
                    frame_data['frame'], model_name, preprocess_functions
                )
        except Exception as e:
            # Skip frames that fail prediction (corrupt, etc.)
            print(f"Warning: Frame {frame_data['index']} prediction failed: {e}")
            result = {'label': 'Unknown', 'confidence': 0.5}

        result['index'] = frame_data['index']
        result['timestamp'] = frame_data['timestamp']
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    # Filter out failed predictions for final decision
    valid_results = [r for r in results if r['label'] != 'Unknown']

    if len(valid_results) == 0:
        return {
            'final_label': 'Unknown',
            'total_frames': total,
            'fake_count': 0,
            'real_count': 0,
            'fake_percentage': 0.0,
            'real_percentage': 0.0,
            'weighted_confidence': 0.5,
            'frame_results': results
        }

    # Apply rolling-window smoothing
    smoothed_results = smooth_predictions(valid_results, window=smoothing_window)

    # Count after smoothing
    fake_count = sum(1 for r in smoothed_results if r['label'] == 'AI Generated')
    real_count = sum(1 for r in smoothed_results if r['label'] == 'REAL')
    total_valid = len(smoothed_results)
    fake_percentage = (fake_count / total_valid * 100) if total_valid > 0 else 0
    real_percentage = (real_count / total_valid * 100) if total_valid > 0 else 0

    # Confidence-weighted final decision
    weighted_confidence = float(np.mean([r['confidence'] for r in smoothed_results]))
    final_label = "REAL" if weighted_confidence >= 0.5 else "AI Generated"

    return {
        'final_label': final_label,
        'total_frames': total_valid,
        'fake_count': fake_count,
        'real_count': real_count,
        'fake_percentage': fake_percentage,
        'real_percentage': real_percentage,
        'weighted_confidence': weighted_confidence,
        'frame_results': smoothed_results
    }
