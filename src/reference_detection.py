
"""Adaptive face detection for reference photograph enrollment."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReferenceFace:
    image: np.ndarray
    landmarks: np.ndarray
    method: str


def choose_dominant_face(boxes, image_shape):
    """Select a clearly dominant, reasonably centered face."""
    if len(boxes) == 0:
        return None

    if len(boxes) == 1:
        return 0

    height, width = image_shape[:2]
    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        areas.append(max(0, x2 - x1) * max(0, y2 - y1))

    order = np.argsort(areas)[::-1]
    largest = int(order[0])
    second = int(order[1])

    ratio = areas[largest] / max(areas[second], 1.0)

    box = boxes[largest]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2

    centered = (
        0.15 * width <= center_x <= 0.85 * width
        and 0.10 * height <= center_y <= 0.90 * height
    )

    # Conservative heuristic, not a guarantee of identity.
    if ratio >= 2.5 and centered:
        return largest

    return None


def detect_reference_face(detector, image):
    """Try normal detection, then padded and resized alternatives.

    Returns the image used for detection and its facial landmarks.
    Padding is applied symmetrically, preserving face geometry.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty reference image")

    height, width = image.shape[:2]

    if min(height, width) < 32:
        raise ValueError("Reference image is too small")

    attempts = [
        ("original", image),
    ]

    # Padding gives close-up faces extra surrounding space.
    for fraction in (0.20, 0.40):
        pad = int(max(height, width) * fraction)

        padded = cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        attempts.append((f"padding_{fraction}", padded))

    ambiguous = False

    for method, candidate in attempts:
        for size in (640, 960):
            detector.prepare(
                ctx_id=-1,
                input_size=(size, size),
                det_thresh=0.35,
            )

            boxes, landmarks = detector.detect(
                candidate,
                max_num=0,
                metric="default",
            )

            if len(boxes) == 0:
                continue

            if landmarks is None or len(landmarks) != len(boxes):
                continue

            index = choose_dominant_face(
                boxes,
                candidate.shape,
            )

            if index is None:
                ambiguous = True
                continue

            selected_landmarks = np.asarray(
                landmarks[index],
                dtype=np.float32,
            )

            if selected_landmarks.shape != (5, 2):
                continue

            if not np.all(np.isfinite(selected_landmarks)):
                continue

            return ReferenceFace(
                image=candidate,
                landmarks=selected_landmarks,
                method=f"{method}, resolution={size}",
            )

    if ambiguous:
        raise ValueError(
            "Multiple faces detected without a clearly dominant face"
        )

    raise ValueError(
        "No usable face found after adaptive detection"
    )
