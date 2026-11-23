
"""Offline face detection using a local SCRFD ONNX model."""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import argparse
import time
from pathlib import Path

import cv2
from insightface.model_zoo import get_model


def detect_faces(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    detection_size: int = 640,
) -> int:
    """Detect faces, draw bounding boxes, and save the result."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Local model not found: {model_path}\n"
            "Download and place the model before running offline."
        )

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    start = time.perf_counter()

    detector = get_model(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    if detector is None:
        raise RuntimeError("Failed to load the local detection model.")

    detector.prepare(
        ctx_id=-1,
        input_size=(detection_size, detection_size),
        det_thresh=0.5,
    )

    boxes, landmarks = detector.detect(
        image,
        max_num=0,
        metric="default",
    )

    elapsed = time.perf_counter() - start

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box[:4].astype(int)
        score = float(box[4])

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2,
        )

        label = f"Face {index}: {score:.2f}"

        cv2.putText(
            image,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save output: {output_path}")

    print(f"Detected faces: {len(boxes)}")
    print(f"Elapsed time (including model loading): {elapsed:.3f}s")
    print(f"Saved annotated image: {output_path}")

    return len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline SCRFD face detection"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/detector.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detected.jpg"),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        choices=[320, 640, 960, 1280],
    )

    args = parser.parse_args()

    detect_faces(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        detection_size=args.size,
    )


if __name__ == "__main__":
    main()
