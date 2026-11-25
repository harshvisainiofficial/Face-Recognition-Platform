
"""Offline group-photo recognition for FaceRoster.

Enroll reference faces, recognize faces in a group photo,
and export an annotated image and CSV results.
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.utils.face_align import norm_crop

from src.enrollment import FaceEnrollment, read_image


def get_embedding(recognizer, image, landmarks):
    """Align a detected face and produce an L2-normalized embedding."""
    aligned = norm_crop(
        image,
        landmark=np.asarray(landmarks, dtype=np.float32),
        image_size=112,
    )

    embedding = np.asarray(
        recognizer.get_feat(aligned),
        dtype=np.float32,
    ).reshape(-1)

    if embedding.shape != (512,):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid embedding norm")

    embedding = embedding / norm

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Non-finite embedding")

    return embedding


def match_identity(embedding, gallery, threshold):
    """Return the closest enrolled identity or Unknown."""
    if not gallery:
        raise ValueError("Reference gallery is empty")

    names = list(gallery.keys())
    gallery_matrix = np.stack(
        [gallery[name] for name in names],
        axis=0,
    )

    # All vectors are L2-normalized, so dot product = cosine similarity.
    similarities = gallery_matrix @ embedding

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])
    best_name = names[best_index]

    assigned_name = (
        best_name if best_score >= threshold else "Unknown"
    )

    return assigned_name, best_name, best_score


def draw_label(image, box, label, face_id):
    """Draw a face bounding box and readable identity label."""
    height, width = image.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box[:4]]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    color = (
        (0, 170, 0) if label != "Unknown"
        else (0, 140, 255)
    )

    cv2.rectangle(
        image, (x1, y1), (x2, y2), color, 2
    )

    text = f"#{face_id} {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(0.45, min(0.8, width / 1400))
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Place above the face when possible, otherwise below.
    if y1 >= text_height + baseline + 12:
        top = y1 - text_height - baseline - 10
    else:
        top = min(y2 + 5, height - text_height - baseline - 8)

    left = max(0, min(x1, width - text_width - 10))
    right = min(width - 1, left + text_width + 10)
    bottom = min(height - 1, top + text_height + baseline + 8)

    cv2.rectangle(
        image, (left, top), (right, bottom), color, -1
    )

    cv2.putText(
        image,
        text,
        (left + 5, top + text_height + 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def recognize_group(
    references,
    image_path,
    output_path,
    csv_path,
    threshold=0.45,
    detection_size=640,
):
    """Recognize group-photo faces using locally enrolled references."""

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between -1 and 1")

    start = time.perf_counter()

    enrollment = FaceEnrollment()
    enrollment.detector.prepare(
        ctx_id=-1,
        input_size=(detection_size, detection_size),
        det_thresh=0.5,
    )

    gallery = enrollment.enroll_folder(str(references))

    if not gallery:
        raise RuntimeError(
            "No identities enrolled. Check the reference images."
        )

    image = read_image(Path(image_path))

    if image is None:
        raise ValueError(
            f"Cannot read group image: {image_path}"
        )

    boxes, landmarks = enrollment.detector.detect(
        image,
        max_num=0,
        metric="default",
    )

    if landmarks is None and len(boxes) > 0:
        raise RuntimeError("Detector returned no face landmarks")

    if landmarks is not None and len(landmarks) != len(boxes):
        raise RuntimeError("Face boxes and landmarks count mismatch")

    # Stable face numbering: top-to-bottom, then left-to-right.
    indices = sorted(
        range(len(boxes)),
        key=lambda i: (
            int(boxes[i][1]) // 80,
            float(boxes[i][0]),
        ),
    )

    annotated = image.copy()
    results = []

    print("\n========== RECOGNITION RESULTS ==========")

    for face_id, index in enumerate(indices, start=1):
        box = boxes[index]
        face_landmarks = landmarks[index]

        embedding = get_embedding(
            enrollment.recognizer,
            image,
            face_landmarks,
        )

        assigned_name, best_candidate, score = match_identity(
            embedding,
            gallery,
            threshold,
        )

        draw_label(
            annotated,
            box,
            assigned_name,
            face_id,
        )

        row = {
            "face_id": face_id,
            "name": assigned_name,
            "best_candidate": best_candidate,
            "cosine_similarity": round(score, 6),
            "detection_score": round(float(box[4]), 6),
            "x1": int(box[0]),
            "y1": int(box[1]),
            "x2": int(box[2]),
            "y2": int(box[3]),
        }

        results.append(row)

        print(
            f"Face {face_id}: {assigned_name} | "
            f"Best candidate: {best_candidate} | "
            f"Cosine similarity: {score:.4f}"
        )

    output_path = Path(output_path)
    csv_path = Path(csv_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(
            f"Could not save annotated image: {output_path}"
        )

    fields = [
        "face_id",
        "name",
        "best_candidate",
        "cosine_similarity",
        "detection_score",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    known_count = sum(
        row["name"] != "Unknown" for row in results
    )
    unknown_count = len(results) - known_count

    elapsed = time.perf_counter() - start

    print("\n========== SUMMARY ==========")
    print(f"Enrolled identities: {len(gallery)}")
    print(f"Detected faces: {len(results)}")
    print(f"Known assignments: {known_count}")
    print(f"Unknown assignments: {unknown_count}")
    print(f"Total time including enrollment: {elapsed:.3f}s")
    print(f"Annotated image: {output_path}")
    print(f"CSV results: {csv_path}")
    print("=============================")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Offline FaceRoster group recognition"
    )

    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/references"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/recognized.jpg"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/recognition_results.csv"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        choices=[320, 640, 960, 1280],
    )

    args = parser.parse_args()

    recognize_group(
        references=args.references,
        image_path=args.image,
        output_path=args.output,
        csv_path=args.csv,
        threshold=args.threshold,
        detection_size=args.size,
    )


if __name__ == "__main__":
    main()
