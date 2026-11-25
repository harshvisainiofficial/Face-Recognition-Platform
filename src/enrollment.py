
"""Reference enrollment for FaceRoster.

Detect one face per image, align it, extract an ArcFace embedding,
and create an in-memory gallery of normalized embeddings.
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_image(path: Path):
    """Read an image, including filenames with Unicode characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except (OSError, ValueError, cv2.error):
        return None


class FaceEnrollment:
    """Enroll reference identities using local ONNX models."""

    def __init__(
        self,
        detector_path: str = "models/detector.onnx",
        recognizer_path: str = "models/recognizer.onnx",
    ):
        for path in (detector_path, recognizer_path):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Required local model not found: {path}"
                )

        print("Loading SCRFD detector...")
        self.detector = get_model(
            detector_path,
            providers=["CPUExecutionProvider"],
        )

        print("Loading ArcFace recognizer...")
        self.recognizer = get_model(
            recognizer_path,
            providers=["CPUExecutionProvider"],
        )

        if self.detector is None or self.recognizer is None:
            raise RuntimeError("Could not load the ONNX models.")

        self.detector.prepare(
            ctx_id=-1,
            input_size=(640, 640),
            det_thresh=0.5,
        )
        self.recognizer.prepare(ctx_id=-1)

        self.gallery = {}
        self.failures = []

        print("Both models loaded successfully!")

    def enroll_folder(self, folder: str):
        """Enroll one reference image per identity."""
        folder_path = Path(folder)

        if not folder_path.is_dir():
            raise NotADirectoryError(
                f"Reference folder not found: {folder}"
            )

        # Replacing a folder replaces the entire reference gallery.
        self.gallery = {}
        self.failures = []

        files = sorted(
            p for p in folder_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"\nFound {len(files)} supported image files.\n")

        for path in files:
            name = path.stem.strip()
            print(f"Processing: {path.name}")

            if not name:
                self.failures.append((path.name, "Empty identity name"))
                continue

            if name.casefold() in (
                registered.casefold() for registered in self.gallery
            ):
                self.failures.append(
                    (path.name, "Duplicate identity name")
                )
                continue

            image = read_image(path)

            if image is None:
                self.failures.append(
                    (path.name, "Invalid or unreadable image")
                )
                print("  FAILED: Cannot read image")
                continue

            try:
                boxes, landmarks = self.detector.detect(
                    image,
                    max_num=0,
                    metric="default",
                )

                if len(boxes) == 0:
                    raise ValueError("No face detected")

                if len(boxes) > 1:
                    raise ValueError(
                        f"Multiple faces detected: {len(boxes)}"
                    )

                if landmarks is None or len(landmarks) != 1:
                    raise ValueError("Face landmarks unavailable")

                # Align using the five facial landmarks from SCRFD.
                aligned = norm_crop(
                    image,
                    landmark=landmarks[0],
                    image_size=112,
                )

                # InsightFace handles the model's required preprocessing.
                embedding = self.recognizer.get_feat(aligned)
                embedding = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1)

                if embedding.shape != (512,):
                    raise ValueError(
                        f"Unexpected embedding shape: {embedding.shape}"
                    )

                norm = np.linalg.norm(embedding)

                if not np.isfinite(norm) or norm <= 1e-12:
                    raise ValueError("Invalid embedding norm")

                normalized = embedding / norm

                if not np.all(np.isfinite(normalized)):
                    raise ValueError("Non-finite embedding values")

                self.gallery[name] = normalized

                print(
                    f"  SUCCESS: {name} | "
                    f"Embedding: {normalized.shape}"
                )

            except (ValueError, RuntimeError, cv2.error) as exc:
                self.failures.append((path.name, str(exc)))
                print(f"  FAILED: {exc}")

        print("\n========== ENROLLMENT SUMMARY ==========")
        print(f"Total supported files: {len(files)}")
        print(f"Successfully enrolled: {len(self.gallery)}")
        print(f"Failed: {len(self.failures)}")

        if self.failures:
            print("\nFailed files:")
            for filename, reason in self.failures:
                print(f"  {filename}: {reason}")

        print("========================================")

        return self.gallery


def main():
    parser = argparse.ArgumentParser(
        description="Enroll reference faces offline"
    )
    parser.add_argument(
        "--folder",
        default="data/references",
    )
    parser.add_argument(
        "--detector",
        default="models/detector.onnx",
    )
    parser.add_argument(
        "--recognizer",
        default="models/recognizer.onnx",
    )

    args = parser.parse_args()

    enrollment = FaceEnrollment(
        detector_path=args.detector,
        recognizer_path=args.recognizer,
    )

    enrollment.enroll_folder(args.folder)


if __name__ == "__main__":
    main()
