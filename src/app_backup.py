
"""FaceRoster: local Gradio interface for offline face recognition."""

import os

# Disable dependency update checks before importing ML libraries.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import csv
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.enrollment import FaceEnrollment, read_image
from src.reference_detection import detect_reference_face
from src.recognition import get_embedding, match_identity, draw_label


ROOT = Path(__file__).resolve().parent.parent
DETECTOR = ROOT / "models" / "detector.onnx"
RECOGNIZER = ROOT / "models" / "recognizer.onnx"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "Face ID",
    "Assigned Name",
    "Best Candidate",
    "Cosine Similarity",
    "Detection Score",
]

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}
#hero {
    padding: 26px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #253b66);
    color: white;
}
#hero h1, #hero p {
    color: white !important;
}
#hero h1 {
    font-size: 36px !important;
}
#recognize-button {
    font-weight: 700 !important;
    font-size: 17px !important;
}
"""

# Loaded once and reused across UI operations.
engine = None


def get_engine():
    """Load local ONNX models only when first needed."""
    global engine

    if engine is None:
        engine = FaceEnrollment(
            detector_path=str(DETECTOR),
            recognizer_path=str(RECOGNIZER),
        )

    return engine


def extract_uploaded_paths(files):
    """Normalize Gradio's uploaded file objects or path strings."""
    if not files:
        return []

    if not isinstance(files, list):
        files = [files]

    paths = []

    for item in files:
        path = Path(item if isinstance(item, str) else item.name)

        if path.is_file():
            paths.append(path)

    return paths



def enroll_references(files):
    """Enroll uploaded reference faces using adaptive detection."""
    paths = extract_uploaded_paths(files)

    if not paths:
        return {}, "No reference photographs selected.", 0

    face_engine = get_engine()

    gallery = {}
    failures = []
    recovery_notes = []
    supported = {".jpg", ".jpeg", ".png", ".webp"}

    for path in sorted(paths, key=lambda p: p.name.casefold()):
        if path.suffix.lower() not in supported:
            failures.append(f"{path.name}: Unsupported extension")
            continue

        name = path.stem.strip()

        if not name:
            failures.append(f"{path.name}: Empty identity name")
            continue

        if any(name.casefold() == n.casefold() for n in gallery):
            failures.append(f"{path.name}: Duplicate identity name")
            continue

        image = read_image(path)

        if image is None:
            failures.append(f"{path.name}: Unreadable image")
            continue

        try:
            selected = detect_reference_face(
                face_engine.detector,
                image,
            )

            embedding = get_embedding(
                face_engine.recognizer,
                selected.image,
                selected.landmarks,
            )

            gallery[name] = embedding

            if selected.method != "original, resolution=640":
                recovery_notes.append(
                    f"{path.name}: {selected.method}"
                )

        except (ValueError, RuntimeError, cv2.error) as exc:
            failures.append(f"{path.name}: {exc}")

    lines = [
        f"Enrollment complete: {len(gallery)} identities registered.",
        f"Files received: {len(paths)}",
        f"Failed or skipped: {len(failures)}",
        f"Adaptive selections: {len(recovery_notes)}",
    ]

    if recovery_notes:
        lines.append("")
        lines.append("Adaptive selections (review recommended):")
        lines.extend(f"- {item}" for item in recovery_notes)

    if failures:
        lines.append("")
        lines.append("Issues:")
        lines.extend(f"- {item}" for item in failures)

    return gallery, "\n".join(lines), len(gallery)

def recognize_photo(photo_path, gallery, threshold, detection_size):
    """Recognize faces and return UI outputs."""
    if not gallery:
        raise gr.Error(
            "Please select and enroll a reference folder first."
        )

    if not photo_path:
        raise gr.Error("Please upload a group photograph.")

    image = read_image(Path(photo_path))

    if image is None:
        raise gr.Error("Cannot decode the uploaded photograph.")

    face_engine = get_engine()

    face_engine.detector.prepare(
        ctx_id=-1,
        input_size=(int(detection_size), int(detection_size)),
        det_thresh=0.5,
    )

    boxes, landmarks = face_engine.detector.detect(
        image,
        max_num=0,
        metric="default",
    )

    if len(boxes) and (
        landmarks is None or len(landmarks) != len(boxes)
    ):
        raise gr.Error("Face detection returned invalid landmarks.")

    annotated = image.copy()
    table_rows = []
    csv_rows = []
    known_count = 0
    unknown_count = 0

    # Sort faces approximately from top to bottom, then left to right.
    indices = sorted(
        range(len(boxes)),
        key=lambda i: (
            int(boxes[i][1]) // 80,
            float(boxes[i][0]),
        ),
    )

    for face_id, index in enumerate(indices, start=1):
        box = boxes[index]

        try:
            embedding = get_embedding(
                face_engine.recognizer,
                image,
                landmarks[index],
            )

            assigned, candidate, similarity = match_identity(
                embedding,
                gallery,
                float(threshold),
            )

        except (ValueError, RuntimeError, cv2.error):
            # This face is detected but could not be reliably embedded.
            assigned = "Unknown"
            candidate = "Unavailable"
            similarity = float("nan")

        if assigned == "Unknown":
            unknown_count += 1
        else:
            known_count += 1

        draw_label(annotated, box, assigned, face_id)

        score_display = (
            round(similarity, 4)
            if np.isfinite(similarity)
            else None
        )

        detection_score = round(float(box[4]), 4)

        table_rows.append([
            face_id,
            assigned,
            candidate,
            score_display,
            detection_score,
        ])

        csv_rows.append({
            "face_id": face_id,
            "name": assigned,
            "best_candidate": candidate,
            "cosine_similarity": score_display,
            "detection_score": detection_score,
            "x1": int(box[0]),
            "y1": int(box[1]),
            "x2": int(box[2]),
            "y2": int(box[3]),
        })

    # Unique output paths avoid overwriting another user's results.
    run_id = uuid.uuid4().hex[:12]
    image_output = OUTPUTS / f"recognized_{run_id}.jpg"
    csv_output = OUTPUTS / f"results_{run_id}.csv"

    if not cv2.imwrite(str(image_output), annotated):
        raise gr.Error("Failed to save the annotated photograph.")

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

    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = (
        f"Detected {len(boxes)} faces. "
        f"Known assignments: {known_count}. "
        f"Unknown: {unknown_count}. "
        f"Threshold: {float(threshold):.2f} (experimental)."
    )

    return (
        str(image_output),
        table_rows,
        len(boxes),
        known_count,
        unknown_count,
        summary,
        str(image_output),
        str(csv_output),
    )


def reset_all():
    """Clear the current session's reference gallery and UI."""
    return (
        None,       # Folder upload
        None,       # Photo upload
        {},         # Session gallery
        "No reference folder enrolled.",
        0,          # Registered count
        None,       # Annotated photo
        [],         # Results
        0,          # Detected count
        0,          # Known count
        0,          # Unknown count
        "Ready for a new session.",
        None,       # Image download
        None,       # CSV download
    )


with gr.Blocks(
    title="FaceRoster | Offline Recognition",
    css=CSS,
    analytics_enabled=False,
) as demo:

    gallery_state = gr.State({})

    gr.HTML("""
    <section id="hero">
      <h1>FaceRoster</h1>
      <p>Offline Group Photo Recognition</p>
      <p>Private, local face detection and identity matching.</p>
    </section>
    """)

    gr.Markdown(
        "### Identify people in three simple steps\n"
        "Select a reference folder, enroll the identities, "
        "then upload a group photograph."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 01 · Reference Enrollment")

            reference_files = gr.File(
                label="Select reference folder",
                file_count="directory",
                file_types=None,
                type="filepath",
            )

            enroll_button = gr.Button(
                "Enroll Reference Faces",
                variant="secondary",
            )

            registered_count = gr.Number(
                label="Registered identities",
                value=0,
                interactive=False,
                precision=0,
            )

            enrollment_status = gr.Textbox(
                label="Enrollment report",
                value="No reference folder enrolled.",
                lines=6,
                interactive=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 02 · Group Photograph")

            group_photo = gr.Image(
                label="Upload group photo",
                type="filepath",
                sources=["upload"],
                height=300,
            )

            threshold = gr.Slider(
                minimum=0.20,
                maximum=0.80,
                value=0.45,
                step=0.01,
                label="Cosine similarity threshold (experimental)",
            )

            detection_size = gr.Dropdown(
                choices=[320, 640, 960, 1280],
                value=640,
                label="Face detection resolution",
            )

    with gr.Row():
        recognize_button = gr.Button(
            "Recognize Faces",
            variant="primary",
            elem_id="recognize-button",
            scale=3,
        )

        reset_button = gr.Button(
            "Reset",
            variant="secondary",
            scale=1,
        )

    gr.Markdown("### 03 · Recognition Results")

    with gr.Row():
        detected_count = gr.Number(
            label="Faces detected",
            value=0,
            precision=0,
            interactive=False,
        )

        known_count = gr.Number(
            label="Known assignments",
            value=0,
            precision=0,
            interactive=False,
        )

        unknown_count = gr.Number(
            label="Unknown faces",
            value=0,
            precision=0,
            interactive=False,
        )

    result_status = gr.Textbox(
        label="Processing summary",
        value="Ready to process.",
        interactive=False,
    )

    annotated_image = gr.Image(
        label="Annotated group photograph",
        type="filepath",
        interactive=False,
        height=460,
    )

    results_table = gr.Dataframe(
        headers=COLUMNS,
        datatype=["number", "str", "str", "number", "number"],
        label="Recognition details",
        interactive=False,
    )

    with gr.Row():
        download_image = gr.File(
            label="Download annotated photograph",
            interactive=False,
        )

        download_csv = gr.File(
            label="Download CSV results",
            interactive=False,
        )

    gr.Markdown(
        "**Privacy:** Face recognition runs locally using ONNX models. "
        "Reference embeddings are held in session memory. "
        "Generated output files remain on this computer.\n\n"
        "**Note:** Similarity scores are not identity probabilities. "
        "The threshold is experimental and has not yet been calibrated."
    )

    enroll_button.click(
        fn=enroll_references,
        inputs=[reference_files],
        outputs=[
            gallery_state,
            enrollment_status,
            registered_count,
        ],
        show_progress="full",
    )

    recognize_button.click(
        fn=recognize_photo,
        inputs=[
            group_photo,
            gallery_state,
            threshold,
            detection_size,
        ],
        outputs=[
            annotated_image,
            results_table,
            detected_count,
            known_count,
            unknown_count,
            result_status,
            download_image,
            download_csv,
        ],
        show_progress="full",
    )

    reset_button.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            reference_files,
            group_photo,
            gallery_state,
            enrollment_status,
            registered_count,
            annotated_image,
            results_table,
            detected_count,
            known_count,
            unknown_count,
            result_status,
            download_image,
            download_csv,
        ],
    )


if __name__ == "__main__":
    if not DETECTOR.is_file() or not RECOGNIZER.is_file():
        raise FileNotFoundError(
            "Required ONNX model files are missing from models/."
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        allowed_paths=[str(OUTPUTS)],
    )
