
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
/* Global */
:root { --fr-bg:#F6F8FC; --fr-card:#FFFFFF; --fr-ink:#17233D; --fr-muted:#64748B; --fr-primary:#4F46E5; --fr-hover:#4338CA; --fr-green:#16A085; --fr-amber:#D97706; --fr-border:#E2E8F0; }
body, .gradio-container, .gradio-container .main { background:var(--fr-bg) !important; color:var(--fr-ink) !important; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif !important; }
.gradio-container { max-width:1260px !important; margin:auto !important; padding:0 22px 24px !important; }
.gradio-container .prose, .gradio-container label, .gradio-container .prose p { color:var(--fr-ink); }
/* Navigation */
#fr-nav { display:flex; align-items:center; justify-content:space-between; padding:19px 0; border-bottom:1px solid var(--fr-border); gap:16px; }
.fr-brand { display:flex; gap:11px; align-items:center; font-size:19px; font-weight:750; letter-spacing:-.5px; color:var(--fr-ink); }
.fr-logo { display:grid; place-items:center; width:36px; height:36px; border-radius:11px; background:var(--fr-primary); color:white; }
.fr-logo svg { width:21px; height:21px; }
.fr-brand small { display:block; color:var(--fr-muted); font-size:11px; font-weight:500; letter-spacing:0; }
.fr-private { display:flex; align-items:center; gap:8px; font-size:12px; font-weight:650; color:#147d68; background:#eaf8f3; border:1px solid #c9ede2; padding:8px 12px; border-radius:30px; }
.fr-dot { background:var(--fr-green); width:7px; height:7px; border-radius:50%; }
/* Hero / workflow */
#fr-intro { padding:25px 0 18px; }
#fr-intro h1 { font-size:clamp(25px,3vw,34px); font-weight:750; letter-spacing:-1.1px; line-height:1.18; margin:0 0 8px; color:var(--fr-ink); }
#fr-intro p { margin:0; color:var(--fr-muted); font-size:14px; line-height:1.6; }
.fr-steps { display:flex; gap:10px; align-items:center; padding-top:16px; color:var(--fr-muted); font-size:12px; font-weight:600; }
.fr-step { padding:6px 11px; border:1px solid var(--fr-border); border-radius:24px; background:white; }
.fr-step:first-child { color:var(--fr-primary); border-color:#c7d2fe; background:#eef2ff; }
.fr-sep { color:#94a3b8; }
/* Cards */
.fr-input-row { align-items:stretch !important; gap:22px !important; }
.fr-card { background:var(--fr-card) !important; border:1px solid var(--fr-border) !important; border-radius:17px !important; padding:21px !important; box-shadow:0 4px 22px rgba(23,35,61,.035); min-width:0; }
.fr-card-title { font-size:18px; font-weight:730; color:var(--fr-ink); margin:0 0 5px; letter-spacing:-.35px; }
.fr-card-desc { font-size:12.5px; color:var(--fr-muted); line-height:1.55; margin:0 0 14px; }
/* Uploads */
#fr-folder, #fr-photo { border:1px dashed #bac7da !important; background:#fbfcff !important; border-radius:12px !important; overflow:hidden; }
#fr-folder { max-height:265px; overflow-y:auto; }
#fr-folder .file-preview { max-height:225px; overflow-y:auto !important; scrollbar-width:thin; }
#fr-photo { min-height:230px; }
#fr-photo img { object-fit:contain !important; }
.fr-card .block { border-color:var(--fr-border); }
/* Buttons */
#fr-enroll, #fr-recognize { background:var(--fr-primary) !important; border:1px solid var(--fr-primary) !important; color:white !important; border-radius:10px !important; font-weight:700 !important; min-height:42px; box-shadow:none !important; }
#fr-enroll:hover, #fr-recognize:hover { background:var(--fr-hover) !important; }
#fr-reset { background:white !important; color:var(--fr-ink) !important; border:1px solid var(--fr-border) !important; border-radius:10px !important; font-weight:650 !important; min-height:42px; }
#fr-reset:hover { background:#f1f5f9 !important; }
.fr-actions { align-items:center !important; gap:12px !important; margin:17px 0 6px; }
#fr-recognize { max-width:220px; }
#fr-reset { max-width:110px; }
/* Enrollment */
.fr-enroll-status { border-top:1px solid var(--fr-border); padding-top:12px; margin-top:9px; }
#fr-registered input { font-size:22px !important; font-weight:750 !important; color:var(--fr-green) !important; }
#fr-enrollment-report textarea { font-size:12px !important; line-height:1.5; background:#f8fafc !important; }
/* Results */
#fr-results { margin-top:25px; }
.fr-results-head h2 { font-size:23px; letter-spacing:-.6px; margin:0 0 5px; color:var(--fr-ink); }
.fr-results-head p { font-size:13px; color:var(--fr-muted); margin:0 0 15px; }
.fr-metrics { gap:14px !important; }
.fr-metric { border:1px solid var(--fr-border) !important; border-radius:13px !important; background:white !important; padding:13px 16px !important; }
.fr-metric .label-wrap { font-size:12px !important; color:var(--fr-muted) !important; }
.fr-metric input { font-size:26px !important; font-weight:750 !important; color:var(--fr-ink) !important; background:transparent !important; border:0 !important; padding:0 !important; }
#fr-known input { color:var(--fr-green) !important; }
#fr-unknown input { color:var(--fr-amber) !important; }
.fr-results-card { background:white !important; border:1px solid var(--fr-border) !important; border-radius:15px !important; padding:17px !important; }
.fr-results-card .prose h3 { font-size:16px; color:var(--fr-ink); margin:0 0 8px; }
#fr-annotated img { object-fit:contain !important; }
#fr-table { overflow-x:auto; }
#fr-table table { font-size:12px !important; }
#fr-table th { background:#f8fafc !important; color:var(--fr-ink) !important; font-weight:700 !important; }
#fr-table td { color:var(--fr-ink) !important; white-space:nowrap; }
#fr-summary textarea { font-size:12px !important; color:var(--fr-ink) !important; background:#f8fafc !important; }
.fr-exports { gap:12px !important; }
.fr-export-file { min-height:0 !important; }
.fr-export-file .file-preview { max-height:95px; overflow:auto; }
/* Footer */
#fr-footer { border-top:1px solid var(--fr-border); margin-top:23px; padding:15px 0 0; font-size:11.5px; line-height:1.7; color:var(--fr-muted); }
/* Responsive */
@media(max-width:800px) { .gradio-container { padding:0 13px 20px !important; } .fr-input-row { flex-direction:column !important; gap:14px !important; } .fr-card { padding:16px !important; } .fr-metrics { flex-wrap:wrap !important; } #fr-recognize { max-width:none; } #fr-nav { padding:14px 0; } .fr-private { font-size:11px; } }

/* Light-theme contrast and scroll fixes */
#fr-table .label-wrap, #fr-table .label-wrap *, #fr-table label, #fr-table label * { color:#334155 !important; opacity:1 !important; }
.gradio-container {
    --body-text-color: #17233D !important;
    --body-text-color-subdued: #475569 !important;
    --block-label-text-color: #334155 !important;
    --block-info-text-color: #475569 !important;
    --input-background-fill: #F8FAFC !important;
    --input-placeholder-color: #64748B !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F8FAFC !important;
}
#fr-folder { height:250px !important; max-height:250px !important;
    min-height:250px !important; overflow-y:auto !important;
    overflow-x:hidden !important; scrollbar-gutter:stable; }
#fr-folder .file-preview, #fr-folder .file-list,
#fr-folder .file-preview-list, #fr-folder .files {
    max-height:205px !important; overflow-y:auto !important;
    min-height:0 !important; scrollbar-gutter:stable;
}
/* Keep upload prompts light, but uploaded file rows dark. */
#fr-folder, #fr-photo { color:#334155 !important; }
#fr-folder :is(.upload-container,.upload-text), #fr-photo :is(.upload-container,.upload-text) { color:#334155 !important; background:#FBFCFF !important; }
#fr-folder :is(.upload-container,.upload-text) :is(p,span), #fr-photo :is(.upload-container,.upload-text) :is(p,span) { color:#334155 !important; }
#fr-folder :is(table,.file-preview,.file-list,.file-preview-list,.files) { background:#182536 !important; color:#F8FAFC !important; }
#fr-folder :is(table,.file-preview,.file-list,.file-preview-list,.files) :is(td,th,span,p,a,button,svg,div) { color:#F8FAFC !important; }
#fr-folder tbody tr:nth-child(odd) { background:#0F172A !important; }
#fr-folder tbody tr:nth-child(even) { background:#1E293B !important; }
#fr-folder tbody td { background:transparent !important; color:#F8FAFC !important; }
#fr-enrollment-report textarea, #fr-summary textarea {
    color:#17233D !important; -webkit-text-fill-color:#17233D !important;
    background:#F8FAFC !important; opacity:1 !important;
    border-color:#CBD5E1 !important; font-size:13px !important;
}
#fr-enrollment-report label, #fr-summary label, #fr-table label,
#fr-enrollment-report .label-wrap, #fr-summary .label-wrap,
#fr-table .label-wrap, .fr-card label, .fr-results-card label {
    color:#334155 !important; opacity:1 !important;
}
#fr-enroll, #fr-recognize, #fr-photo button.upload-button,
#fr-folder button.upload-button {
    background:#4F46E5 !important; color:#FFFFFF !important;
    border:1px solid #4F46E5 !important; border-radius:10px !important;
    min-height:44px !important; font-size:14px !important;
    font-weight:700 !important; padding:10px 18px !important;
}
#fr-enroll:hover, #fr-recognize:hover,
#fr-photo button.upload-button:hover, #fr-folder button.upload-button:hover {
    background:#4338CA !important;
}
#fr-folder button.upload-button *, #fr-photo button.upload-button * { color:#FFFFFF !important; }
#fr-folder, #fr-folder * {
    scrollbar-width:thin; scrollbar-color:#94A3B8 #E2E8F0;
}
#fr-folder ::-webkit-scrollbar, #fr-folder::-webkit-scrollbar { width:9px; height:9px; }
#fr-folder ::-webkit-scrollbar-thumb, #fr-folder::-webkit-scrollbar-thumb { background:#94A3B8; border-radius:8px; }
/* Explicit contrast pairs for results and dark component labels. */
#fr-table tbody tr:nth-child(odd) { background:#0F172A !important; }
#fr-table tbody tr:nth-child(even) { background:#1E293B !important; }
#fr-table tbody td, #fr-table tbody td * { background:transparent !important; color:#F8FAFC !important; -webkit-text-fill-color:#F8FAFC !important; }
#fr-table thead th, #fr-table thead th * { background:#F8FAFC !important; color:#17233D !important; -webkit-text-fill-color:#17233D !important; }
#fr-table .label-wrap, #fr-table .label-wrap * { background:#FFFFFF !important; color:#334155 !important; }
#fr-detected, #fr-known, #fr-unknown, #fr-registered, #fr-summary, #fr-enrollment-report { background:#1E293B !important; color:#F8FAFC !important; }
#fr-detected input, #fr-known input, #fr-unknown input, #fr-registered input { background:transparent !important; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; opacity:1 !important; }
.fr-metric .label-wrap, .fr-metric .label-wrap *, #fr-registered label, #fr-registered label *, #fr-summary label, #fr-summary label *, #fr-enrollment-report label, #fr-enrollment-report label * { color:#F8FAFC !important; }
#fr-photo .label-wrap, #fr-photo .label-wrap *, #fr-annotated .label-wrap, #fr-annotated .label-wrap * { background:#1E293B !important; color:#F8FAFC !important; }
#fr-photo :is(.upload-container,.upload-text) svg { color:#475569 !important; }

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


# UI-only callbacks. The enrollment and recognition algorithms above are unchanged.
def show_enrollment(gallery, report, count):
    return gr.update(visible=True)


def show_results(*_values):
    return gr.update(visible=True)


def clear_result_display():
    """Invalidate old output when either input changes."""
    return (gr.update(visible=False), None, [], 0, 0, 0,
            "Ready to process.", None, None)


def clear_gallery_display():
    """Changing reference files requires fresh enrollment."""
    return ({}, gr.update(visible=False), "No reference folder enrolled.", 0)


with gr.Blocks(
    title="FaceRoster | Offline Recognition",
    css=CSS,
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    analytics_enabled=False,
) as demo:
    gallery_state = gr.State({})

    gr.HTML('''<header id="fr-nav"><div class="fr-brand"><span class="fr-logo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="10" r="3"/><path d="M6.5 19c.5-3 2.4-4.5 5.5-4.5s5 1.5 5.5 4.5"/></svg></span><span>FaceRoster<small>Intelligent Photo Recognition</small></span></div><span class="fr-private"><span class="fr-dot"></span> Offline &amp; Private</span></header>''')
    gr.HTML('''<section id="fr-intro"><h1>Identify everyone in the frame.</h1><p>Upload reference photos and analyze group pictures with private, on-device face recognition.</p><div class="fr-steps"><span class="fr-step">01 Enroll</span><span class="fr-sep">→</span><span class="fr-step">02 Upload</span><span class="fr-sep">→</span><span class="fr-step">03 Recognize</span></div></section>''')

    with gr.Row(elem_classes=["fr-input-row"]):
        with gr.Column(scale=1, elem_classes=["fr-card"]):
            gr.HTML('<div class="fr-card-title">01 · Reference Enrollment</div><p class="fr-card-desc">Select a folder of individual photographs. Each filename identifies a person.</p>')
            reference_files = gr.File(label="Select reference folder", file_count="directory",
                                      file_types=None, type="filepath", elem_id="fr-folder", height=250)
            enroll_button = gr.Button("Enroll Reference Faces", variant="primary", elem_id="fr-enroll")
            with gr.Column(visible=False, elem_classes=["fr-enroll-status"]) as enrollment_panel:
                registered_count = gr.Number(label="Registered identities", value=0,
                                             interactive=False, precision=0, elem_id="fr-registered")
                enrollment_status = gr.Textbox(label="Enrollment report", value="No reference folder enrolled.",
                                               lines=5, interactive=False, elem_id="fr-enrollment-report")

        with gr.Column(scale=1, elem_classes=["fr-card"]):
            gr.HTML('<div class="fr-card-title">02 · Group Photograph</div><p class="fr-card-desc">Upload a group photograph to identify registered and unfamiliar people.</p>')
            group_photo = gr.Image(label="Upload group photo", type="filepath", sources=["upload"],
                                   height=250, elem_id="fr-photo")
            with gr.Accordion("Advanced Recognition Settings", open=False):
                threshold = gr.Slider(minimum=0.20, maximum=0.80, value=0.45,
                                      step=0.01, label="Cosine similarity threshold (experimental)")
                detection_size = gr.Dropdown(choices=[320, 640, 960, 1280], value=640,
                                             label="Face detection resolution")

    with gr.Row(elem_classes=["fr-actions"]):
        recognize_button = gr.Button("Recognize Faces", variant="primary", elem_id="fr-recognize", scale=0)
        reset_button = gr.Button("Reset", variant="secondary", elem_id="fr-reset", scale=0)

    with gr.Column(visible=False, elem_id="fr-results") as results_panel:
        gr.HTML('<div class="fr-results-head"><h2>Recognition Results</h2><p>Review detected faces, identified people, and exported reports.</p></div>')
        with gr.Row(elem_classes=["fr-metrics"]):
            with gr.Column(elem_classes=["fr-metric"]):
                detected_count = gr.Number(label="Faces Detected", value=0, precision=0,
                                           interactive=False, elem_id="fr-detected")
            with gr.Column(elem_classes=["fr-metric"]):
                known_count = gr.Number(label="Known Individuals", value=0, precision=0,
                                        interactive=False, elem_id="fr-known")
            with gr.Column(elem_classes=["fr-metric"]):
                unknown_count = gr.Number(label="Unknown Faces", value=0, precision=0,
                                          interactive=False, elem_id="fr-unknown")
        result_status = gr.Textbox(label="Processing summary", value="Ready to process.",
                                   interactive=False, elem_id="fr-summary")
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["fr-results-card"]):
                gr.Markdown("### Analyzed Photograph")
                annotated_image = gr.Image(label="Annotated group photograph", type="filepath",
                                           interactive=False, height=430, elem_id="fr-annotated")
            with gr.Column(scale=2, elem_classes=["fr-results-card"]):
                gr.Markdown("### Identification Details")
                results_table = gr.Dataframe(headers=COLUMNS,
                    datatype=["number", "str", "str", "number", "number"],
                    label="Recognition details", interactive=False, elem_id="fr-table")
        with gr.Column(elem_classes=["fr-results-card"]):
            gr.Markdown("### Export Results")
            with gr.Row(elem_classes=["fr-exports"]):
                download_image = gr.File(label="Annotated image download", interactive=False,
                                         elem_classes=["fr-export-file"])
                download_csv = gr.File(label="CSV results download", interactive=False,
                                       elem_classes=["fr-export-file"])

    gr.HTML('<footer id="fr-footer">Face recognition runs locally with ONNX models. Reference embeddings are held in session memory; generated output files remain on this computer. Cosine similarity is not a calibrated identity probability. The threshold remains experimental.</footer>')

    enroll_event = enroll_button.click(
        fn=enroll_references, inputs=[reference_files],
        outputs=[gallery_state, enrollment_status, registered_count], show_progress="full")
    enroll_event.success(fn=show_enrollment,
        inputs=[gallery_state, enrollment_status, registered_count], outputs=[enrollment_panel])

    recognize_event = recognize_button.click(
        fn=recognize_photo,
        inputs=[group_photo, gallery_state, threshold, detection_size],
        outputs=[annotated_image, results_table, detected_count, known_count,
                 unknown_count, result_status, download_image, download_csv],
        show_progress="full")
    recognize_event.success(fn=show_results, inputs=[], outputs=[results_panel])

    reset_event = reset_button.click(
        fn=reset_all, inputs=[],
        outputs=[reference_files, group_photo, gallery_state, enrollment_status,
                 registered_count, annotated_image, results_table, detected_count,
                 known_count, unknown_count, result_status, download_image, download_csv])
    reset_event.then(fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
                     inputs=[], outputs=[enrollment_panel, results_panel])

    # UI-only invalidation: never show recognition outputs for outdated inputs.
    for component in (group_photo, threshold, detection_size):
        component.change(fn=clear_result_display, inputs=[],
                         outputs=[results_panel, annotated_image, results_table,
                                  detected_count, known_count, unknown_count,
                                  result_status, download_image, download_csv])
    reference_files.change(fn=clear_gallery_display, inputs=[],
                           outputs=[gallery_state, enrollment_panel, enrollment_status, registered_count])
    reference_files.change(fn=clear_result_display, inputs=[],
                           outputs=[results_panel, annotated_image, results_table,
                                    detected_count, known_count, unknown_count,
                                    result_status, download_image, download_csv])


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
