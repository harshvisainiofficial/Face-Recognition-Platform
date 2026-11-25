# FaceRoster — Offline Group Photo Recognition

**Take-home Task A: Who Is in the Photo?**  
A locally running Python application that enrolls people from reference photographs and labels faces in a group photograph as a registered name or **Unknown**.

> FaceRoster combines local face detection, recognition and an interactive Gradio interface. It is designed for offline operation after dependencies and model weights have been installed.

## Features

- Select a folder of reference photographs through a local browser UI; identity names are derived from filenames.
- Detect reference faces with **SCRFD**, align using facial landmarks, and extract **512-dimensional ArcFace embeddings**.
- Store enrolled embeddings in the current Gradio session; changing the reference set replaces the gallery.
- Detect and match faces in a group photograph using cosine similarity and a configurable rejection threshold.
- Mark unmatched faces as **Unknown**.
- Show an annotated photograph, detected/known/unknown counts, and a detailed results table.
- Use adaptive reference detection for difficult photographs, with recovery notes for review.
- Browse uploaded reference filenames in a scrollable list; enrollment and recognition panels appear after processing.
- Export the annotated image and a CSV with assignments, similarity scores, and bounding boxes.
- Use local ONNX models with ONNX Runtime's CPU execution provider; no hosted inference API is required.

**Important:** Cosine similarity is **not** a calibrated identity probability. The default threshold (`0.45`) is experimental and has not been validated for arbitrary identities.

## 1. Requirements

### Tested development workstation

| Component | Development configuration |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS |
| CPU | Intel Xeon Silver 4114, 20 logical CPUs |
| GPU | NVIDIA Quadro P1000 (not required for current CPU pipeline) |
| RAM | 32 GB |
| Python | **3.11**, in Conda environment `faceroaster` |
| Runtime | ONNX Runtime CPU |
| UI | Gradio 5.x (verify exact installed version) |

A CUDA-capable GPU is **not** required. Other machines may work, but have not yet been tested.

You will need Git, Conda/Miniconda or a Python 3.11 virtual environment, internet **during initial installation**, and local copies of both ONNX model files.

## 2. Project structure

The relevant project layout is:

```text
FaceRoster/
├── README.md
├── requirements-lock.txt       # Generate from the working environment
├── models/
│   ├── detector.onnx           # Local SCRFD detector (not normally committed)
│   └── recognizer.onnx         # Local ArcFace model (not normally committed)
├── data/
│   ├── references/             # Practice references; keep private
│   └── practice/               # Practice group photos; keep private
├── outputs/                    # Generated photos and CSVs; keep private
├── docs/
├── src/
│   ├── __init__.py
│   ├── detect_faces.py
│   ├── enrollment.py
│   ├── recognition.py
│   ├── reference_detection.py  # Adaptive enrollment, if integrated
│   ├── app.py                 # Original UI, if retained
│   └── app_backup.py          # Latest UI supplied for documentation
└── tests/
```

The exact contents of the repository may vary. The latest supplied UI, `src/app_backup.py`, imports and calls `detect_reference_face`.

## 3. Installation — Linux (Conda recommended)

Open a Linux terminal. The following commands assume you cloned the project into `~/Harshvi/FaceRoster`, as on the development workstation. **Change the path if your clone is elsewhere.**

### Step 1 — Get the repository

If you already have the project, skip cloning and enter its directory:

```bash
cd ~/Harshvi/FaceRoster
```

For a fresh machine, clone the **actual public repository URL** supplied by the author (do not use a placeholder URL):

```bash
git clone <YOUR_ACTUAL_GITHUB_REPOSITORY_URL> FaceRoster
cd FaceRoster
```

The repository URL must be replaced with a real URL before executing that command.

### Step 2 — Create and activate a clean environment

```bash
conda create -n faceroaster python=3.11 -y
conda activate faceroaster
python --version
```

Expected: `Python 3.11.x` and a terminal prompt beginning with `(faceroaster)`.

If Conda is unavailable, install Python 3.11 and its `venv` package using your Linux distribution's supported method, then run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

Do not replace the system Python installation.

### Step 3 — Install dependencies

**Preferred, reproducible installation:** If the repository contains a lock file generated from the verified development environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
```

**Fallback for an unfinalized repository:** Install the baseline ML packages and Gradio 5. These are the versions proposed during development; compatibility of a fresh installation must be checked:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4" "onnxruntime==1.20.1" \
  "opencv-python-headless==4.10.0.84" "Pillow==11.1.0" \
  "insightface==0.7.3" "Cython==3.0.12"
python -m pip install "gradio>=5.0,<6.0"
```

InsightFace may compile native components; if installation fails, read the build error and install missing system development tools rather than changing Python or dependencies at random.

Check imports:

```bash
python - <<'PY'
import cv2, numpy, onnxruntime, insightface, gradio
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('ONNX Runtime:', onnxruntime.__version__)
print('InsightFace:', insightface.__version__)
print('Gradio:', gradio.__version__)
print('Providers:', onnxruntime.get_available_providers())
PY
```

The providers list must contain `CPUExecutionProvider`.

On the development machine, after confirming the complete UI works, generate the exact dependency lock:

```bash
python -m pip freeze > requirements-lock.txt
```

Commit that file only after checking a clean-environment installation.

## 4. Install model files locally

The application expects **two existing local files**:

```text
models/detector.onnx
models/recognizer.onnx
```

The development setup used the InsightFace **Buffalo-L** model pack, with these source filenames:

| Model pack filename | Local filename | Function |
| --- | --- | --- |
| `det_10g.onnx` | `detector.onnx` | SCRFD face detection and landmarks |
| `w600k_r50.onnx` | `recognizer.onnx` | ArcFace recognition embeddings |

1. During **setup while online**, consult the [official InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).
2. Review the **model-weights license** before downloading or redistributing them. InsightFace code and pretrained weights have different terms; Buffalo weights are generally offered for non-commercial research use. Do not assume commercial reuse is permitted for an internship/company deployment.
3. Download and extract the authorized model pack.
4. Copy only the two required files into `models/`:

```bash
mkdir -p models
cp ~/Downloads/buffalo_l/det_10g.onnx models/detector.onnx
cp ~/Downloads/buffalo_l/w600k_r50.onnx models/recognizer.onnx
ls -lh models/
```

Adjust the extracted folder path if necessary. On the development machine, the detector was approximately **17 MB** and the recognizer approximately **167 MB**. File sizes alone do not authenticate a model; for a reproducible release, record checksums of the authorized files.

**Do not rely on InsightFace's automatic model-download behavior.** The application loads explicit local ONNX paths and should fail clearly if they are missing. Model weights should not be committed to a public Git repository without reviewing licensing and repository-size limits.

Verify the recognizer:

```bash
python - <<'PY'
import onnxruntime as ort
session = ort.InferenceSession('models/recognizer.onnx', providers=['CPUExecutionProvider'])
print('Inputs:', [(x.name, x.shape) for x in session.get_inputs()])
print('Outputs:', [(x.name, x.shape) for x in session.get_outputs()])
PY
```

Observed development output: input `['None', 3, 112, 112]`; output `[1, 512]` (tensor names may differ).

## 5. Start the application

From the **project root**, with the environment activated:

```bash
conda activate faceroaster
cd ~/Harshvi/FaceRoster
python -m src.app_backup
```

If you rename the final UI file to `src/app.py`, use `python -m src.app` instead.

Open **http://127.0.0.1:7860** in a browser on the same machine.

The app binds to `127.0.0.1`, sets `share=False`, and is not intended for public hosting. If port 7860 is occupied, stop the other local service before restarting. Press **Ctrl+C** in the terminal to stop FaceRoster.

The application uses local inference. Gradio and dependency network/telemetry behavior still requires a complete offline audit before an unqualified claim of zero network attempts.

## 6. Use FaceRoster

### A. Enroll a reference folder

1. Prepare a directory of reference photographs; ideally one clearly visible face per image.
2. Name each file after the person, e.g. `Priya Sharma.jpg` or `Aarav Mehta.png`.
3. In the browser, click **Select reference folder** and choose the directory. Do not type a folder path into the terminal.
4. Click **Enroll Reference Faces**.
5. Read **Registered identities** and the **Enrollment report**. Fix or review any skipped or ambiguous files.

Supported formats in the enrollment implementation: `.jpg`, `.jpeg`, `.png`, `.webp`. Filename stems are used as labels. The browser's directory upload must preserve filenames correctly; verify this on the target machine. A new folder can contain entirely different people; no identity list is hardcoded.

**Multiple faces:** The original enrollment logic rejects such photos. The later adaptive UI code may select a clearly dominant foreground face, but this is heuristic and can select the wrong person. Review any adaptive selection; do not treat it as automatically verified.

### B. Recognize a group photograph

1. Upload a group photograph under **Group Photograph**.
2. Leave the detection resolution at `640` for the initial run.
3. Leave the cosine similarity threshold at `0.45` **for diagnostic testing only**.
4. Click **Recognize Faces**.
5. Review the annotated photo, results table, and detected/known/unknown counts.
6. Download the annotated photograph or CSV if required.

A face is labeled with its nearest registered identity only if its cosine similarity meets the configured threshold; otherwise it is labeled **Unknown**. This is open-set recognition with a provisional threshold, not a probability-based confidence system.

### C. Replace references or reset

Select a different folder and enroll again to replace the current gallery. Use **Reset** to clear the current UI session and its in-memory gallery. Reset does **not necessarily delete** Gradio upload caches or previously generated output files from disk.

## 7. Command-line operation (optional)

The UI is the intended user workflow; these commands help debug the underlying models.

### Face detection only

```bash
python -m src.detect_faces \
  --image data/practice/group3.jpg \
  --model models/detector.onnx \
  --output outputs/detected.jpg \
  --size 640
```

### Reference enrollment only

```bash
python -m src.enrollment --folder data/references
```

### Group recognition

```bash
python -m src.recognition \
  --references data/references \
  --image data/practice/recognition_test.jpg \
  --output outputs/recognized.jpg \
  --csv outputs/recognition_results.csv \
  --threshold 0.45 \
  --size 640
```

Open the output on a Linux desktop:

```bash
xdg-open outputs/recognized.jpg
```

**Note:** The standalone `src.enrollment` and `src.recognition` commands use the original enrollment behavior unless they have also been explicitly updated to use `src.reference_detection`. Adaptive enrollment changes made to `src.app` do not automatically change those CLI modules.

## 8. Offline operation and verification

**Internet is needed only for initial installation and authorized model acquisition.** To test the finished runtime:

1. Install all dependencies and place both ONNX files locally while online.
2. Stop the app; disconnect Wi-Fi, Ethernet, VPN, and other network interfaces as applicable.
3. Restart the terminal and activate the environment.
4. Start the app with `python -m src.app_backup`.
5. Select a **new** reference folder through the UI and enroll its identities.
6. Upload a group photo, recognize faces, and verify labels and result counts.
7. Download the annotated image and CSV.
8. Repeat after restarting the application.
9. Check logs for update checks, telemetry, CDN assets, model downloads, or other network attempts.

The environment variable `NO_ALBUMENTATIONS_UPDATE=1` suppresses a known Albumentations version-check request. The UI sets `GRADIO_ANALYTICS_ENABLED=False` and `analytics_enabled=False`. These settings are useful but do **not**, by themselves, prove that every dependency makes zero network requests.

**Observed:** The standalone face detector was successfully run offline on three practice group photos. **Not yet verified in the supplied evidence:** a complete offline Gradio UI test with a new reference folder, plus a network-access audit. Record the actual result before claiming full verification.

## 9. Evaluation and observed results

The following are **development observations**, not an official benchmark:

- SCRFD detection was tested on three group photos online and offline; one run detected **13 faces** in **0.382 seconds**, including model loading.
- Initial reference enrollment: **5 of 5** practice identities enrolled, each with a 512-dimensional embedding.
- Expanded UI reference enrollment: **13 of 17** photos enrolled; three reported no face, and one reported two faces. Adaptive reference detection is integrated in the latest supplied UI; its final recovery rate is not recorded.
- Nine command-line group-photo runs completed and two additional runs failed to decode their images. The successful runs reported a total of **41 detected faces**. Their name assignments are not a ground-truth accuracy measure.
- One group-photo run reported two known assignments and one Unknown, with similarities **0.6680**, **0.6534**, and **0.2309**, respectively.
- Some known faces wearing dark sunglasses received similarity scores below the provisional threshold. This is an observed failure mode, not proof of a universal sunglasses effect.
- Per-run processing times in the shared logs were roughly **3.0–4.1 seconds**, **including model loading and re-enrollment**; they are not steady-state UI latency benchmarks.

### Evaluation still needed

Collect labeled development and held-out test photographs; measure face-detection recall, correct known identification, unknown rejection, false acceptance, false rejection, and processing latency. Calibrate thresholds on development data only, then freeze the setting before held-out evaluation. Do not invent results or infer accuracy from assignment counts.

## 10. Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `faceroaster`, check `python --version`, install the lock file in the active environment. |
| `Required local model not found` | Check `models/detector.onnx` and `models/recognizer.onnx` from the project root. |
| `Cannot read group image` | Confirm the path exists; inspect with `file path/to/image.jpg`. Renaming an extension does not convert the image. Re-encode with a suitable image tool if supported. |
| No face detected in a close-up | Try adaptive reference detection and padding; verify the resulting face selection. |
| Multiple faces in a reference photo | Review foreground/background faces; do not silently assume the largest is the intended identity. |
| Known face labeled Unknown | Inspect cosine similarity and image quality; do not lower the threshold without measuring unknown false acceptance. |
| Unknown face given a known name | Threshold may be too permissive; evaluate known and unknown examples and consider margin/ambiguity checks. |
| `warning: det_size is already set` | InsightFace may ignore subsequent `prepare(input_size=...)` calls. Confirm actual detector configuration before claiming a resolution sweep worked. |
| `FutureWarning` from `face_align.py` | Deprecation warning in a dependency; not necessarily a runtime failure. Avoid untested upgrades before submission. |
| Albumentations update-check warning | Ensure `NO_ALBUMENTATIONS_UPDATE=1` is set **before** importing InsightFace. |
| Browser cannot connect | Confirm `python -m src.app_backup` is running and visit `http://127.0.0.1:7860` on the same machine. |
| Folder names lost after upload | Inspect Gradio's uploaded file paths; filename preservation is essential for identity labels. |

## 11. Known limitations

- One reference image per identity is sensitive to pose, blur, lighting, occlusion, sunglasses, and poor crops.
- Detection can miss tiny or heavily cropped faces; adaptive recovery is heuristic.
- The default similarity threshold is **not calibrated** for 50 unfamiliar identities.
- Independent nearest-neighbor matching can assign the same identity to multiple faces in one group photograph.
- Foreground-face selection from images containing multiple people may be incorrect; manual review is recommended.
- The UI's gallery is session-local; output files and upload caches may remain on disk.
- The implementation currently uses CPU inference; GPU performance has not been evaluated.
- The actual license terms of the model weights must be checked before distribution or commercial use.
- Full UI offline verification, privacy audit, and clean-clone reproducibility remain to be completed.

## 12. Repository hygiene and privacy

Do not commit private reference images, group photos, generated output files, caches, credentials, or model weights without explicit authorization. A suitable `.gitignore` includes:

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.gradio/
.env
models/*
!models/.gitkeep
data/*
!data/.gitkeep
outputs/*
!outputs/.gitkeep
*.log
```

Face embeddings are biometric data. Use consented or appropriately licensed images, minimize retention, and delete generated files when no longer needed.

## 13. Architecture and research

The local processing path is:

```text
Reference folder ──> SCRFD detection ──> 5-point face alignment
                 └─> ArcFace embedding ──> L2 normalization ──> In-memory gallery

Group photo ──────> SCRFD detection ──> Alignment ──> ArcFace embedding
                 └─> Cosine similarity against gallery
                 └─> Threshold-based known/Unknown assignment
                 └─> Annotated image + results table + CSV
```

Companion documentation, when included in the repository: `RESEARCH.md` (model and framework research), `ARCHITECTURE.md` (system design), and `AI_COLLAB.md` (engineering decisions and AI collaboration).

## 14. References

- [InsightFace repository](https://github.com/deepinsight/insightface)
- [InsightFace model zoo and model terms](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [Gradio File documentation](https://www.gradio.app/docs/gradio/file)
- [Gradio Blocks documentation](https://www.gradio.app/docs/gradio/blocks)
- [Archify repository](https://github.com/tt-a1i/archify)

---

**FaceRoster is a local experimental recognition application.** Review identity assignments carefully; do not use it as a sole basis for consequential identity or access decisions.
