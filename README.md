# FaceRoster — Offline Group Photo Recognition

**Take-home Task A: Who Is in the Photo?**

FaceRoster is a local Python application that enrolls identities from reference photographs and identifies faces in a group photograph. Unmatched faces are labeled **Unknown**. It uses SCRFD, ArcFace, ONNX Runtime and a Gradio interface, with no hosted inference API required.

## Features

- Reference folder upload with identity names derived from filenames.
- Face detection, landmark-based alignment and 512-dimensional face embeddings.
- Session-local reference gallery and cosine similarity matching.
- Configurable rejection threshold and Unknown labels.
- Annotated group photograph, known/unknown counts and recognition details.
- CSV and annotated image downloads.
- Adaptive reference detection for difficult photographs, with recovery notes for review.

## 1. Requirements

Tested development configuration:

| Component | Configuration |
|---|---|
| OS | Ubuntu 24.04.2 LTS |
| CPU | Intel Xeon Silver 4114, 20 logical CPUs |
| RAM | 32 GB |
| GPU | NVIDIA Quadro P1000 (not required) |
| Python | 3.11 |
| Runtime | ONNX Runtime CPU |
| Interface | Gradio |

Internet access is required for initial dependency installation and authorized model acquisition. A GPU is not required. Other platforms have not been verified.

## 2. Installation

Clone the repository and enter the project root. Replace the URL if the repository is published under a different name.

```bash
git clone https://github.com/harshvisainiofficial/FaceRoster.git
cd FaceRoster
```

If you already have a local copy, simply enter that directory instead.

Create and activate a Conda environment:

```bash
conda create -n faceroaster python=3.11 -y
conda activate faceroaster
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
```

Alternatively, use a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
```

The lock file is intended to be generated from the working development environment. Clean-clone installation has not yet been independently verified. If InsightFace compilation fails, install the appropriate system build tools and inspect the error before changing package versions.

## 3. Required model files (important)

The model weights are deliberately excluded from GitHub. The recognition model exceeds GitHub's normal individual file size limit, and the pretrained weights have separate licensing restrictions. The empty `models/` directory is intentional.

The development setup uses the InsightFace Buffalo-L model pack:

| Original filename | Required local filename | Purpose |
|---|---|---|
| `det_10g.onnx` | `models/detector.onnx` | SCRFD face detection and landmarks |
| `w600k_r50.onnx` | `models/recognizer.onnx` | ArcFace embeddings |

### Obtain and install the models

1. Visit the [official InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) or [Python package model documentation](https://github.com/deepinsight/insightface/tree/master/python-package).
2. Review the pretrained model terms and obtain the Buffalo-L archive only if your intended use is authorized. The pretrained models are generally provided for non-commercial research; the code license does not automatically grant rights to the weights. Company or commercial use may require separate permission.
3. The [official Buffalo-L release archive](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) is the expected source. If unavailable, consult the official documentation rather than substituting an unverified model.
4. After obtaining the archive, extract it and copy the two required files. The example assumes the archive is in Downloads and contains the ONNX files at its root:

```bash
mkdir -p models
mkdir -p ~/Downloads/buffalo_l
unzip ~/Downloads/buffalo_l.zip -d ~/Downloads/buffalo_l
find ~/Downloads/buffalo_l -name '*.onnx'
cp ~/Downloads/buffalo_l/det_10g.onnx models/detector.onnx
cp ~/Downloads/buffalo_l/w600k_r50.onnx models/recognizer.onnx
ls -lh models/
```

Adjust the source paths if extraction creates a nested directory. The expected local files are approximately 17 MB and 167 MB, respectively.

If `MODEL_CHECKSUMS.txt` is included, verify the files from the project root:

```bash
sha256sum -c MODEL_CHECKSUMS.txt
```

Verify that both models load locally:

```bash
python - <<'PY'
import onnxruntime as ort
for path in ['models/detector.onnx', 'models/recognizer.onnx']:
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    print(path, 'loaded successfully')
    print('Inputs:', [(x.name, x.shape) for x in session.get_inputs()])
    print('Outputs:', [(x.name, x.shape) for x in session.get_outputs()])
PY
```

The recognizer expects 112 × 112 face inputs and produces 512-dimensional embeddings. Do not rely on automatic model downloading at runtime.

## 4. Run FaceRoster

From the project root:

```bash
conda activate faceroaster
export GRADIO_ANALYTICS_ENABLED=False
export NO_ALBUMENTATIONS_UPDATE=1
python -m src.app_backup
```

Open **http://127.0.0.1:7860** in your browser on the same machine. If the final UI has been renamed to `src/app.py`, use `python -m src.app` instead. The documented entry point is based on the supplied previous README.

Press Ctrl+C in the terminal to stop the application. If port 7860 is occupied, inspect the existing service before stopping it.

## 5. How to use the application

### Reference enrollment

1. Prepare a folder of consented or appropriately licensed reference photographs, ideally one clearly visible face per image.
2. Name each file after the person, for example `Priya Sharma.jpg` or `Aarav Mehta.png`.
3. Click **Select reference folder** and select the directory.
4. Click **Enroll Reference Faces** and review the registered identities and enrollment report.

Supported enrollment formats include JPG, JPEG, PNG and WebP. A different folder can contain entirely different identities. No identity list is hardcoded. Adaptive recovery may select a dominant foreground face in ambiguous photographs; review such selections because they can be incorrect.

### Group photo recognition

1. Upload a group photograph.
2. Start with detection resolution 640 and cosine similarity threshold 0.45.
3. Click **Recognize Faces**.
4. Review the annotated image, recognition details and known/unknown counts.
5. Download the annotated image and CSV if required.

The default threshold is experimental. Cosine similarity is not a calibrated identity probability. A new enrollment replaces the current gallery. Reset clears the UI session but may not remove cached or generated files from disk.

## 6. Optional command-line usage

Create your own `data/references/` and `data/practice/` folders and supply images before running these examples. Practice images are not included in the repository.

```bash
mkdir -p data/references data/practice outputs
```

Detection:

```bash
python -m src.detect_faces --image data/practice/group3.jpg \
  --model models/detector.onnx --output outputs/detected.jpg --size 640
```

Enrollment:

```bash
python -m src.enrollment --folder data/references
```

Recognition:

```bash
python -m src.recognition --references data/references \
  --image data/practice/recognition_test.jpg \
  --output outputs/recognized.jpg --csv outputs/recognition_results.csv \
  --threshold 0.45 --size 640
```

The CLI may retain the original strict enrollment behavior even when adaptive recovery is integrated in the UI.

## 7. Project structure

```text
FaceRoster/
├── README.md
├── requirements-lock.txt
├── MODEL_CHECKSUMS.txt
├── AI_COLLAB.md
├── RESEARCH.md
├── models/                 # Local weights, excluded from Git
│   ├── .gitkeep
│   ├── detector.onnx       # Obtain separately
│   └── recognizer.onnx     # Obtain separately
├── data/                   # Local input photographs
├── outputs/                # Generated results
├── docs/
├── src/
│   ├── __init__.py
│   ├── detect_faces.py
│   ├── enrollment.py
│   ├── recognition.py
│   ├── reference_detection.py
│   └── app_backup.py
└── tests/
```

The exact source and companion documentation filenames may vary with the final repository revision.

## 8. Architecture

```text
Reference photos → SCRFD → Face alignment → ArcFace → Normalized gallery
Group photo → SCRFD → Face alignment → ArcFace → Cosine similarity
                                              → Known / Unknown
                                              → Image + table + CSV
```

## 9. Offline verification and development observations

After installing dependencies and models, disconnect from the internet, restart the app, enroll a new folder, recognize a group photo and export both results. Repeat after restarting the app. Check logs for network attempts.

- The standalone detector was tested on three group photographs online and offline. One run detected 13 faces in 0.382 seconds, including model loading.
- Initial enrollment succeeded for 5 of 5 reference photographs.
- Expanded enrollment initially succeeded for 13 of 17 photographs. Adaptive detection was subsequently introduced; its final recovery rate is not recorded.
- Nine CLI recognition runs completed, with 41 detected faces in total; two additional images failed to decode. These counts are not accuracy measurements.
- Observed recognition runtimes were approximately 3.0–4.1 seconds, including loading and re-enrollment.

Full offline UI verification, network auditing, threshold calibration and clean-clone testing remain unverified in the supplied evidence. These results are development observations, not a formal benchmark.

## 10. Troubleshooting

| Problem | Suggested action |
|---|---|
| ModuleNotFoundError | Activate the environment and install the lock file. |
| Model not found | Check the two local ONNX paths from the project root. |
| Image cannot be read | Verify the actual image format and re-encode if necessary. |
| No face or multiple faces in a reference | Review image quality and adaptive recovery notes. |
| Known face labeled Unknown | Review pose, lighting, occlusion and similarity; avoid blindly lowering the threshold. |
| Browser cannot connect | Confirm the app is running and check its printed local URL. |
| Albumentations update warning | Set `NO_ALBUMENTATIONS_UPDATE=1` before importing dependencies. |

## 11. Limitations and privacy

Recognition can be affected by pose, blur, lighting, sunglasses, small faces and poor crops. Adaptive face selection is heuristic. The threshold is not calibrated for arbitrary identities, and independent matching may assign the same identity to multiple faces. GPU performance has not been evaluated.

Reference images, face embeddings and outputs are sensitive biometric data. Use authorized images, review results manually, minimize retention and do not use this experimental application as the sole basis for consequential identity or access decisions.

Model weights, private photographs, generated outputs, caches and credentials are excluded from Git. See `AI_COLLAB.md`, `RESEARCH.md` and any accompanying architecture documentation for development decisions.

## 12. References

- [InsightFace](https://github.com/deepinsight/insightface)
- [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [Gradio File component](https://www.gradio.app/docs/gradio/file)
- [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)
