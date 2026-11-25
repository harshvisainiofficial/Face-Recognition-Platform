# AI Collaboration — FaceRoster

This document records the engineering choices discussed with AI during the development of FaceRoster. I used AI for implementation guidance, debugging and interpreting experimental outputs, while testing the application locally.

## Decision 001 — Pretrained models instead of training from scratch

**AI suggested:** Use pretrained SCRFD for face detection and ArcFace for face embeddings instead of training a new model.

**I decided:** Proceed with the pretrained-model approach and test detection before moving to recognition.

**Why:** The task requires a working offline recognition application, and pretrained models provide a practical starting point without a separate training pipeline.

**Evidence:** My SCRFD detector correctly processed three practice group photographs in my reported tests. The run on `group3.jpg` detected 13 faces. See chat excerpt #1.

## Decision 002 — Reference image quality and ambiguity

**AI suggested:** Reject reference photographs containing zero or multiple detected faces rather than guessing which face belongs to the filename.

**I decided:** Replace the problematic photographs with clearer, single-person photographs during the initial enrollment test.

**Why:** An ambiguous reference image could associate the wrong face with a person's name.

**Evidence:** The subsequent enrollment run successfully processed all five reference images with zero failures, as recorded in my initial decision notes.

**Practical lesson:** Reliable enrollment depends on both model performance and reference image quality.

## Decision 003 — CPU-based local inference

**AI suggested:** Use ONNX Runtime with `CPUExecutionProvider` to load and run the models locally.

**I decided:** Test the detector using CPU inference before proceeding with the recognition pipeline.

**Why:** This provides a practical execution path without requiring GPU-specific runtime configuration.

**Evidence:** The terminal confirmed `CPUExecutionProvider`. The detector processed `group3.jpg` in 0.382 seconds, including model loading, and saved the annotated image. This is a single observed runtime, not a general performance benchmark.

## Decision 004 — Verify offline operation through actual tests

**AI suggested:** Avoid unnecessary network connections, including the Albumentations version check, by setting `NO_ALBUMENTATIONS_UPDATE` before importing InsightFace.

**I decided:** Test the detector both with and without internet connectivity. These tests established the offline detection milestone before proceeding further.

**Why:** Offline operation is a central requirement of FaceRoster. Successful execution with internet access alone would not establish that requirement.

**Evidence:** I reported correct detection results on three group photographs in online and offline tests. The terminal also showed a failed Albumentations version check while detection completed successfully. See chat excerpts #1 and #2.

**Verification limit:** The supplied chat does not establish that the proposed environment-variable fix was subsequently verified or that the entire application was tested for all network requests.

## Decision 005 — Develop and validate the pipeline in stages

**AI suggested:** Keep the working detector unchanged and implement reference enrollment as a separate module before adding identity matching.

**I decided:** Complete and test the detection milestone first, then proceed toward enrollment and recognition.

**Why:** Testing one component at a time makes it easier to isolate failures and avoid disturbing a working detector.

**Evidence:** I supplied the detection output and confirmed the available detector model and practice images before moving to the next milestone. See chat excerpt #3. The excerpt establishes the development sequence; it does not independently verify the later module implementation.

## Chat excerpt #1 — Detection and offline testing

> Me: I tried on three group photos with both online and offline and I got correct results in all of them.
>
> AI: The reported 0.382 seconds includes model loading, so it is not yet a steady-state inference benchmark.

## Chat excerpt #2 — Unnecessary network requests

> AI: This is not a face detection error. Albumentations attempted to check its latest version over the internet, which failed because you were offline.
>
> AI: However, our application should not attempt unnecessary network connections.

## Chat excerpt #3 — Staged development

> AI: We will implement this as a separate src/enrollment.py module and test it before implementing identity matching.
>
> AI: Send me the output of those three commands. Then we'll build and test the enrollment module without disturbing your working face detector.

## Full AI chat history

The excerpts above are selected portions of the development conversation, not the complete chat history. The complete conversation, starting with the first prompt, must be attached to the repository separately as an exported file or share link before submission.
