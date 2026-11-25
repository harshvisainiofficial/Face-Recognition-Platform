# Research — FaceRoster

## 1. Research objective

My initial exploration focused on finding a practical way to detect and recognize faces locally without training a new model or relying on an online recognition service. The approach was developed through AI-assisted technical discussions and local experiments rather than an extensive literature review.

## 2. Tools and models considered

| Component | Selected approach | Reason |
| --- | --- | --- |
| Face detection | SCRFD | A pretrained detector suitable for locating multiple faces in an image. |
| Face recognition | ArcFace | Produces numerical face embeddings that can be used to compare identities. |
| Model execution | ONNX Runtime (CPU) | Allows local inference using downloaded ONNX model files. |
| Reference enrollment | Filename-based identity registration | Provides a straightforward way to associate reference photographs with names. |

The AI-assisted discussion identified the InsightFace Buffalo-L model package, including `det_10g.onnx` for detection and `w600k_r50.onnx` for recognition. Additional landmark and age/gender models were discussed but were not needed for the core task.

## 3. Alternatives and trade-offs

- **Training from scratch:** Not selected because the task can be approached using pretrained models without building a separate training pipeline.
- **Cloud-based recognition:** Not selected because the application is required to work offline.
- **GPU-dependent inference:** CPU inference was used for the initial detection tests, avoiding a requirement for GPU-specific configuration. No formal CPU-versus-GPU benchmark was performed in the supplied experiments.
- **Accepting ambiguous reference photographs:** Rejected in the initial enrollment approach because multiple faces could cause incorrect identity registration.

These were practical design choices, not the results of a formal comparison of competing face recognition models.

## 4. Data and initial experiments

I used three practice group photographs (`group1.jpg`, `group2.jpg`, and `group3.jpg`) to test face detection. I tested the detector with and without internet access and observed correct results in these initial tests.

For `group3.jpg`, the detector returned 13 faces and saved an annotated image. The reported runtime was 0.382 seconds, including model loading. The initial enrollment test also successfully processed five reference photographs after problematic photographs were replaced.

These were small functional tests, not a standardized accuracy evaluation. The supplied development notes do not document the original image-source URLs or establish permission to redistribute the photographs.

## 5. Technical references

The following links identify the tools and methods discussed during development and provide reference material for the selected approach. They are included for reproducibility, not as a claim that I conducted a detailed review of every paper.

- [InsightFace — official project](https://github.com/deepinsight/insightface)
- [InsightFace Python package and model information](https://github.com/deepinsight/insightface/tree/master/python-package)
- [SCRFD — Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714)
- [ArcFace — Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [ONNX Runtime — official documentation](https://onnxruntime.ai/docs/)

## 6. Limitations and licensing

The initial experiments establish basic feasibility but do not establish recognition accuracy across large or diverse datasets. The pretrained InsightFace weights were identified in the development discussion as subject to non-commercial research restrictions. Their suitability for the assessment and redistribution must be verified before sharing the model files. The original photograph sources and redistribution permissions should also be checked.
