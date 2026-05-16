# Robust-Canvas-Bot

**A degradation-aware vision-to-robot pipeline.** Takes an image — possibly low-light or motion-blurred — detects what's wrong with it, restores only what needs restoring, runs modality-specific perception, and produces safe robot stroke paths to redraw it on a canvas.

The interesting part is not the drawing. It's the **architecture**: instead of assuming clean input or blindly stacking filters, the pipeline *measures* the degradation present and routes accordingly. The evaluation is honest about where the architecture pays off (low-light restoration) and where the current implementation underperforms (motion-blur deconvolution, scene classification) — both with clear next experiments.

---

## Architecture

```
Input image
   │
   ▼
[1] ADRR — Adaptive Degradation-Aware Restoration Router
     • low-light detection   (mean brightness + 10th-percentile intensity)
     • motion-blur detection (Laplacian-variance focus measure)
     • routes selectively → Zero-DCE enhancement / Wiener deconvolution
   │
   ▼
[2] Content Classifier  (logistic regression on 6 handcrafted features)
     → TEXT or SCENE
   │
   ├── SCENE ──▶ [3a] Volumetric Perception
   │              MiDaS-small monocular depth → Sobel surface normals
   │              → parametric hatching (strokes tangent to normal gradient)
   │
   └── TEXT  ──▶ [3b] Linguistic Perception
                  EasyOCR detection + spell correction
                  → 36-glyph affine vector synthesis
   │
   ▼
[4] Robot Simulation  (PyBullet, 6-DOF KUKA IIWA)
     • geofence            (workspace bounds)
     • singularity check   (det(J·Jᵀ) manipulability measure)
     • collision check     (closest-point monitoring)
     • headless 2D renderer for batch evaluation
```

### Stage 1 — Adaptive Degradation-Aware Restoration Router (ADRR)

`src/perception/restoration/adaptive_restoration.py`

Computes lightweight image statistics and applies *only* the restoration that is needed:

- **Low-light:** mean grayscale brightness `< 90` **and** 10th-percentile intensity `< 40` (double-threshold to avoid false positives on properly-exposed dark scenes).
- **Motion blur:** variance of the Laplacian response `< 30` (focus measure, after Pertuz et al. 2013).
- If low-light → **Zero-DCE** (Li-Chongyi iterative curve estimation; pretrained weights, with a gamma-correction fallback).
- If motion blur → **Wiener deconvolution** with a fixed horizontal PSF + unsharp mask.
- If neither → image passes through untouched.

The point of detect-then-route is that blindly stacking restoration *degrades* clean images — Zero-DCE washes out a well-exposed photo, Wiener with a misspecified PSF adds ringing. The evaluation below confirms this is a genuine concern for the current Wiener path.

### Stage 2 — Content Classifier

`src/perception/content_classifier.py`

Logistic regression over 6 handcrafted features (grayscale variance, Canny edge density, Sobel H/V energy ratio, mean gradient magnitude, Otsu connected-component density, DCT high-frequency ratio), trained on 400 auto-generated synthetic images (200 text / 200 scene). Routes the cleaned image to the volumetric or linguistic perception engine.

### Stage 3 — Modality-specific perception

- **Volumetric** (`src/perception/vision/depth_to_hatching.py`): MiDaS-small monocular depth → Sobel-gradient surface normals → parametric hatching, strokes laid tangent to the normal gradient at a fixed grid density.
- **Linguistic** (`src/perception/text/ocr_hatching.py`): EasyOCR detection + `pyspellchecker` correction → a skeletal glyph library (A–Z, 0–9, punctuation) → affine transformation mapping each glyph onto its OCR bounding box.

### Stage 4 — Robot simulation

`src/simulation/robot_sim_3d_env.py`

6-DOF KUKA IIWA in PyBullet with three safety protocols enforced at the controller level:

- **Geofence** — workspace bounds on the end-effector position.
- **Singularity** — `det(J·Jᵀ)` of the 3×N translational Jacobian (the manipulability measure; `det(J)` doesn't apply to a non-square Jacobian). Halts below `1e-4`.
- **Collision** — `getClosestPoints` between robot and table.

A headless `RobotDrawingRenderer` renders stroke paths to a 2D canvas for batch evaluation without a GUI.

---

## Evaluation

`src/tests/benchmark_generator.py` builds a synthetic benchmark — 13 clean source images (5 scene / 5 text / 3 mixed) × 12 degradation configs (low-light at 4 severities, motion blur at 4 kernels/angles, 2 Gaussian blurs, 2 compound dark+blur) = **156 test cases** with a self-documenting manifest.

`src/tests/eval_metrics.py` runs the full suite. Tables below are auto-generated; full per-image CSVs live in `outputs/`.

### Restoration quality (PSNR / SSIM)

| Degradation | N | Δ PSNR | Δ SSIM |
|---|---|---|---|
| Low-light (dark) | 52 | **+3.38 dB** | **+0.276** |
| Combined (dark + blur) | 26 | **+3.38 dB** | **+0.254** |
| Motion blur | 52 | −0.51 dB | −0.004 |
| Gaussian blur | 26 | **−5.59 dB** | −0.050 |
| **Overall** | **156** | **+0.59 dB** | **+0.124** |

### ADRR routing accuracy

| Detector | Accuracy |
|---|---|
| Low-light detection | 81.4% (127 / 156) |
| Motion-blur detection | 58.3% (91 / 156) |
| Combined | 69.9% (218 / 312) |

### Edge recovery vs baselines

| Method | Overall | Dark | Blur | Combined |
|---|---|---|---|---|
| Canny only | 26.6% | 41.3% | 24.7% | 2.6% |
| CLAHE + Canny | 32.0% | 51.9% | 26.3% | 9.3% |
| HistEq + Canny | 36.1% | 89.7% | 8.6% | 11.6% |
| Adaptive Threshold | **55.4%** | 59.0% | 47.2% | **72.8%** |
| **Ours (ADRR + Canny)** | 32.3% | 63.4% | 19.8% | 7.6% |

### Content classifier (test set)

| Content type | Accuracy | N |
|---|---|---|
| Text | 100.0% | 60 |
| Scene | 3.3% | 60 |
| Mixed | 5.6% | 36 |
| **Overall** | **41.0%** | **156** |

### Ablation

| Configuration | Avg SSIM | Δ vs degraded |
|---|---|---|
| Degraded input (no restoration) | 0.6085 | — |
| **Full pipeline (ADRR + restoration)** | **0.7330** | **+0.1245** |

---

## Honest assessment of the results

These are the conclusions an interviewer should expect me to defend; they are also the next-experiment list.

**Where the architecture pays off.** ADRR with the Zero-DCE low-light path is a clear win — **+3.38 dB PSNR and +0.276 SSIM on low-light inputs**, and **81% routing accuracy on low-light detection**. The detect-then-route premise is validated specifically for low-light: the pipeline handles brightness recovery without blowing out properly-exposed images, and the gain transfers to compound dark+blur cases (+3.38 dB / +0.254 SSIM).

**Where the current implementation underperforms.**

1. **Wiener deconvolution with a fixed horizontal PSF degrades Gaussian-blurred inputs (−5.59 dB)** and is essentially flat on directional motion blur (−0.51 dB). The PSF mismatch is the obvious culprit. The right next experiment is blind deconvolution that estimates the PSF jointly with the latent image, or a learned deblurring head (NAFNet / Restormer-style). Motion-blur detection accuracy (58%) is also weak, suggesting the Laplacian-variance threshold needs joint calibration with the restoration module rather than independent tuning.
2. **The content classifier overfits toward TEXT** — 100% recall on text but 3% on scenes. The classifier was trained on 400 synthetic images and the features (high-frequency DCT energy, Otsu connected-component density) over-weight text-like signal. The fix is straightforward: enrich the synthetic scene generator with more diverse geometry, or move to a small learned feature extractor.
3. **Edge recovery is uneven against baselines.** ADRR + Canny beats Canny-alone (32.3% vs 26.6%) and roughly ties CLAHE + Canny, but is outperformed by HistEq + Canny on dark images (89.7%) and by Adaptive Threshold overall (55.4%). The honest read: restoration is helping where it should (low-light), but the gain is not yet enough to dominate simpler baselines that target the same regime. Combining ADRR with histogram-aware Canny thresholding is the obvious next ablation.
4. **OCR legibility is not meaningfully improved by restoration** (average −0.4 detections per image across all degradation types). EasyOCR's own internal preprocessing is already handling much of what Zero-DCE / Wiener would add, so the marginal value of upstream restoration for downstream OCR is small in this configuration.

**Net.** The architectural pattern — detect, route, restore selectively, then perceive — is sound. The Zero-DCE branch validates the pattern. The Wiener branch and the content classifier are the two specific modules to replace in the next iteration; both have well-defined fixes.

---

## Run it

```bash
conda activate dl-env          # or your environment
pip install -r requirements.txt
# single image through the full pipeline
python src/main.py --input path/to/image.jpg --render

# generate benchmark + run evaluation
python src/tests/benchmark_generator.py
python src/tests/eval_metrics.py
```

Outputs land in `outputs/`: `empirical_results.md`, `restoration_metrics.csv`, `edge_recovery_metrics.csv`, `ocr_metrics.csv`.

## Project structure

```
src/
  main.py                              pipeline orchestrator (DrawingRouter)
  perception/
    restoration/adaptive_restoration.py ADRR — degradation detection + routing
    restoration/zero_dce.py             low-light enhancement
    restoration/deblur.py               Wiener deconvolution
    content_classifier.py               TEXT vs SCENE logistic-regression gate
    vision/depth_to_hatching.py         MiDaS depth → normals → hatching paths
    text/ocr_hatching.py                EasyOCR → glyph vector synthesis
  simulation/robot_sim_3d_env.py        PyBullet KUKA + safety protocols + renderer
  utils/data_degradation.py             synthetic degradation models
  tests/benchmark_generator.py          156-case synthetic benchmark
  tests/eval_metrics.py                 full evaluation suite
```

## Known limitations and next experiments

- **Wiener deconvolution with fixed horizontal PSF** — the dominant performance issue. Replace with blind deconvolution or a learned deblurring head.
- **Content classifier overfits toward TEXT** — enrich the synthetic scene generator with more diverse geometry, or replace handcrafted features with a small CNN feature extractor.
- **Motion-blur detection at 58% accuracy** — Laplacian-variance threshold needs joint calibration with the restoration module rather than independent threshold tuning.
- **Synthetic benchmark only** — no real-world camera noise tested yet. A small held-out set of real low-light + motion-blurred images would tighten the analysis.
- **Foreground masking via depth threshold** is a heuristic that doesn't generalize across scenes.
- **No multi-frame fusion / temporal smoothing** — a real moving-camera application would want both; out of scope here.

## Acknowledgements

Zero-DCE architecture from Li et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement* (2020). MiDaS depth estimation from Intel ISL. Laplacian-variance focus measure after Pertuz et al. (2013), *Pattern Recognition* 46(5). Built as an independent project.
