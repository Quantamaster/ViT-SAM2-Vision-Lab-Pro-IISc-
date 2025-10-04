# Q1 — Vision Transformer on CIFAR-10 (PyTorch, Colab)

This repository contains a Colab-ready Jupyter notebook `q1.ipynb` that implements a Vision Transformer (ViT) and trains it on CIFAR-10.

## Files to include in the repo
- `q1.ipynb` — Colab notebook with full code (model, training, evaluation).
- `README.md` — this file.

## How to run in Google Colab
1. Open a new Colab notebook.
2. Paste the notebook cells from `q1.ipynb` (or upload `q1.ipynb`) and set the runtime to **GPU**.
3. Run the cells in order.
4. Training config is found near the top in a `CFG` class — change `epochs`, `batch_size`, or model `embed_dim/depth` to explore the tradeoffs.
5. Best model is saved to `/content/vit_cifar_checkpt.pth`.

## Best model configuration (suggested, fits in Colab/T4-ish)
- `patch_size`: 4
- `embed_dim`: 192
- `depth`: 12
- `num_heads`: 3
- `batch_size`: 256 (increase if you have more GPU memory)
- `epochs`: 200
- `optimizer`: AdamW (lr 3e-4)
- `scheduler`: cosine warmup (warmup ~ 5 epochs)
- augmentations: RandomCrop + RandomHorizontalFlip; optional RandAugment
- regularization: label smoothing 0.1, weight decay 0.05, gradient clipping 1.0

## Tiny results table (published & expected)
| Source | CIFAR-10 Test Accuracy |
|---|---:|
| ViT paper (pre-trained + fine-tuned) — ViT-L variants (Table 5) | **99.38%** (pretrained on larger datasets). |
| Expected training-from-scratch in Colab (this notebook, with augmentations) | **~98.5% — 99.0%** (depends on runtime, epochs, and augmentations). |

> Note: achieving the 99.3%+ from the ViT paper typically requires large-scale pretraining (ImageNet-21k or JFT) — without pretraining you still can reach very high accuracy (~98.5–99.0) with good augmentations and training schedule.

## Short analysis & ablation ideas (bonus)
- **Patch size**: smaller patches increase sequence length and can improve local detail — tradeoff is memory/time. On CIFAR-10 (32×32), patch size 4 is a good balance.
- **Depth vs width**: scaling depth tends to help more than width for transformer-style models. Try depth 12 vs 8 and monitor overfitting.
- **Augmentations**: RandAugment / MixUp / CutMix reliably boost CIFAR performance.
- **Optimizer & reg**: AdamW with suitable weight decay + label smoothing works well; cosine LR with warmup is a standard strong choice.
- **Overlapping patches**: using conv-based patch embedding with stride < kernel_size (overlap) is worth exploring — often adds useful local inductive bias.

- # Q2 — Text-Driven Image Segmentation with SAM 2

This repository contains a **Google Colab notebook** that demonstrates **text-driven segmentation** of a chosen object in an image using **SAM 2** (or fallbacks).  

---

## Files
- `q2.ipynb` — runnable notebook implementing the pipeline.  
- `README.md` — this file.

---

## Pipeline Overview
1. **Load an image** — either a sample image or user-uploaded.  
2. **Accept a text prompt** — free-form description of the target object (e.g., "a red car").  
3. **Convert text to region seeds** — using **GroundingDINO**, **Grounded-SAM**, **GLIP**, or similar. Seeds can be bounding boxes or points.  
4. **Feed seeds to SAM 2** — produces high-quality segmentation masks of the object.  
5. **Display final mask overlay** — the notebook shows the mask over the image and saves it.  

**Optional (bonus)**: video extension for 10–30s clips. Masks are propagated across frames using optical flow or tracking, producing a segmented video.

---

## How to Run (Colab)
1. Open `q2.ipynb` in Google Colab.  
2. Run all cells from top to bottom.  
3. Upload your own image if desired (via Colab file upload UI or replace the sample).  
4. Adjust the `text_prompt` variable to target a different object.  

The notebook saves:
- `mask.png` — the binary mask of the detected object.  
- `overlay.png` — the mask overlay on the original image.  
- `README.md` — this documentation.  
- `propagated_mask.mp4` (if video extension is run).

---

## Limitations
- **SAM 2 availability**: Official SAM 2 weights or API may require manual download, licensing, or API keys. The notebook uses community fallbacks where needed.  
- **Grounding accuracy**: Grounding models (text → boxes) can be noisy, especially with ambiguous prompts. Refinement with points or additional boxes may be necessary.  
- **Runtime**: Large images can be slow on free Colab GPUs.  
- **Video propagation**: Optical-flow-based propagation is a naive method; accuracy may degrade with fast motion or occlusions.

---

## References
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)  
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)  
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)  
- Community Colab demos for SAM + GroundingDINO integration.



