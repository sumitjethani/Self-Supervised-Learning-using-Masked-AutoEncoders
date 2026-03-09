# Masked Autoencoder (MAE) From Scratch in PyTorch

Self-supervised image representation learning using Masked Autoencoders, implemented from scratch using base PyTorch layers.


---

## What is MAE?

The idea is simple — take an image, randomly hide 75% of its patches, and train a model to reconstruct the missing parts. No labels, no supervision. The model learns meaningful visual representations just by filling in blanks.

To reconstruct well, the model has to understand textures, shapes, edges, and spatial context — not because it was told to, but because that's the only way it can do its job.

---

## Architecture

An asymmetric encoder-decoder design built entirely with base PyTorch:

| Component | Details |
|-----------|---------|
| **Encoder** | ViT-Base B/16 — 12 layers, dim=768, 12 heads, ~86M params |
| **Decoder** | ViT-Small S/16 — 12 layers, dim=384, 6 heads, ~22M params |
| **Masking** | 75% random patch masking (147/196 patches hidden) |
| **Patch Size** | 16×16 pixels |
| **Image Size** | 224×224 |
| **Pos. Embedding** | 2D Sinusoidal (fixed) |

The encoder processes **only the 25% visible patches** — no mask tokens are fed to it. The decoder receives the encoded visible tokens along with learnable mask tokens and reconstructs the full image at pixel level.

---

## Training

| Setting | Value |
|---------|-------|
| Dataset | TinyImageNet (100k images, 200 classes) |
| Platform | Kaggle — GPU T4 × 2 |
| Epochs | 35 (early stopping) |
| Batch Size | 64 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Scheduler | Cosine LR with 5 epoch linear warmup |
| Loss | MSE on masked patches only |
| Regularization | Dropout (0.1), Weight Decay (0.05) |
| Precision | Mixed precision (float16) |
| Early Stopping | Patience = 7 epochs |
| Gradient Clipping | 1.0 |

**Training results:**
- Train loss: 0.866 → 0.386
- Val loss: 0.740 → 0.383
- No overfitting — train and val loss stayed extremely close throughout

---

## Project Structure

```
├── MAE_Assignment.ipynb   # Main notebook — full implementation
├── app.py                 # Gradio demo app
├── model.py               # MAE architecture (encoder + decoder)
├── utils.py               # Patchify, visualisation helpers
├── config.py              # All hyperparameters
├── requirements.txt       # Dependencies
```

---

## Live Demo

Try the model live on HuggingFace Spaces — upload any image, adjust the masking ratio, and watch the model reconstruct the missing patches in real time.

👉 **[HuggingFace Space — Live Demo](#)**

---

## Model Weights

The trained checkpoint is hosted on HuggingFace due to GitHub's file size limit.

👉 **[Download mae_best.pth](#)**

To use locally, download `mae_best.pth` and place it in the root directory.

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/masked-autoencoder
cd masked-autoencoder
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model weights**

Download `mae_best.pth` from the HuggingFace link above and place it in the root folder.

**4. Run the Gradio app**
```bash
python app.py
```

**5. Or run the full notebook**

Open `MAE_Assignment.ipynb` on Kaggle with GPU T4 × 2 and add the TinyImageNet dataset from [here](https://www.kaggle.com/datasets/akash2sharma/tinyimagenet).

---

## Results

**Quantitative:**
| Metric | Score |
|--------|-------|
| PSNR | evaluated on val set |
| SSIM | evaluated on val set |

**Qualitative:**

The model reconstructs masked patches with reasonable textures, colors, and structure despite never seeing 75% of the image during encoding.

---

## References

- [Masked Autoencoders Are Scalable Vision Learners — He et al. 2021](https://arxiv.org/abs/2111.06377)
- [An Image is Worth 16x16 Words — Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929)

---

*Built by Sumit Jethani*
