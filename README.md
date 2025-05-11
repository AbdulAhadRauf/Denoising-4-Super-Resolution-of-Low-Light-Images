# DLP Jan 2025 NPPE3 - Denoising & 4× Super-Resolution of Low-Light Images

## 📄 Competition Link

[DLP Jan 2025 NPPE3 on Kaggle](https://www.kaggle.com/competitions/dlp-jan-2025-nppe-3/overview)

## 🏆 Overview

Build robust deep learning pipelines that simultaneously **denoise** and perform **4× super-resolution** on low-light images, transforming noisy, low-resolution inputs into high-quality, high-resolution outputs.

* **Start:** April 4, 2025
* **Close:** April 7, 2025
* **Metric:** Peak Signal-to-Noise Ratio (PSNR)

## 📂 Repository Structure

```
├── data/
│   ├── train/       # Noisy & GT pairs for training
│   ├── val/         # Noisy & GT pairs for validation
│   └── test/        # Noisy images only (for submission)
├── notebooks/       # Exploratory analysis & training scripts
├── src/             # Utility functions & custom modules
├── outputs/         # Model outputs & submission files
├── submission.py    # Script to generate `submission.csv`
└── README.md        # This file
```

## 🗃️ Dataset Description

* **Training Set:** Noisy low-res images + high-res clean ground truth.
* **Validation Set:** Same structure as training, for tuning and early stopping.
* **Test Set:** Only noisy low-res images; you submit super-resolved predictions.

The sample submission (`sample_submission.csv`) defines the required format: one row per test image, with flattened pixel values.

## 🛠️ Environment & Dependencies

* **Python 3.x**
* **PyTorch** & **Torchvision**
* **NumPy**, **Pandas**, **Pillow**, **OpenCV**, **scikit-image**
* **matplotlib**, **seaborn** (EDA)
* **tqdm** (progress bars)
* **basicsr**, **facexlib**, **gfpgan** (for RealESRGAN)
* **Kagglehub** (for data import in notebook)

Install all requirements via:

```bash
pip install -r requirements.txt
```

or individually:

```bash
pip install torch torchvision numpy pandas pillow opencv-python scikit-image matplotlib seaborn tqdm basicsr gfpgan facexlib kagglehub
```

## 🔍 Data Loading & EDA

1. **Custom loader** (`load_images_from_dlp_competition`): reads and aligns noisy/clean pairs.
2. **Exploratory Plots**: sample image pairs, pixel-intensity histograms, noise-difference maps.
3. **Baseline PSNR**: bicubic upsampling of noisy inputs vs. ground truth.

## 🏗️ Model Architectures

### 1. Enhanced Deep Super-Resolution (EDSR)

* **Paper:** *Accurate Image Super-Resolution Using Very Deep Residual Learning* (Lim *et al.*, 2017).
* **Core Idea:** a deep residual network with *n* residual blocks and a sub-pixel upsampling module.
* **Implementation Details:**

  * 8 residual blocks (`n_resblocks=8`)
  * 64 feature channels (`n_feats=64`)
  * Upscaling factor ×4
  * L1 loss, Adam optimizer (lr=1e-4), StepLR scheduler
  * Mixed-precision training with `torch.cuda.amp.GradScaler`

### 2. SwinIR (Swin Transformer for Image Restoration)

* **Paper:** *SwinIR: Image Restoration Using Swin Transformer* (Wang *et al.*, 2021)
* **Core Idea:** window-based self-attention in a hierarchical transformer backbone, excels at denoising and super-resolution.
* **Usage:**

  * Clone the [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) repo.
  * Download pretrained weights for denoising & SR.
  * Run official inference scripts on validation/test sets.

### 3. RealESRGAN

* **Paper:** *Real-World Super-Resolution via Kernel Estimation and Noise Injection* (Wang *et al.*, 2021)
* **Core Idea:** GAN-based enhancement with RRDB blocks, robust to real-world degradations.
* **Usage:**

  * Install `basicsr`, `facexlib`, `gfpgan`.
  * Run `inference_realesrgan.py -n RealESRGAN_x4plus --face_enhance` on model outputs for extra refinement.

## 🎓 Training Pipeline (EDSR)

1. **Dataset & DataLoader** with batch size=4.
2. **Criterion:** `nn.L1Loss()`
3. **Optimizer:** `Adam(model.parameters(), lr=1e-4)`
4. **Scheduler:** `StepLR(step_size=5, gamma=0.5)`
5. **Epoch Loop:**

   * `train_one_epoch`: forward, backward, step.
   * `validate`: compute loss, PSNR, SSIM on val set.
   * Save best checkpoint by validation PSNR.

## 📊 Evaluation Metrics

* **PSNR:** measures reconstruction fidelity.
* **SSIM:** measures perceptual structural similarity.

## 🚀 Inference & Submission

1. **Load best checkpoint** (`best_model.pth`).
2. **Run model** on test set (batch size=1).
3. **Clamp outputs** to \[0, 255], convert to uint8 images.
4. **Generate submission CSV**: flatten each output image into a row via `images_to_csv` utility.

Run:

```bash
python submission.py --input_dir outputs/EDSR --output submission.csv
```

## 💡 Skills & Techniques Demonstrated

* **Deep Learning Fundamentals:** CNNs, Transformers, GANs
* **PyTorch Proficiency:** custom `Dataset`/`DataLoader`, training loops, AMP, schedulers
* **Image Processing:** denoising, 4× super-resolution, noise analysis
* **Model Integration:** fine-tuning vs. pretrained inference (EDSR, SwinIR, RealESRGAN)
* **Performance Evaluation:** PSNR & SSIM computation
* **Data Visualization:** Matplotlib & Seaborn EDA
* **Kaggle Workflow:** data import, notebook structuring, submission generation

## 🏃 How to Reproduce

1. Clone this repo and navigate to `notebooks/`.
2. Install dependencies.
3. Place data under `data/`.
4. Open `train_EDSR.ipynb` for full pipeline.
5. Run inference notebooks or scripts for SwinIR and RealESRGAN.
6. Execute `submission.py` to produce `submission.csv`.

---

**Happy super-resolving!**
