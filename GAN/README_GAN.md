# Seeing Sounds II: GAN Approach

Teaching AI to dream in sound by generating images from audio files.

This section details the implementation of the Generative Adversarial Network (GAN) approach for the "Seeing Sounds II" project. Unlike the diffusion-based methods explored in this project, this approach trains a custom generator and discriminator end-to-end to learn the mapping between CLAP audio embeddings and visual structures.

The development was iterative, evolving from a baseline 64x64 model to 128×128 then to 256×256, and eventually settling on an optimized 128×128 architecture that prioritizes stability over raw resolution.

---

## How It Works

The model implements a **Conditional GAN (cGAN)** architecture regarding the Generator (conditioned on audio), but uses an **Unconditional Discriminator** that focuses purely on visual realism, relying on auxiliary losses for semantic alignment.

### Audio Encoding

Raw audio waveforms are processed by a frozen CLAP (Contrastive Language-Audio Pretraining) encoder to extract semantic audio features
((e \in \mathbb{R}^{512})).

### Projection & Concatenation

These features are projected via an MLP to a higher dimension (2048) and concatenated with a random Gaussian noise vector
((z \in \mathbb{R}^{128})), creating a combined latent vector.

### Generation

The Generator progressively upsamples this combined vector through transpose convolutional layers to produce an RGB image.

### Alignment

* **Realism:** The Discriminator checks if the image looks real.
* **Semantics:** A Contrastive Loss (using frozen CLIP/CLAP encoders) ensures the generated image semantically matches the input audio.

---

## Model Architecture (v7)

This describes the architecture of the final version (birdv7).

---

### Generator

The generator employs a transposed convolutional network with Self-Attention mechanisms to ensure global coherence.

* **Input:** Concatenated vector (128-dim Noise + 2048-dim Projected Audio).
* **Structure:** 5 blocks of ConvTranspose2d with Batch Normalization and ReLU activations, upsampling from (4 \times 4) to (128 \times 128).
* **Attention:** A Self-Attention block is applied after the second upsampling block, at the 16×16 resolution feature map.
* **Output:** Tanh activation producing values in ([-1, 1]).

---

### Discriminator

The discriminator is an Unconditional patch-based classifier. It does not receive the audio embedding as input.

* **Input:** Image (128 \times 128).
* **Spectral Normalization:** Applied to all convolutional layers to constrain the Lipschitz constant, preventing mode collapse.
* **Architecture:** 5 layers of downsampling convolutions (Kernel 4, Stride 2) reducing the image to a final validity score.

---

## Loss Functions

The models leverage a comprehensive "cocktail" of loss functions.


### 1. Adversarial Loss (BCE)

Standard binary cross-entropy loss. The Discriminator tries to classify real vs. fake, while the Generator tries to fool the Discriminator.

[
\mathcal{L}_{adv} = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z, e)))]
]


### 2. Contrastive Loss (Alignment)

Ensures the generated image is semantically similar to the input audio.

[
\mathcal{L}*{cont} = \text{CrossEntropy}(\text{logits}(e*{audio}, e_{img}))
]


### 3. Perceptual Loss (VGG)

Crucial for stability. Computes the Mean Squared Error (MSE) between feature maps of the generated image and the real image using VGG16.

Implementation uses 3 layers: **relu1_2**, **relu2_2**, **relu3_3**.


### 4. R1 Gradient Penalty

Applied to the Discriminator during training to penalize rapid changes in gradients, ensuring training stability.


### 5. Total Variation (TV) Loss

Introduced in v4. Penalizes high-frequency noise by minimizing differences between adjacent pixels.

[
\mathcal{L}*{TV} = \sum*{i,j} |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|
]

---

## File & Version Overview

The development process is captured across six Jupyter Notebooks.

---

### **Baseline Phase**

#### **birdv2.ipynb — Scaled-up Baseline (128×128)**

* Architecture: 128px, Attention at 16×16
* Training: 150 Epochs
* Status: Initial working model, but unstable

#### **birdv3.ipynb — Extended Training**

* Architecture: Same as v2
* Training: 200 Epochs
* Key Change: Adjusted loss weights (high contrastive weight initially)

---

### **Refinement Phase**

#### **birdv4.ipynb — Noise Reduction**

* Key Change: Introduced Total Variation (TV) Loss
* Training: 250 Epochs
* Goal: Smooth out high-frequency noise observed in v2/v3

#### **birdv5.ipynb — High-Resolution Experiment (256×256)**

* Architecture: Scaled to 256×256
* Added block6 (upsample)
* Attention at 32×32
* Training: 400 Epochs
* Result: Expensive and harder to converge

---

### **Optimization Phase**

#### **birdv6.ipynb — Augmentation & EMA (256×256)**

* Architecture: 256×256, Attention at 32×32
* Added: Horizontal Flip, Color Jitter
* Added: EMA for generator weights
* Training: 400 Epochs

#### **birdv7.ipynb — Final Optimized Model (128×128)**

* Architecture: Reverted to 128×128
* Attention back to 16×16
* Features: EMA + Augmentation
* Training: 300 Epochs
* Status: Most balanced and high-performing model

---

## Key Findings

* **Core Architecture Viability:**
  The CLAP → MLP → GAN pipeline successfully learns to generate bird shapes conditioned only on audio using concatenated noise embeddings.

* **Stability > Resolution:**
  High-res 256×256 models (v5/v6) were expensive and artifact-prone. The optimized 128×128 model (v7) produced more coherent structures.

* **Perceptual Loss is Critical:**
  The multi-layer Perceptual Loss was the single most effective stabilizing factor against generator collapse.