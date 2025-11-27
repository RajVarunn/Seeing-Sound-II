## Project Description

**Seeing Sounds II: Teaching AI to dream in sound by generating images from audio files**

This project explores how artificial intelligence can learn to "imagine" visuals from sounds by learning meaningful cross-modal relationships between audio features and visual generation. Using two separate strategies, our team first trained Stable Diffusion using CLAP audio embeddings, MLP-derived conditioning, and U-Net optimization, enabling the diffusion model to generate images from audio. We then introduced a custom GAN trained end-to-end on audio signals to learn direct cross-modal mapping. Both approaches produced visuals showing early semantic alignment with their audio inputs. Stable Diffusion generated more coherent and detailed imagery, while the GAN captured broader structural patterns derived from sound.

By combining techniques in scale consistency, fine-grained and broad multimodal mapping, our findings highlight the feasibility of multisensory generative systems and provide a solid foundation for future work such as embedding audio inside images for compact storage, generating visual equivalents of sound for the deaf or hard-of-hearing, and multimodal search. This approach brings us closer to AI capable of richer, more human-like sensory integration.


## Stable Diffusion (Dual Head)

### Project File & Folder Overview 
- **hf_model_hub**
	- `audio2image_mapper_dual_best.pt`: Holds the best model trained using MLP dual mapper + Unet Optimization.

- **hf_space**
	- `app.py`: Imports the model script, loads the best model, and provides the interface for the interactive demo in hugging space.
	- `main2.py`: Python script for dual-head MLP architecture with Unet Optimization; contains model architecture, training loop, evaluation logic, inference functions, dataset loading & preprocessing.
	- `requirements.txt`: Needed dependencies for Hugging Face deployment.

- `deploy_to_hf_hub.py`: Script to automate uploading the trained model and related files to the Hugging Face Model Hub.
- `app_audio2image.py`: The application interface for the audio-to-image model running on localhost.

- `hf_requirements_clean.txt`: Needed Python packages and dependencies for running the Hugging Face model.
- `main_dataV1.csv`: Dataset for training and evaluating the audio-to-image model; contains audio file paths, image file paths, and text captions.

- `main.py`: Python script for single-head MLP architecture without Unet Optimization.
- `main2.py`: Python script for dual-head MLP architecture with Unet Optimization; contains model architecture, training loop, evaluation logic, inference functions, dataset loading & preprocessing.
- `mlponly.py`: Python script for dual-head MLP optimization without Unet Optimization; contains model architecture, training loop, evaluation logic, inference functions, dataset loading & preprocessing.
- `main_dream_mode.py`: Uses main2’s model configurations; allows two audio inputs and fuses them using a fusion prompt rather than the base prompt.

- `audio2image_ui.html`: Local UI file & JavaScript functions for interactivity (button presses, etc.).
- `audio_2_image_mapper_dual_best`: Holds the best model trained using MLP dual mapper + Unet Optimization.
- `audio_2_image_mapper_dual_mlp_only_best`: Holds the best model trained using MLP dual mapper.

- `plot_training_loss.py`: Used to plot the training loss points from the training log for MLP (dual head) only model.
- `plot_unet_training_loss.py`: Used to plot the training loss points from the training log for MLP (dual head) + Unet optimized model.
- `requirements.txt`: Lists all dependencies needed for this project.
- `Seeing_Sound_II_MLP_ONLY.ipynb`: Training notebook for the MLP only model.
- `Seeing_Sound_II_MLP_UNET.ipynb`: Training notebook for the MLP + UNET model.

- `training_loss_curves.png`: Image generated from `plot_training_loss.py`.
- `unet_training_loss_curves.png`: Image generated from `plot_unet_training_loss.py`.

## Stable Diffusion (Single Head)

### Project File & Folder Overview 

## GANs

The GANs implementation uses a **Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate 64x64 pixel images conditioned on audio embeddings. It leverages **SPADE (Spatially-Adaptive Normalization)** in the generator and a **Projection Discriminator** to effectively fuse audio information into the visual generation process.

## How It Works

The model consists of two main components: a **Generator** and a **Discriminator**, which play a minimax game.

1.  **Generator ($G$):** Takes a random noise vector ($z$) and an audio embedding ($e$) as input. It progressively upsamples the noise to generate an image. The audio embedding modulates the features at each layer using **SPADE** blocks, ensuring the generated image structure is guided by the audio context.
2.  **Discriminator ($D$):** Takes an image (real or generated) and the corresponding audio embedding as input. It tries to distinguish between real images from the dataset and fake images produced by the generator. It uses a **Projection** mechanism to measure the compatibility between the image and the audio.

The training process uses the **WGAN-GP** objective for stability, augmented with **Feature Matching Loss** and an optional **LPIPS Perceptual Loss** to improve image quality and diversity.

## Model Architecture

### Generator
The generator uses a ResNet-based architecture with **SPADE** normalization blocks.
- **Input:** Noise vector $z \in \mathbb{R}^{128}$, Audio Embedding $e \in \mathbb{R}^{512}$.
- **SPADE Block:** Instead of standard Batch Normalization, SPADE modulates the normalized activation using learned scale ($\gamma$) and bias ($\beta$) parameters derived from the audio embedding.

$$ \gamma = \text{Conv}(\text{ReLU}(\text{Conv}(\text{ProjectedEmbedding}))) $$
$$ \beta = \text{Conv}(\text{ReLU}(\text{Conv}(\text{ProjectedEmbedding}))) $$
$$ \text{SPADE}(x, e) = \frac{x - \mu}{\sigma} \cdot (1 + \gamma(e)) + \beta(e) $$

- **Self-Attention:** Applied at 32x32 and 64x64 resolutions to capture long-range dependencies.

### Discriminator
The discriminator is a **Projection Discriminator** with **Spectral Normalization**.
- **Architecture:** A series of downsampling convolutional blocks with Spectral Normalization.
- **Projection:** The final score is a combination of a global realism score and a conditional compatibility score (dot product of image features and projected audio embedding).

$$ D(x, e) = \underbrace{\psi(x)^T \phi(e)}_{\text{Conditional Score}} + \underbrace{\psi'(x)}_{\text{Realism Score}} $$
where $\psi(x)$ are the image features and $\phi(e)$ is the projected audio embedding.

## Loss Functions

The model is trained using the following losses:

### 1. WGAN-GP Loss (Discriminator)
Ensures 1-Lipschitz continuity for the critic using a gradient penalty.

$$ L_D = \underbrace{\mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x}, e)] - \mathbb{E}_{x \sim P_r}[D(x, e)]}_{\text{Wasserstein Loss}} + \lambda_{gp} \underbrace{\mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x}, e)||_2 - 1)^2]}_{\text{Gradient Penalty}} $$

### 2. Generator Loss
Combines adversarial loss with feature matching and perceptual loss.

$$ L_G = \underbrace{-\mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x}, e)]}_{\text{Adversarial Loss}} + \lambda_{fm} L_{fm} + \lambda_{lpips} L_{lpips} $$

- **Feature Matching ($L_{fm}$):** Matches the statistics of the discriminator's intermediate features for real and fake images.
  $$ L_{fm} = ||D_{feat}(x) - D_{feat}(\tilde{x})||_1 $$
- **LPIPS ($L_{lpips}$):** (Optional) Perceptual loss using a pre-trained VGG network to ensure the generated image is perceptually similar to the target (if paired data is used/available in that context).

## Training

The training loop is implemented in `trainer.py` and can be executed via the `train.ipynb` notebook.

- **Optimizer:** Adam ($\beta_1=0.0, \beta_2=0.9$)
- **EMA:** Exponential Moving Average of generator weights is maintained for inference to produce higher quality samples.
- **n_critic:** The discriminator is updated 4 times for every generator update.

## Training Progress

Below is a visualization of the training progress over time:

![Training Progress](training_progress.gif)

