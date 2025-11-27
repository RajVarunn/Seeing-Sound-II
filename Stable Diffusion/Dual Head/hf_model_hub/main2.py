"""
Audio → Image Generator (Multi-Task Loss Version)
Key features:
    - Dual-head MLP: one for CLAP text space, one for SD embedding space
    - Multi-task training: CLAP alignment loss + SD alignment loss
    - Both heads are trained simultaneously
    - to_sd head is properly trained and used during inference
"""

# ========================
#  Imports
# ========================
import os, math, csv, random, sys
from typing import List, Tuple
from dataclasses import dataclass
import zipfile
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoProcessor, ClapModel, AutoTokenizer, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
from PIL import Image
from torchvision import transforms


# ========================
#  Configuration
# ========================
@dataclass
class Config:
    CLAP_ID: str = "laion/clap-htsat-fused"
    SD_ID: str   = "runwayml/stable-diffusion-v1-5"
    CLIP_ID: str = "openai/clip-vit-base-patch32"
    
    # Device configuration - automatically uses GPU if available
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    lr: float = 2e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    
    # Multi-task loss weights
    clap_loss_weight: float = 0.5
    sd_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    
    batch_size: int = 2  # Reduced for Mac GPU memory
    max_epochs: int = 20
    base_prompt: str = "A photo of"
    guidance: float = 7.5
    steps: int = 30
    
    # Dataset paths
    train_csv: str = "/Users/rajvarun/Desktop/SIT/Trimester 4/AAI 3001 - Computer Vision & Deep Learning/Seeing Sound II/raj/main_dataV1.csv"
    image_folder: str = "/Users/rajvarun/OneDrive - Singapore Institute Of Technology/ALEXI KIZHAKKEPURATHU GEORGE's files - VGGSound"  # OneDrive folder with ZIP files
    ckpt_path: str = "audio2image_mapper_dual_best.pt"
    
    # ZIP file support (if data is in ZIP files instead of extracted)
    use_zip_files: bool = True  # Set to True to read from ZIP files directly
    zip_files: dict = None  # Will be populated automatically
    
    # Fine-tuning control
    finetune_sd: bool = False  # Set to False to train without images
    sd_lr: float = 1e-5
    freeze_vae: bool = True
    freeze_text_encoder: bool = True
    
    # Evaluation settings
    eval_every_n_epochs: int = 1  # Evaluate every N epochs
    num_eval_samples: int = 4  # Number of samples to evaluate per batch
    save_eval_images: bool = True  # Save example generated images


# ========================
#  Dataset
# ========================
class AudioCaptionDataset(Dataset):
    """
    Reads a CSV file with audio-image-caption triplets.
    Handles structure where data is in: base_folder/image/ and base_folder/audio/
    
    Can read from extracted folders OR directly from ZIP files (no extraction needed!)
    
    Example:
    - CSV: vggsound_00,g-f_I2yQ_1.png,g-f_I2yQ_000001.wav,people marching
    - Audio path: vggsound_00/audio/g-f_I2yQ_000001.wav
    - Image path: vggsound_00/image/g-f_I2yQ_1.png
    """
    def __init__(self, captions_path: str, image_folder: str = None, use_zip_files: bool = False):
        self.items = []
        base_dir = os.path.dirname(captions_path)
        self.image_folder = image_folder or base_dir
        self.use_zip_files = use_zip_files
        self.zip_handles = {}  # Cache opened ZIP files
        
        # Image preprocessing for SD (512x512, normalized to [-1, 1])
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"Loading dataset from: {captions_path}")
        print(f"Base folder: {self.image_folder}")
        print(f"Use ZIP files: {use_zip_files}")
        
        # If using ZIP files, find and open them
        if use_zip_files:
            self._find_zip_files()
        
        # Read CSV file
        import csv
        with open(captions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                # CSV format: base_folder,image_file,audio_file,caption
                if 'base_folder' in row and 'image_file' in row and 'audio_file' in row and 'caption' in row:
                    base_folder = row['base_folder']  # e.g., "vggsound_00"
                    img_filename = row['image_file']  # e.g., "g-f_I2yQ_1.png"
                    audio_filename = row['audio_file']  # e.g., "g-f_I2yQ_000001.wav"
                    caption = row['caption']
                    
                    if use_zip_files:
                        # Use ZIP file paths
                        audio_path = f"{base_folder}/audio/{audio_filename}"
                        img_path = f"{base_folder}/image/{img_filename}"
                        
                        # Check if files exist in ZIP
                        audio_exists = self._file_in_zip(base_folder, audio_path)
                        img_exists = self._file_in_zip(base_folder, img_path)
                        
                        # Debug first few rows
                        if row_num <= 3:
                            print(f"Row {row_num}: base_folder='{base_folder}', audio='{audio_path}', exists={audio_exists}")
                    else:
                        # Use regular file paths
                        audio_path = os.path.join(self.image_folder, base_folder, "audio", audio_filename)
                        img_path = os.path.join(self.image_folder, base_folder, "image", img_filename)
                        
                        audio_exists = os.path.exists(audio_path)
                        img_exists = os.path.exists(img_path)
                    
                    if audio_exists:
                        if img_exists:
                            self.items.append((base_folder, audio_path, img_path, caption))
                        else:
                            # Audio exists but image doesn't
                            self.items.append((base_folder, audio_path, None, caption))
                            if row_num <= 3:
                                print(f"Warning: Image not found: {img_path}")
                    else:
                        if row_num <= 3:
                            print(f"Warning: Audio not found: {audio_path}")
                else:
                    if row_num <= 3:
                        print(f"Warning: Row {row_num} missing required columns")
        
        if not self.items:
            raise ValueError("Empty dataset: no valid audio files found")
        
        # Count how many have images
        with_images = sum(1 for _, _, img_path, _ in self.items if img_path is not None)
        print(f"✓ Loaded {len(self.items)} audio files ({with_images} with matching images)")
    
    def _find_zip_files(self):
        """Find and open ZIP files in the image_folder"""
        print("Searching for ZIP files...")
        for item in os.listdir(self.image_folder):
            if item.endswith('.zip'):
                zip_name = item.replace('.zip', '')
                zip_path = os.path.join(self.image_folder, item)
                try:
                    self.zip_handles[zip_name] = zipfile.ZipFile(zip_path, 'r')
                    # Get number of files in ZIP for debugging
                    file_count = len(self.zip_handles[zip_name].namelist())
                    print(f"  ✓ Opened {item} (key: '{zip_name}', {file_count} files)")
                except Exception as e:
                    print(f"  ✗ Failed to open {item}: {e}")
    
    def _file_in_zip(self, base_folder, file_path):
        """Check if a file exists in the corresponding ZIP"""
        if base_folder not in self.zip_handles:
            print(f"    ! ZIP handle not found for base_folder='{base_folder}'. Available: {list(self.zip_handles.keys())}")
            return False
        try:
            self.zip_handles[base_folder].getinfo(file_path)
            return True
        except KeyError:
            return False
    
    def _read_from_zip(self, base_folder, file_path):
        """Read a file from ZIP archive"""
        if base_folder in self.zip_handles:
            return self.zip_handles[base_folder].read(file_path)
        return None

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx: int):
        base_folder, audio_path, img_path, cap = self.items[idx]
        
        # Load audio
        if self.use_zip_files:
            # Read audio from ZIP
            audio_bytes = self._read_from_zip(base_folder, audio_path)
            if audio_bytes is None:
                raise FileNotFoundError(f"Audio not found in ZIP: {audio_path}")
            wav, sr = torchaudio.load(BytesIO(audio_bytes))
        else:
            # Read from file system
            wav, sr = torchaudio.load(audio_path)
        
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).float()
        # Resample to 48kHz for CLAP
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            wav = resampler(wav)
        
        # Load image if available
        if img_path is not None:
            if self.use_zip_files:
                # Read image from ZIP
                img_bytes = self._read_from_zip(base_folder, img_path)
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    img_tensor = self.img_transform(img)
                else:
                    img_tensor = torch.zeros((3, 512, 512))
            else:
                # Read from file system
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.img_transform(img)
        else:
            # Create dummy image if not available
            img_tensor = torch.zeros((3, 512, 512))
        
        return wav, 48000, cap, img_tensor, (img_path is not None)
    
    def __del__(self):
        """Close ZIP files when done"""
        for zip_handle in self.zip_handles.values():
            try:
                zip_handle.close()
            except:
                pass

def collate_audio(batch):
    wavs, srs, caps, imgs, has_imgs = [], [], [], [], []
    for w, sr, c, img, has_img in batch:
        wavs.append(w)
        srs.append(sr)
        caps.append(c)
        imgs.append(img)
        has_imgs.append(has_img)
    return wavs, srs[0], caps, torch.stack(imgs), torch.tensor(has_imgs)


# ========================
#  Model Components
# ========================
class AudioProjectionMLP(nn.Module):
    """
    Dual-head MLP projection:
    - to_text: CLAP audio → CLAP text space (for CLAP alignment)
    - to_sd: CLAP audio → SD embedding space (for image generation)
    Both heads are trained with multi-task loss.
    """
    def __init__(self, in_dim, text_dim, sd_dim, hidden=1024):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), 
            nn.GELU(), 
            nn.Dropout(0.1)
        )
        
        # Head 1: CLAP text space (for training alignment)
        self.to_text = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, text_dim)
        )
        
        # Head 2: SD embedding space (for generation)
        self.to_sd = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, sd_dim)
        )
        
    def forward(self, z):
        shared_features = self.shared(z)
        return self.to_text(shared_features), self.to_sd(shared_features)


# ========================
#  Main Model
# ========================
class Audio2ImageModel(nn.Module):
    def __init__(self, cfg: Config, load_sd: bool = False):
        super().__init__()
        self.cfg = cfg
        device = cfg.device

        # -------- Frozen CLAP --------
        print("Loading CLAP model...")
        self.clap = ClapModel.from_pretrained(cfg.CLAP_ID).eval().to(device)
        for p in self.clap.parameters(): 
            p.requires_grad = False
        self.proc = AutoProcessor.from_pretrained(cfg.CLAP_ID)

        # -------- CLIP for Evaluation (Frozen) --------
        print("Loading CLIP for evaluation...")
        self.clip_model = CLIPModel.from_pretrained(cfg.CLIP_ID).eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_ID)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        print("  ✓ CLIP loaded (frozen for evaluation only)")

        # -------- Stable Diffusion (conditionally trainable) --------
        self.sd_pipe = None
        self.sd_tok = None
        self.sd_text_encoder = None
        self.sd_unet = None
        self.sd_vae = None
        self.sd_hidden = 768
        
        # Always load full SD for training or inference
        if True:
            print("Loading Stable Diffusion...")
            # Use float32 for training, float16 for inference only
            dtype = torch.float32 if cfg.finetune_sd else (torch.float16 if device == "cuda" else torch.float32)
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(cfg.SD_ID, torch_dtype=dtype)
            self.sd_pipe.to(device)
            
            self.sd_tok = self.sd_pipe.tokenizer
            self.sd_text_encoder = self.sd_pipe.text_encoder
            self.sd_unet = self.sd_pipe.unet
            self.sd_vae = self.sd_pipe.vae
            self.sd_hidden = self.sd_pipe.text_encoder.config.hidden_size
            
            # Configure trainability based on config
            if cfg.finetune_sd:
                print("🔥 End-to-End Training Mode:")
                
                # UNet: TRAINABLE (this learns to generate!)
                for p in self.sd_unet.parameters():
                    p.requires_grad = True
                self.sd_unet.train()
                print("  ✓ UNet: TRAINABLE")
                
                # VAE: Usually frozen for stability
                if cfg.freeze_vae:
                    for p in self.sd_vae.parameters():
                        p.requires_grad = False
                    self.sd_vae.eval()
                    print("  ✓ VAE: FROZEN")
                else:
                    for p in self.sd_vae.parameters():
                        p.requires_grad = True
                    self.sd_vae.train()
                    print("  ✓ VAE: TRAINABLE")
                
                # Text Encoder: Usually frozen
                if cfg.freeze_text_encoder:
                    for p in self.sd_text_encoder.parameters():
                        p.requires_grad = False
                    self.sd_text_encoder.eval()
                    print("  ✓ Text Encoder: FROZEN")
                else:
                    for p in self.sd_text_encoder.parameters():
                        p.requires_grad = True
                    self.sd_text_encoder.train()
                    print("  ✓ Text Encoder: TRAINABLE")
            else:
                print("Inference Mode: All SD components frozen")
                for comp in (self.sd_unet, self.sd_vae, self.sd_text_encoder):
                    for p in comp.parameters(): 
                        p.requires_grad = False
                    comp.eval()

        # -------- Get CLAP dims --------
        dummy_text = ["test"]
        dummy_audio = [torch.zeros(48000).numpy()]
        
        with torch.no_grad():
            text_proc = self.proc(text=dummy_text, return_tensors="pt")
            text_proc = {k: v.to(device) for k,v in text_proc.items()}
            t = self.clap.get_text_features(**text_proc)
            clap_text_dim = t.shape[-1]
            
            audio_proc = self.proc(audio=dummy_audio, sampling_rate=48000, return_tensors="pt")
            audio_proc = {k: v.to(device) for k,v in audio_proc.items()}
            a = self.clap.get_audio_features(**audio_proc)
            clap_audio_dim = a.shape[-1]

        # -------- Trainable Dual-Head MLP --------
        print(f"Creating MLP: CLAP audio ({clap_audio_dim}) → CLAP text ({clap_text_dim}) & SD ({self.sd_hidden})")
        self.mapper = AudioProjectionMLP(clap_audio_dim, clap_text_dim, self.sd_hidden)

    # --- Encoders ---
    def encode_text_clap(self, caps):
        """Encode text using CLAP text encoder"""
        proc = self.proc(text=caps, return_tensors="pt", padding=True)
        proc = {k: v.to(self.cfg.device) for k,v in proc.items()}
        
        # Ensure CLAP is in eval mode
        was_training = self.clap.training
        self.clap.eval()
        
        with torch.no_grad():
            e = self.clap.get_text_features(**proc)
        
        # Restore training state if needed
        if was_training:
            self.clap.train()
            
        return F.normalize(e, dim=-1)
    
    def encode_text_sd(self, caps):
        """Encode text using SD text encoder (for target embeddings)"""
        tokens = self.sd_tok(
            caps,
            padding="max_length",
            max_length=self.sd_tok.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.cfg.device)
        
        with torch.no_grad():
            # Get the pooled output (last hidden state mean)
            outputs = self.sd_text_encoder(tokens["input_ids"])
            # Use pooler_output if available, else mean pool
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings

    def encode_audio(self, wavs, sr):
        """Returns raw CLAP audio embeddings - batched processing"""
        # Convert all wavs to numpy for batch processing
        audio_list = [w.cpu().numpy() for w in wavs]
        
        # Process all audios in a single batch
        proc = self.proc(audio=audio_list, sampling_rate=sr, return_tensors="pt")
        proc = {k: v.to(self.cfg.device) for k, v in proc.items()}
        
        # Ensure CLAP is in eval mode to avoid batch norm issues
        was_training = self.clap.training
        self.clap.eval()
        
        with torch.no_grad():
            embeddings = self.clap.get_audio_features(**proc)
        
        # Restore training state if needed
        if was_training:
            self.clap.train()
        
        return embeddings

    # --- Loss ---
    @staticmethod
    def info_nce(a, b, temp):
        """InfoNCE contrastive loss"""
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = a @ b.t() / temp
        tgt = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
    
    def compute_diffusion_loss(self, images, audio_emb):
        """
        Diffusion loss: Trains SD UNet to denoise images conditioned on audio.
        This enables end-to-end learning of the generative model!
        
        Args:
            images: Ground truth images [B, 3, 512, 512] in range [-1, 1]
            audio_emb: Audio embeddings from CLAP
        
        Returns:
            Denoising loss (MSE between predicted and actual noise)
        """
        # 1. Encode images to latent space (no grad through VAE)
        with torch.no_grad():
            latents = self.sd_vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # SD's scaling factor
        
        # 2. Sample random timesteps for diffusion training
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, 1000, (bsz,), 
            device=latents.device
        ).long()
        
        # 3. Add noise to latents according to timestep
        if not hasattr(self, 'noise_scheduler'):
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.cfg.SD_ID, 
                subfolder="scheduler"
            )
        
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 4. Get audio conditioning (gradients flow to mapper!)
        _, audio_to_sd = self.mapper(audio_emb)
        
        # Reshape for UNet: [batch, 1, hidden_dim]
        encoder_hidden_states = audio_to_sd.unsqueeze(1)
        
        # 5. UNet predicts noise (THIS IS WHERE SD LEARNS! ✅)
        noise_pred = self.sd_unet(
            noisy_latents,              # Noisy input
            timesteps,                   # Time conditioning
            encoder_hidden_states        # Audio conditioning
        ).sample
        
        # 6. Compute denoising loss
        # Gradients flow back to: UNet ✅ and Mapper ✅
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        return loss
    
    @torch.inference_mode()
    def evaluate_generation(self, wavs, sr, captions, num_samples=None):
        """
        Evaluate quality of generated images using CLIP text-image similarity.
        
        Args:
            wavs: List of audio waveforms
            sr: Sample rate
            captions: List of text captions describing the audio
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            avg_clip_score: Average CLIP similarity score (0-100)
            generated_images: List of PIL images
            clip_scores: List of individual CLIP scores
        """
        was_training = self.training
        self.eval()
        
        if num_samples is not None:
            wavs = wavs[:num_samples]
            captions = captions[:num_samples]
        
        generated_images = []
        clip_scores = []
        
        for wav, caption in zip(wavs, captions):
            # Generate image from audio
            img = self.generate(wav, sr)
            generated_images.append(img)
            
            # Compute CLIP score (text-image similarity)
            inputs = self.clip_processor(
                text=[caption],
                images=[img],
                return_tensors="pt",
                padding=True
            ).to(self.cfg.device)
            
            outputs = self.clip_model(**inputs)
            
            # Get similarity score (logits are already scaled by temperature)
            # Higher score = better match between image and caption
            logits_per_image = outputs.logits_per_image
            clip_score = logits_per_image[0, 0].item()
            clip_scores.append(clip_score)
        
        avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
        
        if was_training:
            self.train()
        
        return avg_clip_score, generated_images, clip_scores

    # --- Forward (Training with Multi-Task Loss) ---
    def forward(self, wavs, sr, caps, images=None, has_images=None):
        """
        Forward pass with three parallel losses:
        1. CLAP alignment (semantic understanding)
        2. SD embedding alignment (embedding compatibility)
        3. Diffusion loss (pixel-level generation) - requires images
        
        All losses train simultaneously in end-to-end fashion!
        """
        # Get target embeddings (frozen encoders)
        clap_text_emb = self.encode_text_clap(caps)
        sd_text_emb = self.encode_text_sd(caps)
        
        # Get audio embeddings
        audio_emb = self.encode_audio(wavs, sr)
        
        # Project audio to both spaces (gradients flow here!)
        audio_to_clap, audio_to_sd = self.mapper(audio_emb)
        
        # Loss 1: CLAP alignment (InfoNCE)
        loss_clap = self.info_nce(audio_to_clap, clap_text_emb, self.cfg.temperature)
        
        # Loss 2: SD embedding alignment (MSE)
        loss_sd = F.mse_loss(audio_to_sd, sd_text_emb)
        
        # Loss 3: Diffusion loss (pixel-level generation)
        loss_diffusion = torch.tensor(0.0, device=self.cfg.device)
        if self.cfg.finetune_sd and images is not None:
            # Only compute on samples that have images
            if has_images is not None:
                valid_mask = has_images.to(self.cfg.device)
                if valid_mask.sum() > 0:
                    valid_imgs = images[valid_mask]
                    valid_audio_emb = audio_emb[valid_mask]
                    loss_diffusion = self.compute_diffusion_loss(valid_imgs, valid_audio_emb)
            else:
                loss_diffusion = self.compute_diffusion_loss(images, audio_emb)
        
        # Combined multi-task loss - all train together! 🚀
        total_loss = (
            self.cfg.clap_loss_weight * loss_clap + 
            self.cfg.sd_loss_weight * loss_sd +
            self.cfg.diffusion_loss_weight * loss_diffusion
        )
        
        # Compute similarities for monitoring
        with torch.no_grad():
            clap_sim = torch.diagonal(
                F.normalize(audio_to_clap, dim=-1) @ F.normalize(clap_text_emb, dim=-1).t()
            ).mean()
            
            sd_sim = F.cosine_similarity(audio_to_sd, sd_text_emb, dim=-1).mean()
        
        return total_loss, {
            "loss_clap": loss_clap.item(),
            "loss_sd": loss_sd.item(),
            "loss_diffusion": loss_diffusion.item(),
            "clap_sim": clap_sim.item(),
            "sd_sim": sd_sim.item()
        }

    # --- Inference ---
    @torch.inference_mode()
    def generate(self, wav, sr):
        if self.sd_pipe is None:
            raise RuntimeError("Stable Diffusion not loaded. Init with load_sd=True.")
        
        # Get audio embedding and project to SD space
        audio_emb = self.encode_audio([wav], sr)
        _, soft_token = self.mapper(audio_emb)  # Use to_sd head
        
        # Tokenize base prompt
        tok = self.sd_tok(
            self.cfg.base_prompt, 
            padding="max_length",
            max_length=self.sd_tok.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.cfg.device)
        
        # Get SD text embeddings
        enc = self.sd_text_encoder(tok["input_ids"])[0]
        
        # Find position to insert audio token (after last real token)
        attention_mask = tok["attention_mask"][0]
        last_token_pos = attention_mask.nonzero(as_tuple=False).max().item()
        
        # Insert audio soft token AFTER the last token
        if last_token_pos + 1 < enc.shape[1]:
            enc[0, last_token_pos + 1:last_token_pos + 2, :] = soft_token
        else:
            # If no space, replace the last token
            enc[0, last_token_pos:last_token_pos + 1, :] = soft_token
        
        # Generate image
        img = self.sd_pipe(
            num_inference_steps=self.cfg.steps, 
            guidance_scale=self.cfg.guidance,        # 7.5
            prompt_embeds=enc
        ).images[0]
        
        return img


# ========================
#  Training
# ========================
def train(cfg: Config):
    # Load dataset with images
    full_ds = AudioCaptionDataset(cfg.train_csv, cfg.image_folder, use_zip_files=cfg.use_zip_files)
    
    # Create train/validation split (90/10)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_ds)} samples")
    print(f"  Validation: {len(val_ds)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=collate_audio,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_audio,
        num_workers=0
    )
    
    # Initialize model
    model = Audio2ImageModel(cfg, load_sd=True).to(cfg.device)
    
    # Separate optimizers with different learning rates
    if cfg.finetune_sd:
        print("\n🔥 Setting up END-TO-END training:")
        
        # Optimizer 1: Mapper (higher LR)
        opt_mapper = torch.optim.AdamW(
            model.mapper.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        print(f"  Mapper optimizer: LR={cfg.lr}")
        
        # Optimizer 2: SD UNet (lower LR for stability)
        opt_sd = torch.optim.AdamW(
            model.sd_unet.parameters(),
            lr=cfg.sd_lr,
            weight_decay=cfg.weight_decay
        )
        print(f"  SD UNet optimizer: LR={cfg.sd_lr}")
        
        opts = [opt_mapper, opt_sd]
    else:
        # Only train mapper
        opt_mapper = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        opts = [opt_mapper]

    print(f"\n{'='*60}")
    print(f"Starting {'End-to-End' if cfg.finetune_sd else 'Mapper-Only'} Training")
    print(f"{'='*60}")
    print(f"Dataset: {len(full_ds)} samples ({len(train_ds)} train, {len(val_ds)} val)")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.max_epochs}")
    print(f"Evaluation: Every {cfg.eval_every_n_epochs} epoch(s)")
    print(f"Loss weights:")
    print(f"  CLAP: {cfg.clap_loss_weight}")
    print(f"  SD Embedding: {cfg.sd_loss_weight}")
    if cfg.finetune_sd:
        print(f"  Diffusion: {cfg.diffusion_loss_weight}")
    print(f"{'='*60}\n")
    
    # Track best model based on CLIP score
    best_clip_score = -float('inf')
    
    for ep in range(1, cfg.max_epochs + 1):
        # ============================================
        # TRAINING PHASE
        # ============================================
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{cfg.max_epochs} [TRAIN]")
        
        epoch_stats = {
            "total": 0, "clap": 0, "sd": 0, "diff": 0,
            "clap_sim": 0, "sd_sim": 0
        }
        
        for wavs, sr, caps, imgs, has_imgs in pbar:
            wavs = [w.to(cfg.device) for w in wavs]
            imgs = imgs.to(cfg.device)
            
            # Forward pass - all losses computed!
            loss, stats = model(wavs, sr, caps, imgs if cfg.finetune_sd else None, has_imgs)
            
            # Zero gradients for all optimizers
            for opt in opts:
                opt.zero_grad()
            
            # Backward pass - gradients flow to mapper AND UNet!
            loss.backward()
            
            # Clip gradients for stability
            if cfg.finetune_sd:
                nn.utils.clip_grad_norm_(model.mapper.parameters(), 1.0)
                nn.utils.clip_grad_norm_(model.sd_unet.parameters(), 1.0)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update all parameters simultaneously! 🚀
            for opt in opts:
                opt.step()
            
            # Accumulate stats
            epoch_stats["total"] += loss.item()
            epoch_stats["clap"] += stats['loss_clap']
            epoch_stats["sd"] += stats['loss_sd']
            epoch_stats["diff"] += stats['loss_diffusion']
            epoch_stats["clap_sim"] += stats['clap_sim']
            epoch_stats["sd_sim"] += stats['sd_sim']
            
            pbar.set_postfix({
                "total loss": f"{loss.item():.3f}",
                "diff": f"{stats['loss_diffusion']:.3f}",
                "c_sim": f"{stats['clap_sim']:.2f}", 
                "s_sim": f"{stats['sd_sim']:.2f}"
            })
        
        # Compute training epoch averages
        n_train = len(train_loader)
        for k in epoch_stats:
            epoch_stats[k] /= n_train
        
        # ============================================
        # VALIDATION & EVALUATION PHASE
        # ============================================
        if ep % cfg.eval_every_n_epochs == 0:
            print(f"\n{'='*60}")
            print(f"🔍 Evaluating Epoch {ep}...")
            print(f"{'='*60}")
            
            model.eval()
            val_clip_scores = []
            all_gen_images = []
            all_captions = []
            
            # Evaluate on validation set (limit to save time)
            eval_batches = min(3, len(val_loader))  # Max 3 batches
            
            for batch_idx, (wavs, sr, caps, imgs, has_imgs) in enumerate(val_loader):
                if batch_idx >= eval_batches:
                    break
                
                wavs = [w.to(cfg.device) for w in wavs]
                
                # Generate images and compute CLIP scores
                avg_score, gen_imgs, scores = model.evaluate_generation(
                    wavs, sr, caps, 
                    num_samples=cfg.num_eval_samples
                )
                
                val_clip_scores.extend(scores)
                all_gen_images.extend(gen_imgs)
                all_captions.extend(caps[:cfg.num_eval_samples])
                
                print(f"  Batch {batch_idx + 1}/{eval_batches}: Avg CLIP = {avg_score:.3f}")
            
            # Compute overall validation CLIP score
            avg_val_clip = sum(val_clip_scores) / len(val_clip_scores) if val_clip_scores else 0.0
            
            # Save example images from evaluation
            if cfg.save_eval_images and all_gen_images:
                os.makedirs("eval_samples", exist_ok=True)
                for i, (img, cap, score) in enumerate(zip(all_gen_images[:4], all_captions[:4], val_clip_scores[:4])):
                    save_path = f"eval_samples/ep{ep}_sample{i}_score{score:.2f}.png"
                    img.save(save_path)
                    print(f"  Sample {i}: '{cap[:50]}...' | CLIP: {score:.3f}")
                    print(f"    Saved to: {save_path}")
            
            # Clear MPS cache after evaluation
            if cfg.device == "mps":
                torch.mps.empty_cache()
            
            print(f"\n{'='*60}")
            print(f"📊 Epoch {ep} Summary:")
            print(f"{'='*60}")
            print(f"Training Metrics:")
            print(f"  Total Loss: {epoch_stats['total']:.4f}")
            print(f"  CLAP Loss: {epoch_stats['clap']:.4f} | Sim: {epoch_stats['clap_sim']:.3f}")
            print(f"  SD Loss: {epoch_stats['sd']:.4f} | Sim: {epoch_stats['sd_sim']:.3f}")
            if cfg.finetune_sd:
                print(f"  Diffusion Loss: {epoch_stats['diff']:.4f}")
            print(f"\nValidation Metrics:")
            print(f"  🎯 CLIP Score: {avg_val_clip:.3f} (higher = better image-text match)")
            print(f"{'='*60}\n")
            
        else:
            # Just print training stats if not evaluating
            avg_val_clip = None
            print(f"\n{'='*60}")
            print(f"Epoch {ep} Summary:")
            print(f"  Total Loss: {epoch_stats['total']:.4f}")
            print(f"  CLAP Loss: {epoch_stats['clap']:.4f} | Sim: {epoch_stats['clap_sim']:.3f}")
            print(f"  SD Loss: {epoch_stats['sd']:.4f} | Sim: {epoch_stats['sd_sim']:.3f}")
            if cfg.finetune_sd:
                print(f"  Diffusion Loss: {epoch_stats['diff']:.4f}")
            print(f"{'='*60}\n")
        
        # ============================================
        # CHECKPOINT SAVING
        # ============================================
        checkpoint = {
            "mapper": model.mapper.state_dict(),
            "epoch": ep,
            "val_clip_score": avg_val_clip if avg_val_clip is not None else -1,
            **{k: v for k, v in epoch_stats.items()},
            "config": {
                "clap_loss_weight": cfg.clap_loss_weight,
                "sd_loss_weight": cfg.sd_loss_weight,
                "diffusion_loss_weight": cfg.diffusion_loss_weight,
                "finetune_sd": cfg.finetune_sd
            }
        }
        
        if cfg.finetune_sd:
            checkpoint["unet"] = model.sd_unet.state_dict()
        
        # Always save latest checkpoint
        torch.save(checkpoint, cfg.ckpt_path)
        print(f"💾 Checkpoint saved: {cfg.ckpt_path}")
        
        # Save best model based on CLIP score
        if avg_val_clip is not None and avg_val_clip > best_clip_score:
            best_clip_score = avg_val_clip
            best_path = cfg.ckpt_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ New best model! CLIP: {avg_val_clip:.3f} -> Saved to {best_path}")
        elif avg_val_clip is not None:
            print(f"   Current best CLIP: {best_clip_score:.3f}")
        
        print()
    
    print("🎉 Training completed!")
    if best_clip_score > -float('inf'):
        print(f"   Best CLIP score achieved: {best_clip_score:.3f}")


# ========================
#  Inference
# ========================
def infer(cfg: Config, wav_path: str, out_path: str):
    # Load audio
    print(f"Loading audio from {wav_path}...")
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1: 
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).float()
    
    # Resample to 48kHz for CLAP
    if sr != 48000:
        print(f"Resampling from {sr}Hz to 48000Hz...")
        resampler = torchaudio.transforms.Resample(sr, 48000)
        wav = resampler(wav)
        sr = 48000
    
    wav = wav.to(cfg.device)
    
    # Load model with SD
    model = Audio2ImageModel(cfg, load_sd=True).to(cfg.device)
    
    # Load trained weights
    print(f"Loading checkpoint from {cfg.ckpt_path}...")
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.mapper.load_state_dict(ckpt["mapper"])
    
    # Load UNet weights if available (from fine-tuning)
    if "unet" in ckpt:
        print("Loading fine-tuned UNet weights...")
        model.sd_unet.load_state_dict(ckpt["unet"])
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  CLAP Sim: {ckpt.get('clap_sim', 'N/A'):.3f}" if isinstance(ckpt.get('clap_sim'), (int, float)) else f"  CLAP Sim: N/A")
    print(f"  SD Sim: {ckpt.get('sd_sim', 'N/A'):.3f}" if isinstance(ckpt.get('sd_sim'), (int, float)) else f"  SD Sim: N/A")
    if "unet" in ckpt:
        print("  Fine-tuned UNet: ✓")
    
    # Generate image
    print("\nGenerating image...")
    img = model.generate(wav, sr)
    img.save(out_path)
    print(f"✓ Generated image saved to {out_path}")


# ========================
#  Main
# ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--wav", help="Audio file path for inference mode")
    parser.add_argument("--out", default="output.png", help="Output image path")
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}")
    
    if args.mode == "train":
        print(f"Dataset: {cfg.train_csv}")
        if not os.path.exists(cfg.train_csv):
            print(f"ERROR: Dataset not found at {cfg.train_csv}")
            print("Please ensure the captions.txt file exists")
            sys.exit(1)
        train(cfg)
    else:
        if not args.wav: 
            raise ValueError("Need --wav for inference mode")
        if not os.path.exists(args.wav):
            raise ValueError(f"Audio file not found: {args.wav}")
        infer(cfg, args.wav, args.out)