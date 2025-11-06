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
from diffusers import StableDiffusionPipeline
from PIL import Image


# ========================
#  Configuration
# ========================
@dataclass
class Config:
    CLAP_ID: str = "laion/clap-htsat-fused"
    SD_ID: str   = "runwayml/stable-diffusion-v1-5"
    CLIP_ID: str = "openai/clip-vit-base-patch32"
    
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    lr: float = 2e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    clap_loss_weight: float = 0.5
    sd_loss_weight: float = 1.0
    batch_size: int = 4
    max_epochs: int = 20
    base_prompt: str = "A photo of"
    guidance: float = 7.5
    steps: int = 30
    
    # Evaluation settings
    eval_every_n_epochs: int = 1
    num_eval_samples: int = 4
    save_eval_images: bool = True
    
    # Dataset settings
    train_csv: str = "/Users/rajvarun/Desktop/SIT/Trimester 4/AAI 3001 - Computer Vision & Deep Learning/Seeing Sound II/extracted_audiocaps/captions.txt"
    audio_folder: str = None  # If None, uses directory of train_csv
    use_zip_files: bool = False  # Set to True to read audio from ZIP files
    ckpt_path: str = "audio2image_mapper_dual.pt"


# ========================
#  Dataset
# ========================
class AudioCaptionDataset(Dataset):
    """Reads a CSV file with format: base_folder,image_file,audio_file,caption"""
    def __init__(self, captions_path: str, audio_folder: str = None, use_zip_files: bool = False):
        self.items = []
        self.use_zip_files = use_zip_files
        self.zip_handles = {}
        
        base_dir = os.path.dirname(captions_path)
        self.audio_folder = audio_folder if audio_folder else base_dir
        
        print(f"Loading dataset from: {captions_path}")
        print(f"Base directory: {base_dir}")
        print(f"Audio folder: {self.audio_folder}")
        print(f"Use ZIP files: {use_zip_files}")
        
        # If using ZIP files, find and open them
        if use_zip_files:
            self._find_zip_files()
        
        with open(captions_path, "r", encoding="utf-8") as f:
            # Skip header row
            header = next(f)
            print(f"Skipped header: {header.strip()}")
            
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma (CSV format)
                parts = line.split(",")
                
                if len(parts) >= 4:
                    base_folder = parts[0]  # e.g., "vggsound_00"
                    # image_file = parts[1]  # Not needed for MLP-only training
                    audio_file = parts[2]   # e.g., "g-f_I2yQ_000001.wav"
                    caption = parts[3]      # e.g., "people marching"
                    
                    # Build path to audio
                    if use_zip_files:
                        # Path inside ZIP: vggsound_00/audio/filename.wav
                        audio_path = f"{base_folder}/audio/{audio_file}"
                        
                        # Check if file exists in ZIP
                        if self._file_in_zip(base_folder, audio_path):
                            self.items.append((base_folder, audio_path, caption))
                        elif line_num <= 5:
                            print(f"  Warning: Audio not found in ZIP at line {line_num}: {audio_path}")
                    else:
                        # Path on disk: base_dir/vggsound_00/audio/filename.wav
                        audio_path = os.path.join(self.audio_folder, base_folder, "audio", audio_file)
                        
                        # Check if audio exists on disk
                        if os.path.exists(audio_path):
                            self.items.append((base_folder, audio_path, caption))
                        elif line_num <= 5:
                            print(f"  Warning: Audio not found at line {line_num}: {audio_path}")
                else:
                    if line_num <= 5:
                        print(f"  Warning: Invalid line {line_num} (expected 4 columns, got {len(parts)})")
        
        if not self.items:
            error_msg = "Empty dataset: no valid audio files found.\n"
            if use_zip_files:
                error_msg += f"Expected audio in ZIP files: {self.audio_folder}/vggsound_XX.zip\n"
                error_msg += "Check that ZIP files exist and contain audio files at: vggsound_XX/audio/*.wav"
            else:
                error_msg += f"Expected audio structure: {self.audio_folder}/vggsound_XX/audio/*.wav\n"
                error_msg += "Please extract audio files from ZIP archives first."
            raise ValueError(error_msg)
        
        print(f"✓ Loaded {len(self.items)} audio-caption pairs")
    
    def _find_zip_files(self):
        """Find and open ZIP files in the audio_folder"""
        print("Searching for ZIP files...")
        
        for item in os.listdir(self.audio_folder):
            if item.endswith('.zip'):
                # Extract base name (e.g., "vggsound_00.zip" → "vggsound_00")
                zip_name = item.replace('.zip', '')
                zip_path = os.path.join(self.audio_folder, item)
                
                try:
                    # Open ZIP file and keep it in memory
                    self.zip_handles[zip_name] = zipfile.ZipFile(zip_path, 'r')
                    
                    file_count = len(self.zip_handles[zip_name].namelist())
                    print(f"  ✓ Opened {item} (key: '{zip_name}', {file_count} files)")
                except Exception as e:
                    print(f"  ✗ Failed to open {item}: {e}")
    
    def _file_in_zip(self, base_folder, file_path):
        """Check if a file exists in the corresponding ZIP"""
        if base_folder not in self.zip_handles:
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

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        if self.use_zip_files:
            # Load from ZIP
            base_folder, audio_path, cap = self.items[idx]
            
            # Read audio bytes from ZIP
            audio_bytes = self._read_from_zip(base_folder, audio_path)
            
            if audio_bytes is None:
                raise FileNotFoundError(f"Audio not found in ZIP: {audio_path}")
            
            # Load audio from bytes
            wav, sr = torchaudio.load(BytesIO(audio_bytes))
        else:
            # Load from disk
            base_folder, path, cap = self.items[idx]
            wav, sr = torchaudio.load(path)
        
        # Convert to mono if needed
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).float()
        
        # Resample to 48kHz for CLAP
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            wav = resampler(wav)
        
        return wav, 48000, cap

def collate_audio(batch):
    wavs, srs, caps = [], [], []
    for w, sr, c in batch:
        wavs.append(w); srs.append(sr); caps.append(c)
    return wavs, srs[0], caps


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

        # -------- Frozen Stable Diffusion --------
        self.sd_pipe = None
        self.sd_tok = None
        self.sd_text_encoder = None
        self.sd_hidden = 768  # Will be updated
        
        if load_sd:
            print("Loading Stable Diffusion...")
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(cfg.SD_ID, torch_dtype=dtype)
            self.sd_pipe.to(device)
            
            # Freeze all SD components
            for comp in (self.sd_pipe.unet, self.sd_pipe.vae, self.sd_pipe.text_encoder):
                for p in comp.parameters(): 
                    p.requires_grad = False
            
            self.sd_tok = self.sd_pipe.tokenizer
            self.sd_text_encoder = self.sd_pipe.text_encoder
            self.sd_hidden = self.sd_pipe.text_encoder.config.hidden_size
        else:
            # For training: still load SD text encoder to get target embeddings
            print("Loading SD text encoder for training...")
            from transformers import CLIPTextModel, CLIPTokenizer
            self.sd_tok = CLIPTokenizer.from_pretrained(cfg.SD_ID, subfolder="tokenizer")
            self.sd_text_encoder = CLIPTextModel.from_pretrained(
                cfg.SD_ID, 
                subfolder="text_encoder"
            ).eval().to(device)
            
            # Freeze SD text encoder
            for p in self.sd_text_encoder.parameters():
                p.requires_grad = False
            
            self.sd_hidden = self.sd_text_encoder.config.hidden_size

        # -------- Frozen CLIP for evaluation --------
        print("Loading CLIP model for evaluation...")
        self.clip_model = CLIPModel.from_pretrained(cfg.CLIP_ID).eval().to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_ID)

        # -------- Get CLAP dims --------
        dummy_text = ["test"]
        dummy_audio = [torch.zeros(48000).numpy()]  # 1 second at 48kHz
        
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

    # --- Forward (Training with Multi-Task Loss) ---
    def forward(self, wavs, sr, caps):
        # Get target embeddings
        clap_text_emb = self.encode_text_clap(caps)  # CLAP text embeddings
        sd_text_emb = self.encode_text_sd(caps)      # SD text embeddings
        
        # Get audio embeddings
        audio_emb = self.encode_audio(wavs, sr)
        
        # Project audio to both spaces
        audio_to_clap, audio_to_sd = self.mapper(audio_emb)
        
        # Loss 1: CLAP alignment (InfoNCE)
        loss_clap = self.info_nce(audio_to_clap, clap_text_emb, self.cfg.temperature)
        
        # Loss 2: SD alignment (MSE in embedding space)
        loss_sd = F.mse_loss(audio_to_sd, sd_text_emb)
        
        # Combined multi-task loss
        total_loss = (
            self.cfg.clap_loss_weight * loss_clap + 
            self.cfg.sd_loss_weight * loss_sd
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
            prompt_embeds=enc, 
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance
        ).images[0]
        
        return img

    # --- CLIP Evaluation ---
    @torch.inference_mode()
    def evaluate_generation(self, wavs, sr, captions, num_samples=4):
        """
        Generate images from audio and evaluate with CLIP.
        
        Args:
            wavs: List of audio waveforms
            sr: Sampling rate
            captions: List of text captions
            num_samples: Number of samples to evaluate
            
        Returns:
            avg_clip_score: Average CLIP similarity score
            generated_images: List of generated PIL images
            clip_scores: List of individual CLIP scores
        """
        if self.sd_pipe is None:
            raise RuntimeError("Stable Diffusion not loaded. Init with load_sd=True.")
        
        generated_images = []
        clip_scores = []
        
        # Generate images for subset
        for i in range(min(num_samples, len(wavs))):
            # Generate image
            img = self.generate(wavs[i], sr)
            generated_images.append(img)
            
            # Compute CLIP similarity
            inputs = self.clip_processor(
                text=[captions[i]], 
                images=[img],
                return_tensors="pt", 
                padding=True
            ).to(self.cfg.device)
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_score = logits_per_image.item()
            clip_scores.append(clip_score)
        
        avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
        
        return avg_clip_score, generated_images, clip_scores


# ========================
#  Training
# ========================
def train(cfg: Config):
    # Load full dataset
    full_ds = AudioCaptionDataset(
        captions_path=cfg.train_csv,
        audio_folder=cfg.audio_folder,
        use_zip_files=cfg.use_zip_files
    )
    
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
    
    model = Audio2ImageModel(cfg, load_sd=False).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    # Track best model
    best_clip_score = -float('inf')

    print(f"\n{'='*60}")
    print(f"Starting Multi-Task Training")
    print(f"{'='*60}")
    print(f"Dataset: {len(full_ds)} samples ({len(train_ds)} train, {len(val_ds)} val)")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.max_epochs}")
    print(f"CLAP loss weight: {cfg.clap_loss_weight}")
    print(f"SD loss weight: {cfg.sd_loss_weight}")
    print(f"Evaluation every: {cfg.eval_every_n_epochs} epochs")
    print(f"{'='*60}\n")
    
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{cfg.max_epochs}")
        
        epoch_loss = 0
        epoch_clap_loss = 0
        epoch_sd_loss = 0
        epoch_clap_sim = 0
        epoch_sd_sim = 0
        
        for wavs, sr, caps in pbar:
            wavs = [w.to(cfg.device) for w in wavs]
            
            loss, stats = model(wavs, sr, caps)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            epoch_clap_loss += stats['loss_clap']
            epoch_sd_loss += stats['loss_sd']
            epoch_clap_sim += stats['clap_sim']
            epoch_sd_sim += stats['sd_sim']
            
            pbar.set_postfix({
                "total": f"{loss.item():.3f}",
                "clap": f"{stats['loss_clap']:.3f}",
                "sd": f"{stats['loss_sd']:.3f}",
                "c_sim": f"{stats['clap_sim']:.2f}",
                "s_sim": f"{stats['sd_sim']:.2f}"
            })
        
        # Compute epoch averages
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_clap_loss = epoch_clap_loss / n
        avg_sd_loss = epoch_sd_loss / n
        avg_clap_sim = epoch_clap_sim / n
        avg_sd_sim = epoch_sd_sim / n
        
        print(f"\n{'='*60}")
        print(f"Epoch {ep} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  CLAP Loss: {avg_clap_loss:.4f} | CLAP Sim: {avg_clap_sim:.3f}")
        print(f"  SD Loss: {avg_sd_loss:.4f} | SD Sim: {avg_sd_sim:.3f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint after every epoch
        checkpoint = {
            "mapper": model.mapper.state_dict(),
            "epoch": ep,
            "loss": avg_loss,
            "clap_loss": avg_clap_loss,
            "sd_loss": avg_sd_loss,
            "clap_sim": avg_clap_sim,
            "sd_sim": avg_sd_sim,
            "config": {
                "clap_loss_weight": cfg.clap_loss_weight,
                "sd_loss_weight": cfg.sd_loss_weight,
            }
        }
        
        torch.save(checkpoint, cfg.ckpt_path)
        print(f"Checkpoint saved to {cfg.ckpt_path}")
        
        # Run evaluation if it's time
        if ep % cfg.eval_every_n_epochs == 0:
            print(f"\n{'='*60}")
            print(f"🔍 Running CLIP Evaluation at Epoch {ep}")
            print(f"{'='*60}")
            
            # Create evaluation model with SD loaded
            eval_model = Audio2ImageModel(cfg, load_sd=True).to(cfg.device)
            eval_model.mapper.load_state_dict(checkpoint["mapper"])
            eval_model.eval()
            
            # Evaluate on validation set (limit to save time)
            val_clip_scores = []
            all_gen_images = []
            all_captions = []
            
            eval_batches = min(3, len(val_loader))  # Max 3 batches
            
            print(f"Evaluating on validation set ({eval_batches} batches)...")
            for batch_idx, (wavs, sr, caps) in enumerate(val_loader):
                if batch_idx >= eval_batches:
                    break
                
                wavs = [w.to(cfg.device) for w in wavs]
                
                # Generate images and compute CLIP scores
                avg_score, gen_imgs, scores = eval_model.evaluate_generation(
                    wavs, sr, caps,
                    num_samples=min(cfg.num_eval_samples, len(wavs))
                )
                
                val_clip_scores.extend(scores)
                all_gen_images.extend(gen_imgs)
                all_captions.extend(caps[:len(gen_imgs)])
                
                print(f"  Batch {batch_idx + 1}/{eval_batches}: Avg CLIP = {avg_score:.3f}")
            
            # Compute overall validation CLIP score
            avg_val_clip = sum(val_clip_scores) / len(val_clip_scores) if val_clip_scores else 0.0
            
            print(f"\nCLIP Evaluation Results:")
            print(f"  Average CLIP Score: {avg_val_clip:.4f}")
            print(f"  Evaluated {len(val_clip_scores)} samples from validation set")
            
            # Save example images from evaluation
            if cfg.save_eval_images and all_gen_images:
                os.makedirs("eval_images", exist_ok=True)
                for i in range(min(4, len(all_gen_images))):
                    img = all_gen_images[i]
                    cap = all_captions[i]
                    score = val_clip_scores[i]
                    img_path = f"eval_images/epoch{ep}_sample{i+1}_score{score:.2f}.png"
                    img.save(img_path)
                    print(f"    Sample {i+1}: '{cap[:50]}...' | CLIP: {score:.3f}")
                    print(f"      Saved to: {img_path}")
            
            # Update best model if improved
            if avg_val_clip > best_clip_score:
                best_clip_score = avg_val_clip
                best_ckpt_path = cfg.ckpt_path.replace('.pt', '_best.pt')
                checkpoint['best_clip_score'] = best_clip_score
                torch.save(checkpoint, best_ckpt_path)
                print(f"\n🎯 New best model! CLIP Score: {best_clip_score:.4f}")
                print(f"   Saved to: {best_ckpt_path}")
            else:
                print(f"\n   Best CLIP Score so far: {best_clip_score:.4f}")
            
            print(f"{'='*60}\n")
            
            # Clean up evaluation model
            del eval_model
            if cfg.device == "cuda":
                torch.cuda.empty_cache()
            elif cfg.device == "mps":
                torch.mps.empty_cache()
    
    print("Training completed!")
    print(f"Best CLIP Score: {best_clip_score:.4f}")


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
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  CLAP Sim: {ckpt.get('clap_sim', 'N/A'):.3f}")
    print(f"  SD Sim: {ckpt.get('sd_sim', 'N/A'):.3f}")
    
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