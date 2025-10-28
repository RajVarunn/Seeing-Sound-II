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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoProcessor, ClapModel, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image


# ========================
#  Configuration
# ========================
@dataclass
class Config:
    CLAP_ID: str = "laion/clap-htsat-fused"
    SD_ID: str   = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    clap_loss_weight: float = 1.0
    sd_loss_weight: float = 0.5
    batch_size: int = 4
    max_epochs: int = 20
    base_prompt: str = "A photo of"
    guidance: float = 7.5
    steps: int = 30
    train_csv: str = "/Users/rajvarun/Desktop/SIT/Trimester 4/AAI 3001 - Computer Vision & Deep Learning/Seeing Sound II/extracted_audiocaps/captions.txt"
    ckpt_path: str = "audio2image_mapper.pt"


# ========================
#  Dataset
# ========================
class AudioCaptionDataset(Dataset):
    """Reads a tab-separated file of (audio_path, caption)."""
    def __init__(self, captions_path: str):
        self.items = []
        base_dir = os.path.dirname(captions_path)
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        filename = parts[0]
                        caption = parts[1]
                        audio_path = os.path.join(base_dir, filename)
                        if os.path.exists(audio_path):
                            self.items.append((audio_path, caption))
        if not self.items:
            raise ValueError("Empty dataset: no valid audio files found")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, cap = self.items[idx]
        wav, sr = torchaudio.load(path)
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


# ========================
#  Training
# ========================
def train(cfg: Config):
    ds = AudioCaptionDataset(cfg.train_csv)
    loader = DataLoader(
        ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=collate_audio,
        num_workers=0,  # Set to 0 for debugging
        drop_last=True  # Drop last incomplete batch to avoid batch_size=1
    )
    
    model = Audio2ImageModel(cfg, load_sd=False).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    print(f"\n{'='*60}")
    print(f"Starting Multi-Task Training")
    print(f"{'='*60}")
    print(f"Dataset: {len(ds)} samples")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.max_epochs}")
    print(f"CLAP loss weight: {cfg.clap_loss_weight}")
    print(f"SD loss weight: {cfg.sd_loss_weight}")
    print(f"{'='*60}\n")
    
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {ep}/{cfg.max_epochs}")
        
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
        n = len(loader)
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
        
        # Save checkpoint
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
        print(f"Checkpoint saved to {cfg.ckpt_path}\n")
    
    print("Training completed!")


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