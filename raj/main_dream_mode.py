"""
Audio Fusion Dream Mode - Combine Two Audio Inputs into One Image
Key features:
    - Takes TWO audio files as input
    - Combines their embeddings in creative ways
    - Generates a fusion image representing both sounds
    - Uses pre-trained UNET model from main2.py
"""

# ========================
#  Imports
# ========================
import os, sys
from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import AutoProcessor, ClapModel
from diffusers import StableDiffusionPipeline
from PIL import Image

# Import the trained model architecture from main2
from main2 import Audio2ImageModel, Config as BaseConfig


# ========================
#  Configuration
# ========================
@dataclass
class DreamConfig(BaseConfig):
    """Extended config for dream mode"""
    # Fusion modes
    fusion_mode: str = "blend"  # Options: "blend", "creative"
    blend_ratio: float = 0.5  # 0.0 = audio1 only, 1.0 = audio2 only, 0.5 = equal blend
    
    # Creative prompt enhancement
    use_creative_prompt: bool = True
    dream_prompt_template: str = "A surreal fusion of {concept1} and {concept2}, dreamlike, artistic"
    
    # Model checkpoint (use UNET trained model)
    ckpt_path: str = "audio2image_mapper_dual_best.pt"


# ========================
#  Dream Model
# ========================
class DreamFusionModel(Audio2ImageModel):
    """
    Extended model that can fuse two audio inputs into one image.
    Inherits from Audio2ImageModel but adds fusion capabilities.
    """
    
    def __init__(self, cfg: DreamConfig, load_sd: bool = True):
        super().__init__(cfg, load_sd=load_sd)
        self.dream_cfg = cfg
    
    def fuse_embeddings(self, emb1: torch.Tensor, emb2: torch.Tensor, mode: str = "blend", ratio: float = 0.5):
        """
        Fuse two audio embeddings using different strategies.
        
        Args:
            emb1: First audio embedding [1, dim]
            emb2: Second audio embedding [1, dim]
            mode: Fusion strategy ('blend' or 'creative')
            ratio: Blend ratio (0.0 = audio1 only, 1.0 = audio2 only)
        
        Returns:
            Fused embedding [1, dim]
        """
        if mode == "blend":
            # Simple linear interpolation - clean fusion
            return (1 - ratio) * emb1 + ratio * emb2
        
        elif mode == "creative":
            # Creative fusion: blend + slight noise for more interesting/dreamlike results
            blend = (1 - ratio) * emb1 + ratio * emb2
            # Add small amount of noise for variation
            noise = torch.randn_like(blend) * 0.05  # Reduced from 0.1 for stability
            return blend + noise
        
        else:
            # Default to blend
            return (1 - ratio) * emb1 + ratio * emb2
    
    @torch.inference_mode()
    def generate_fusion(self, wav1, sr1, wav2, sr2, concept1: str = None, concept2: str = None):
        """
        Generate a fusion image from two audio inputs by injecting BOTH embeddings separately.
        
        Args:
            wav1: First audio waveform
            sr1: Sample rate of first audio
            wav2: Second audio waveform
            sr2: Sample rate of second audio
            concept1: Optional text description of first audio (for display only)
            concept2: Optional text description of second audio (for display only)
        
        Returns:
            PIL Image representing the fusion
        """
        if self.sd_pipe is None:
            raise RuntimeError("Stable Diffusion not loaded. Init with load_sd=True.")
        
        # Get embeddings for both audio inputs
        audio_emb1 = self.encode_audio([wav1], sr1)
        audio_emb2 = self.encode_audio([wav2], sr2)
        
        # Project both to SD space - keep them SEPARATE!
        _, soft_token1 = self.mapper(audio_emb1)
        _, soft_token2 = self.mapper(audio_emb2)
        
        # Keep BOTH embeddings at full strength - no ratio blending!
        # We'll inject them separately with "and" between them.
        
        # Prompt structure: "A fusion image of [AUDIO1] and [AUDIO2]"
        prompt = "A fusion image of"
        
        # Tokenize prompt
        tok = self.sd_tok(
            prompt,
            padding="max_length",
            max_length=self.sd_tok.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.cfg.device)
        
        # Get SD text embeddings for the base prompt
        enc = self.sd_text_encoder(tok["input_ids"])[0]
        
        # Tokenize "and" to get its embedding
        and_tok = self.sd_tok(
            "and",
            return_tensors="pt"
        ).to(self.cfg.device)
        and_enc = self.sd_text_encoder(and_tok["input_ids"])
        and_emb = and_enc[0][0, 1, :]  # Get "and" token embedding (batch 0, token 1)
        
        # Find position after "A fusion image of"
        attention_mask = tok["attention_mask"][0]
        last_token_pos = attention_mask.nonzero(as_tuple=False).max().item()
        
        # Insert: [audio1] "and" [audio2]
        # This creates: "A fusion image of [AUDIO1_EMB] and [AUDIO2_EMB]"
        if last_token_pos + 3 < enc.shape[1]:
            # Enough space for both audio embeddings + "and"
            enc[0, last_token_pos + 1:last_token_pos + 2, :] = soft_token1  # Insert audio 1
            enc[0, last_token_pos + 2, :] = and_emb                          # Insert "and"
            enc[0, last_token_pos + 3:last_token_pos + 4, :] = soft_token2  # Insert audio 2
        elif last_token_pos + 2 < enc.shape[1]:
            # Not enough space for "and", just put both audios
            enc[0, last_token_pos + 1:last_token_pos + 2, :] = soft_token1
            enc[0, last_token_pos + 2:last_token_pos + 3, :] = soft_token2
        else:
            # Very limited space, average them as fallback
            fused = (soft_token1 + soft_token2) / 2
            enc[0, last_token_pos + 1:last_token_pos + 2, :] = fused
        
        # Generate fusion image
        print(f"🎨 Generating fusion image:")
        if concept1 and concept2:
            print(f"   Audio 1: '{concept1}'")
            print(f"   Audio 2: '{concept2}'")
        print(f"   Prompt: 'A fusion image of [AUDIO1] and [AUDIO2]'")
        print(f"   Both embeddings injected separately at full strength!")
        
        img = self.sd_pipe(
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance,
            prompt_embeds=enc
        ).images[0]
        
        return img


# ========================
#  Inference Functions
# ========================
def load_dream_model(cfg: DreamConfig):
    """Load the dream fusion model with trained weights"""
    print("Loading Dream Fusion Model...")
    model = DreamFusionModel(cfg, load_sd=True).to(cfg.device)
    
    # Load trained weights
    print(f"Loading checkpoint from {cfg.ckpt_path}...")
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.mapper.load_state_dict(ckpt["mapper"])
    
    # Load UNet weights if available
    if "unet" in ckpt:
        print("Loading fine-tuned UNet weights...")
        model.sd_unet.load_state_dict(ckpt["unet"])
    
    print(f"✓ Model loaded (Epoch {ckpt.get('epoch', '?')})")
    
    return model


def infer_fusion(
    cfg: DreamConfig,
    wav_path1: str,
    wav_path2: str,
    out_path: str,
    concept1: str = None,
    concept2: str = None,
    fusion_mode: str = "blend",
    blend_ratio: float = 0.5
):
    """
    Generate a fusion image from two audio files.
    
    Args:
        cfg: Dream configuration
        wav_path1: Path to first audio file
        wav_path2: Path to second audio file
        out_path: Output image path
        concept1: Description of first audio (optional)
        concept2: Description of second audio (optional)
        fusion_mode: How to fuse embeddings (blend/concat/max/creative)
        blend_ratio: Blend ratio for 'blend' mode (0.0 to 1.0)
    """
    # Update config
    cfg.fusion_mode = fusion_mode
    cfg.blend_ratio = blend_ratio
    
    # Load first audio
    print(f"\n📻 Loading audio 1: {wav_path1}")
    wav1, sr1 = torchaudio.load(wav_path1)
    if wav1.size(0) > 1:
        wav1 = wav1.mean(0, keepdim=True)
    wav1 = wav1.squeeze(0).float()
    
    if sr1 != 48000:
        print(f"   Resampling from {sr1}Hz to 48000Hz...")
        wav1 = torchaudio.transforms.Resample(sr1, 48000)(wav1)
        sr1 = 48000
    
    wav1 = wav1.to(cfg.device)
    
    # Load second audio
    print(f"📻 Loading audio 2: {wav_path2}")
    wav2, sr2 = torchaudio.load(wav_path2)
    if wav2.size(0) > 1:
        wav2 = wav2.mean(0, keepdim=True)
    wav2 = wav2.squeeze(0).float()
    
    if sr2 != 48000:
        print(f"   Resampling from {sr2}Hz to 48000Hz...")
        wav2 = torchaudio.transforms.Resample(sr2, 48000)(wav2)
        sr2 = 48000
    
    wav2 = wav2.to(cfg.device)
    
    # Load model
    model = load_dream_model(cfg)
    
    # Generate fusion image
    print("\n🌀 Generating fusion image...")
    img = model.generate_fusion(wav1, sr1, wav2, sr2, concept1, concept2)
    
    # Save
    img.save(out_path)
    print(f"✅ Fusion image saved to {out_path}")
    print(f"   Mode: {fusion_mode}, Ratio: {blend_ratio:.2f}")


# ========================
#  Main (CLI Interface)
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Fusion Dream Mode")
    parser.add_argument("--audio1", required=True, help="First audio file")
    parser.add_argument("--audio2", required=True, help="Second audio file")
    parser.add_argument("--out", default="fusion_output.png", help="Output image path")
    parser.add_argument("--concept1", help="Description of first audio (e.g., 'dog barking')")
    parser.add_argument("--concept2", help="Description of second audio (e.g., 'piano music')")
    parser.add_argument("--mode", default="blend", choices=["blend", "creative"],
                       help="Fusion mode: blend (clean mix) or creative (dreamlike)")
    parser.add_argument("--ratio", type=float, default=0.5,
                       help="Blend ratio (0.0 = audio1 only, 1.0 = audio2 only)")
    parser.add_argument("--checkpoint", default="audio2image_mapper_dual_best.pt",
                       help="Model checkpoint path")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio1):
        print(f"❌ Audio file 1 not found: {args.audio1}")
        sys.exit(1)
    if not os.path.exists(args.audio2):
        print(f"❌ Audio file 2 not found: {args.audio2}")
        sys.exit(1)
    
    # Create config
    cfg = DreamConfig()
    cfg.ckpt_path = args.checkpoint
    
    print("="*70)
    print("🌙 DREAM FUSION MODE")
    print("="*70)
    print(f"Device: {cfg.device}")
    print(f"Audio 1: {args.audio1}")
    print(f"Audio 2: {args.audio2}")
    print(f"Fusion Mode: {args.mode}")
    print(f"Blend Ratio: {args.ratio}")
    print("="*70)
    
    # Run fusion
    infer_fusion(
        cfg,
        args.audio1,
        args.audio2,
        args.out,
        args.concept1,
        args.concept2,
        args.mode,
        args.ratio
    )
    
    print("\n🎉 Done! Try different fusion modes and ratios for varied results!")
