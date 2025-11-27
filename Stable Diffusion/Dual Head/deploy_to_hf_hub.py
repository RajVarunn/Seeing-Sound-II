#!/usr/bin/env python3
import os
import shutil

def create_hf_hub_deployment():
    # Create directories
    model_dir = "hf_model_hub"
    space_dir = "hf_space"
    
    for dir_name in [model_dir, space_dir]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    
    # Model Hub README
    model_readme = '''---
license: mit
tags:
- audio-to-image
- stable-diffusion
---

# Audio2Image Model

Generates images from audio using neural synthesis.
'''
    
    # Space app.py
    space_app = '''import gradio as gr
import torch
import torchaudio
from huggingface_hub import hf_hub_download

def load_model():
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/audio2image-model", 
        filename="audio2image_mapper_dual_best.pt"
    )
    
    from main2 import Audio2ImageModel, Config
    config = Config()
    config.ckpt_path = model_path
    model = Audio2ImageModel(config, load_sd=True)
    
    ckpt = torch.load(model_path, map_location=config.device)
    model.mapper.load_state_dict(ckpt["mapper"])
    
    return model, config

model, config = load_model()

def generate_image(audio_file):
    if audio_file is None:
        return "Upload an audio file"
    
    wav, sr = torchaudio.load(audio_file)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0).float()
    
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        wav = resampler(wav)
    
    wav = wav.to(config.device)
    
    with torch.no_grad():
        image = model.generate(wav, 48000)
    
    return image

gr.Interface(
    fn=generate_image,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Image(label="Generated Image"),
    title="🎵 Audio → Image Generator"
).launch()
'''
    
    # Space requirements
    space_requirements = '''torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.20.0
diffusers>=0.10.0
gradio>=3.0.0
huggingface_hub>=0.16.0
'''
    
    # Write files
    with open(f"{model_dir}/README.md", "w") as f:
        f.write(model_readme)
    
    with open(f"{space_dir}/app.py", "w") as f:
        f.write(space_app)
    
    with open(f"{space_dir}/requirements.txt", "w") as f:
        f.write(space_requirements)
    
    # Copy model files
    if os.path.exists("audio2image_mapper_dual_best.pt"):
        shutil.copy("audio2image_mapper_dual_best.pt", model_dir)
        print("✓ Copied model to hub folder")
    
    if os.path.exists("main2.py"):
        shutil.copy("main2.py", model_dir)
        shutil.copy("main2.py", space_dir)
        print("✓ Copied main2.py")
    
    print("\n✅ Files created!")
    print("1. Upload 'hf_model_hub/' to HF Model Hub")
    print("2. Upload 'hf_space/' to HF Spaces")
    print("3. Update YOUR_USERNAME in app.py")

if __name__ == "__main__":
    create_hf_hub_deployment()