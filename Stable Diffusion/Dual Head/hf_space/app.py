import gradio as gr
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download

def load_model():
    model_path = hf_hub_download(
        repo_id="Suyamprakasam/audio2image-model", 
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
    
    # Use soundfile to avoid torchcodec requirement
    wav, sr = sf.read(audio_file)
    wav = torch.from_numpy(wav).float()
    if len(wav.shape) > 1:
        wav = wav.T  # soundfile returns (samples, channels), we need (channels, samples)
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
