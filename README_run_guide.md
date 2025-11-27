# Seeing Sound II: Project Usage Guide

## How to Run the Project

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Seeing-Sound-II
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
- Ensure `main_dataV1.csv` is present in a accessible location in your computer.
- Place your audio and image files in the correct subfolders as referenced in the CSV.
- The VGG data files are acessible in this location: https://drive.google.com/drive/folders/1rGY-ddTV85Alp3vol1xfeEz63A7Alxqx?usp=sharing 

### 4. Train the Model
To train the dual-head MLP + UNet model:
```bash
python3 main2.py --mode train
```
- For single-head MLP, use `main.py`.
- For MLP only (no UNet), use `mlponly.py`.
- For fusion/dream mode, use `main_dream_mode.py`.
- The model files are accessible in this location: https://drive.google.com/drive/folders/1rGY-ddTV85Alp3vol1xfeEz63A7Alxqx?usp=sharing

### 5. Inference (Generate Images from Audio)
```bash
python3 main2.py --mode infer --wav <path_to_audio.wav> --out <output_image.png>
```

### 6. Run the Local Web App
```bash
python3 app_audio2image.py
```
- This launches a local interface for interactive audio-to-image generation.

### 7. Hugging Face Deployment
- The `hf_space` folder contains files for deploying to Hugging Face Spaces.
- To deploy, push the contents of `hf_space` to your Hugging Face Space repository.
- Use `deploy_to_hf_hub.py` to automate model uploads to the Hugging Face Model Hub.

### 8. Notebooks
- `Seeing_Sound_II_MLP_ONLY.ipynb` and `Seeing_Sound_II_MLP_UNET.ipynb` provide step-by-step training and experimentation in Jupyter format.
