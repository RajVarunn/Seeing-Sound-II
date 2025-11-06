from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import tempfile
import base64
from io import BytesIO
import torch
import torchaudio
from PIL import Image

app = Flask(__name__)

# Load AudioCaps captions mapping
CAPTIONS_FILE = '../extracted_audiocaps/captions.txt'
audio_captions = {}

def load_captions():
    """Load captions from AudioCaps captions.txt file"""
    global audio_captions
    
    if not os.path.exists(CAPTIONS_FILE):
        print(f"Warning: Captions file not found at {CAPTIONS_FILE}")
        return
    
    try:
        with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab (format: filename\tcaption)
                parts = line.split('\t')
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    caption = parts[1].strip()
                    audio_captions[filename] = caption
        
        print(f"✓ Loaded {len(audio_captions)} audio captions from AudioCaps")
    except Exception as e:
        print(f"Error loading captions: {e}")

def get_caption_for_audio(filename):
    """Get caption for an audio file by filename"""
    # Try exact match first
    if filename in audio_captions:
        return audio_captions[filename]
    
    # Try without extension
    base_name = os.path.splitext(filename)[0]
    for key in audio_captions.keys():
        if os.path.splitext(key)[0] == base_name:
            return audio_captions[key]
    
    return None

# Global model instances - support for multiple models
models = {
    'unet': {
        'model': None,
        'config': None,
        'checkpoint': 'audio2image_mapper_dual_best.pt',
        'module': 'main2',
        'description': 'MLP + UNET'
    },
    'mlponly': {
        'model': None,
        'config': None,
        'checkpoint': 'audio2image_mapper_dual_mlp_only_best.pt',
        'module': 'mlponly',
        'description': 'MLP Only'
    },
    'dream': {
        'model': None,
        'config': None,
        'checkpoint': 'audio2image_mapper_dual_best.pt',
        'module': 'main_dream_mode',
        'description': 'Dream Fusion - Combine Two Audios'
    }
}

current_model = 'unet'  # Default model

def load_model(model_type='unet'):
    """Load the trained audio2image model"""
    global current_model
    
    if model_type not in models:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'unet' or 'mlponly'")
    
    model_info = models[model_type]
    
    # Import the correct module
    if model_info['module'] == 'main2':
        from main2 import Audio2ImageModel, Config
    elif model_info['module'] == 'mlponly':
        from mlponly import Audio2ImageModel, Config
    elif model_info['module'] == 'main_dream_mode':
        from main_dream_mode import DreamFusionModel as Audio2ImageModel, DreamConfig as Config
    else:
        raise ValueError(f"Unknown module: {model_info['module']}")
    
    config = Config()
    config.ckpt_path = model_info['checkpoint']
    print(f"Loading {model_info['description']}")
    print(f"Using checkpoint: {config.ckpt_path}")
    print(f"Device: {config.device}")
    
    # Load model with Stable Diffusion
    model = Audio2ImageModel(config, load_sd=True).to(config.device)
    
    # Load trained weights
    if os.path.exists(config.ckpt_path):
        print(f"Loading checkpoint from {config.ckpt_path}")
        ckpt = torch.load(config.ckpt_path, map_location=config.device)
        model.mapper.load_state_dict(ckpt["mapper"])
        print(f"Model loaded successfully!")
        print(f"Checkpoint info - Epoch: {ckpt.get('epoch', 'unknown')}")
    else:
        print(f"Warning: No checkpoint found at {config.ckpt_path}")
        print("Model will use random weights")
    
    # Store in global dict
    model_info['model'] = model
    model_info['config'] = config
    current_model = model_type
    
    return model, config

@app.route('/')
def index():
    """Serve the main UI"""
    with open('audio2image_ui.html', 'r') as f:
        return f.read()

@app.route('/generate_image', methods=['POST'])
def generate_image():
    """Generate image from uploaded audio"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get model type from form data (default to current)
        model_type = request.form.get('model_type', current_model)
        
        # Load model if not already loaded
        if models[model_type]['model'] is None:
            print(f"Loading {model_type} model...")
            load_model(model_type)
        
        model = models[model_type]['model']
        config = models[model_type]['config']
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_audio_path = tmp_file.name
        
        try:
            # Load and preprocess audio
            print(f"Processing audio file: {audio_file.filename} with {model_type} model")
            wav, sr = torchaudio.load(temp_audio_path)
            
            # Convert to mono if stereo
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).float()
            
            # Resample to 48kHz for CLAP
            if sr != 48000:
                print(f"Resampling from {sr}Hz to 48000Hz")
                resampler = torchaudio.transforms.Resample(sr, 48000)
                wav = resampler(wav)
                sr = 48000
            
            wav = wav.to(config.device)
            
            # Generate image
            print("Generating image...")
            with torch.no_grad():
                generated_image = model.generate(wav, sr)
            
            # Convert PIL image to base64
            buffered = BytesIO()
            generated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Get caption for this audio file
            caption = get_caption_for_audio(audio_file.filename)
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return jsonify({
                'success': True,
                'image_url': f'data:image/png;base64,{img_str}',
                'model_used': models[model_type]['description'],
                'caption': caption,
                'filename': audio_file.filename
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise e
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_fusion', methods=['POST'])
def generate_fusion():
    """Generate fusion image from TWO audio files (Dream Mode)"""
    try:
        if 'audio1' not in request.files or 'audio2' not in request.files:
            return jsonify({'error': 'Two audio files required for fusion mode'}), 400
        
        audio_file1 = request.files['audio1']
        audio_file2 = request.files['audio2']
        
        if audio_file1.filename == '' or audio_file2.filename == '':
            return jsonify({'error': 'Both audio files must be selected'}), 400
        
        fusion_mode = request.form.get('fusion_mode', 'blend')
        blend_ratio = float(request.form.get('blend_ratio', 0.5))
        
        # Load dream model if not already loaded
        if models['dream']['model'] is None:
            print("Loading dream model...")
            load_model('dream')
        
        model = models['dream']['model']
        config = models['dream']['config']
        
        # Save both audio files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp1:
            audio_file1.save(tmp1.name)
            temp_path1 = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
            audio_file2.save(tmp2.name)
            temp_path2 = tmp2.name
        
        try:
            # Load and process both audios
            print(f"Processing audio 1: {audio_file1.filename}")
            wav1, sr1 = torchaudio.load(temp_path1)
            if wav1.size(0) > 1:
                wav1 = wav1.mean(dim=0, keepdim=True)
            wav1 = wav1.squeeze(0).float()
            
            if sr1 != 48000:
                print(f"Resampling audio 1 from {sr1}Hz to 48000Hz")
                resampler = torchaudio.transforms.Resample(sr1, 48000)
                wav1 = resampler(wav1)
                sr1 = 48000
            
            wav1 = wav1.to(config.device)
            
            print(f"Processing audio 2: {audio_file2.filename}")
            wav2, sr2 = torchaudio.load(temp_path2)
            if wav2.size(0) > 1:
                wav2 = wav2.mean(dim=0, keepdim=True)
            wav2 = wav2.squeeze(0).float()
            
            if sr2 != 48000:
                print(f"Resampling audio 2 from {sr2}Hz to 48000Hz")
                resampler = torchaudio.transforms.Resample(sr2, 48000)
                wav2 = resampler(wav2)
                sr2 = 48000
            
            wav2 = wav2.to(config.device)
            
            # Get captions for both audios
            caption1 = get_caption_for_audio(audio_file1.filename)
            caption2 = get_caption_for_audio(audio_file2.filename)
            
            # Update fusion config
            config.fusion_mode = fusion_mode
            config.blend_ratio = blend_ratio
            
            # Generate fusion image
            print(f"Generating fusion image (mode: {fusion_mode}, ratio: {blend_ratio})...")
            with torch.no_grad():
                generated_image = model.generate_fusion(
                    wav1, sr1, wav2, sr2, 
                    caption1, caption2
                )
            
            # Convert PIL image to base64
            buffered = BytesIO()
            generated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up temp files
            os.unlink(temp_path1)
            os.unlink(temp_path2)
            
            # Create fusion caption
            fusion_caption = f"Fusion: {caption1 or audio_file1.filename} + {caption2 or audio_file2.filename}"
            
            return jsonify({
                'success': True,
                'image_url': f'data:image/png;base64,{img_str}',
                'model_used': models['dream']['description'],
                'caption': fusion_caption,
                'caption1': caption1 or audio_file1.filename,
                'caption2': caption2 or audio_file2.filename,
                'fusion_mode': fusion_mode,
                'blend_ratio': blend_ratio,
                'filename': f"fusion_{audio_file1.filename}_{audio_file2.filename}"
            })
            
        except Exception as e:
            # Clean up temp files on error
            if os.path.exists(temp_path1):
                os.unlink(temp_path1)
            if os.path.exists(temp_path2):
                os.unlink(temp_path2)
            raise e
            
    except Exception as e:
        print(f"Error generating fusion: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            model_type: {
                'loaded': info['model'] is not None,
                'description': info['description'],
                'checkpoint': info['checkpoint']
            }
            for model_type, info in models.items()
        },
        'current_model': current_model,
        'device': models[current_model]['config'].device if models[current_model]['config'] else 'unknown'
    })

@app.route('/switch_model/<model_type>')
def switch_model(model_type):
    """Switch between models"""
    global current_model
    
    if model_type not in models:
        return jsonify({'error': f'Invalid model type. Choose: {list(models.keys())}'}), 400
    
    try:
        if models[model_type]['model'] is None:
            print(f"Loading {model_type} model...")
            load_model(model_type)
        else:
            current_model = model_type
        
        return jsonify({
            'success': True,
            'current_model': current_model,
            'description': models[current_model]['description']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def list_models():
    """List available models"""
    return jsonify({
        'models': {
            model_type: {
                'description': info['description'],
                'checkpoint': info['checkpoint'],
                'loaded': info['model'] is not None
            }
            for model_type, info in models.items()
        },
        'current_model': current_model
    })

if __name__ == '__main__':
    print("Starting Audio2Image Web Interface...")
    
    # Load captions
    print("\nLoading AudioCaps captions...")
    load_captions()
    
    print("\nLoading default model (UNet)...")
    
    try:
        load_model('unet')
        print("✓ UNet model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading UNet model: {e}")
        print("The app will start but image generation may not work.")
    
    print("\n" + "="*50)
    print("🎵 AUDIO → IMAGE NEURAL SYNTHESIS 🖼️")
    print("="*50)
    print("\nAvailable Models:")
    for model_type, info in models.items():
        status = "✓ Loaded" if info['model'] is not None else "○ Not loaded"
        print(f"  {status} [{model_type}]: {info['description']}")
    print("\nEndpoints:")
    print("  Main UI: http://localhost:5010")
    print("  Health: http://localhost:5010/health")
    print("  Models: http://localhost:5010/models")
    print("  Switch: http://localhost:5010/switch_model/<unet|mlponly>")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5010)