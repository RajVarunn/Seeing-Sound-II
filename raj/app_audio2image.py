from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import tempfile
import base64
from io import BytesIO
import torch
import torchaudio
from PIL import Image

app = Flask(__name__)

# Global model instances - support for multiple models
models = {
    'unet': {
        'model': None,
        'config': None,
        'checkpoint': 'audio2image_mapper_dual_best.pt',
        'module': 'main2',
        'description': '2HEAD + Larger Dataset + UNET (Best Quality)'
    },
    'no_unet': {
        'model': None,
        'config': None,
        'checkpoint': 'audio2image_mapper_dual.pt',
        'module': 'main2_backup',
        'description': '2HEAD + Different Dataset + No UNET (Faster)'
    }
}

current_model = 'unet'  # Default model

def load_model(model_type='unet'):
    """Load the trained audio2image model"""
    global current_model
    
    if model_type not in models:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'unet' or 'no_unet'")
    
    model_info = models[model_type]
    
    # Import the correct module
    if model_info['module'] == 'main2':
        from main2 import Audio2ImageModel, Config
    else:
        from main2_backup import Audio2ImageModel, Config
    
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
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return jsonify({
                'success': True,
                'image_url': f'data:image/png;base64,{img_str}',
                'model_used': models[model_type]['description']
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise e
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
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
    print("Loading default model (UNet)...")
    
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
    print("  Switch: http://localhost:5010/switch_model/<unet|no_unet>")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5010)