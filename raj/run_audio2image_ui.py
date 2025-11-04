#!/usr/bin/env python3
"""
Audio2Image Web Interface Launcher
Run this script to start the web interface for your audio-to-image model.
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'main2.py',
        'audio2image_ui.html', 
        'app_audio2image.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def main():
    print("🎵 Audio2Image Neural Synthesis Web Interface 🖼️")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not check_requirements():
        print("\n❌ Please make sure you're running this from the correct directory")
        print("   and all required files are present.")
        sys.exit(1)
    
    # Check for model checkpoint
    checkpoint_path = "audio2image_mapper_dual.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Warning: Model checkpoint not found at {checkpoint_path}")
        print("   The interface will start but you need to train the model first.")
        print("   Run: python main2.py --mode train")
        print()
    
    print("✅ Starting web interface...")
    print("📱 Open your browser and go to: http://localhost:5010")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 55)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'app_audio2image.py'])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main()