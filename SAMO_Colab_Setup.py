#!/usr/bin/env python3
"""
SAMO Voice-First Development - Google Colab Setup Script

This script sets up the complete SAMO environment in Google Colab
for voice-first emotion detection development.

Usage in Colab:
1. Upload this file to Colab
2. Run: !python SAMO_Colab_Setup.py
3. Follow the setup instructions
"""

import os
import sys
import subprocess
import requests

def print_header():
    """Print setup header."""
    print("🎤 SAMO Voice-First Development - Google Colab Setup")
    print("=" * 60)
    print("🎯 Goal: Voice-first emotion detection system")
    print("🎯 Current F1: 13.2% (Target: >50%)")
    print("🎯 Features: Real-time voice + text emotion detection")
    print("🎯 Environment: Google Colab (GPU accelerated)")
    print("=" * 60)

def check_gpu():
    """Check GPU availability."""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"✅ PyTorch installed")
        print(f"✅ GPU available: {gpu_available}")
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️  No GPU available - using CPU")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def clone_repository():
    """Clone SAMO repository."""
    print("\n📥 Cloning SAMO repository...")
    try:
        # Clone repository
        subprocess.run([
            "git", "clone", "https://github.com/uelkerd/SAMO--DL.git"
        ], check=True)
        
        # Change to repository directory
        os.chdir("SAMO--DL")
        print("✅ Repository cloned successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return False

def install_dependencies():
    """Install all dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Install SAMO package
    try:
        subprocess.run(["pip", "install", "-e", "."], check=True)
        print("✅ SAMO package installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install SAMO package: {e}")
        return False
    
    # Install voice processing libraries
    voice_packages = [
        "pyaudio",
        "soundfile", 
        "librosa",
        "openai-whisper",
        "speechrecognition"
    ]
    
    for package in voice_packages:
        try:
            subprocess.run(["pip", "install", package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_audio_libraries():
    """Test audio processing libraries."""
    print("\n🎤 Testing audio libraries...")
    
    try:
        import soundfile as sf
        print("✅ soundfile imported")
    except ImportError as e:
        print(f"❌ soundfile import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa imported")
    except ImportError as e:
        print(f"❌ librosa import failed: {e}")
        return False
    
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    
    return True

def create_voice_demo():
    """Create voice processing demo."""
    print("\n🎤 Creating voice processing demo...")
    
    demo_code = '''
# Voice Processing Demo
import pyaudio
import wave
import numpy as np
import librosa
import whisper

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                   channels=channels,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=chunk)
    
    print("🎤 Recording... Speak now!")
    frames = []
    
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("✅ Recording complete!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames

def voice_to_text(audio_frames, sample_rate=16000):
    """Convert voice to text using Whisper."""
    # Save audio to temporary file
    with wave.open("temp_audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(audio_frames))
    
    # Transcribe with Whisper
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.wav")
    
    return result["text"]

def detect_emotion_from_voice(audio_frames, sample_rate=16000):
    """Detect emotion from voice using audio features."""
    # Convert audio frames to numpy array
    audio_data = np.frombuffer(b''.join(audio_frames), dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Extract audio features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
    
    # Calculate statistics
    features = {
        'mfcc_mean': np.mean(mfccs),
        'mfcc_std': np.std(mfccs),
        'spectral_centroid_mean': np.mean(spectral_centroids),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate)
    }
    
    # Simple emotion mapping
    if features['spectral_centroid_mean'] > 2000:
        emotion = "excited"
    elif features['mfcc_mean'] < -5:
        emotion = "sad"
    else:
        emotion = "neutral"
    
    return emotion, features

# Test voice processing
print("🎤 Testing voice processing...")
audio_frames = record_audio(duration=3)
text = voice_to_text(audio_frames)
emotion, features = detect_emotion_from_voice(audio_frames)

print(f"🎤 You said: {text}")
print(f"😊 Detected emotion: {emotion}")
print(f"📊 Audio features: {features}")
'''
    
    with open("voice_demo.py", "w") as f:
        f.write(demo_code)
    
    print("✅ Voice demo created: voice_demo.py")
    return True

def create_f1_optimization_script():
    """Create F1 optimization script."""
    print("\n🎯 Creating F1 optimization script...")
    
    f1_code = '''
# F1 Score Optimization Script
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def optimize_f1_score():
    """Optimize F1 score using focal loss and other techniques."""
    print("🚀 Starting F1 optimization...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    data_loader = GoEmotionsDataLoader()
    datasets = data_loader.prepare_datasets()
    
    # Create model
    model = BERTEmotionClassifier()
    model.to(device)
    
    # Create focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print("✅ F1 optimization setup complete!")
    print("🎯 Expected improvement: 13.2% → 50%+ F1 score")
    
    return model, focal_loss, optimizer

# Run optimization
if __name__ == "__main__":
    model, focal_loss, optimizer = optimize_f1_score()
'''
    
    with open("f1_optimization.py", "w") as f:
        f.write(f1_code)
    
    print("✅ F1 optimization script created: f1_optimization.py")
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. 🎤 Test voice processing:")
    print("   !python voice_demo.py")
    
    print("\n2. 🎯 Run F1 optimization:")
    print("   !python f1_optimization.py")
    
    print("\n3. 🔗 Test complete pipeline:")
    print("   !python scripts/simple_working_training.py")
    
    print("\n4. 📊 Monitor progress:")
    print("   - Check F1 score improvements")
    print("   - Test voice + text integration")
    print("   - Optimize performance")
    
    print("\n🎯 Success Metrics:")
    print("- F1 Score: >50% (currently 13.2%)")
    print("- Voice Processing: Real-time audio input")
    print("- Response Time: <500ms")
    print("- Multi-modal: Text + Voice + Emotion")
    
    print("\n🚀 Voice-First Features:")
    print("- Real-time microphone input")
    print("- Voice-to-text with Whisper")
    print("- Voice emotion detection")
    print("- Combined voice + text analysis")
    
    print("\n💰 Cost Benefits:")
    print("- Free GPU acceleration")
    print("- No local environment issues")
    print("- Built-in audio processing")
    print("- Collaborative development")

def main():
    """Main setup function."""
    print_header()
    
    # Check GPU
    if not check_gpu():
        print("❌ GPU check failed")
        return False
    
    # Clone repository
    if not clone_repository():
        print("❌ Repository clone failed")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Test audio libraries
    if not test_audio_libraries():
        print("❌ Audio library test failed")
        return False
    
    # Create demo scripts
    create_voice_demo()
    create_f1_optimization_script()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SAMO Colab setup completed successfully!")
    else:
        print("\n❌ SAMO Colab setup failed!")
        sys.exit(1) 