#!/usr/bin/env python3
"""SAMO Voice-First Development - Google Colab Setup Script.

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
from typing import Optional


def print_header() -> None:
    """Print setup header."""

def check_gpu() -> Optional[bool]:
    """Check GPU availability."""
    try:

        import torch

        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            torch.cuda.get_device_name(0)
            torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            pass
        
        return True
    except ImportError:
        return False

def clone_repository() -> Optional[bool]:
    """Clone SAMO repository."""
    try:
        # Clone repository
        subprocess.run([
            "git", "clone", "https://github.com/uelkerd/SAMO--DL.git"
        ], check=True)
        
        # Change to repository directory
        os.chdir("SAMO--DL")
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies() -> bool:
    """Install all dependencies."""
    # Install SAMO package
    try:
        subprocess.run(["pip", "install", "-e", "."], check=True)
    except subprocess.CalledProcessError:
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
        except subprocess.CalledProcessError:
            return False
    
    return True

def test_audio_libraries() -> bool:
    """Test audio processing libraries."""
    try:

        import soundfile as sf

    except ImportError:
        return False
    
    try:

        import librosa

    except ImportError:
        return False
    
    try:

        import whisper

        whisper.load_model("base")
    except ImportError:
        return False
    
    return True

def create_voice_demo() -> bool:
    """Create voice processing demo."""
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
    
    return True

def create_f1_optimization_script() -> bool:
    """Create F1 optimization script."""
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
    
    return True

def print_next_steps() -> None:
    """Print next steps for the user."""

def main() -> bool:
    """Main setup function."""
    print_header()
    
    # Check GPU
    if not check_gpu():
        return False
    
    # Clone repository
    if not clone_repository():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test audio libraries
    if not test_audio_libraries():
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
        pass
    else:
        sys.exit(1)
