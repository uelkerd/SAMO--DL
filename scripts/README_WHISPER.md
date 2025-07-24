# OpenAI Whisper Integration - Voice-to-Text Processing

This directory contains scripts for testing, evaluating, and using the OpenAI Whisper-based voice-to-text processing module for SAMO Deep Learning.

## Voice Processing Components

The SAMO voice processing module includes:

1. **WhisperTranscriber**: Core integration with OpenAI Whisper model
2. **AudioPreprocessor**: Audio format handling and preprocessing
3. **TranscriptionAPI**: High-level API for integration with application
4. **CI Tests**: Continuous integration testing scripts

## Running CI Tests

The CI tests ensure that the Whisper integration is working correctly:

```bash
# Run the Whisper CI test
python scripts/ci/whisper_transcription_test.py

# Run API health checks (includes voice processing endpoints)
python scripts/ci/api_health_check.py
```

The Whisper CI test verifies:
- Proper module imports
- Model instantiation
- Audio preprocessing functionality
- Basic transcription pipeline (when not in CI environment)

## WER Evaluation

To evaluate Word Error Rate (WER) against the LibriSpeech test set:

```bash
# Basic evaluation with default settings
python scripts/evaluate_whisper_wer.py

# Evaluation with custom settings
python scripts/evaluate_whisper_wer.py --model-size base --samples 50 --output-dir ./evaluation_results
```

Options:
- `--model-size`: Whisper model size (tiny, base, small, medium, large)
- `--samples`: Number of LibriSpeech samples to evaluate
- `--output-dir`: Directory to save evaluation results
- `--librispeech-dir`: Directory for LibriSpeech samples (downloads if not provided)

The evaluation script:
1. Downloads LibriSpeech test-clean samples
2. Transcribes each sample using the specified model
3. Calculates WER and other metrics
4. Reports results and saves detailed analysis (if output directory specified)

## Usage in Application

To use the voice transcription in your application:

```python
from models.voice_processing.transcription_api import create_transcription_api

# Create API with desired model size
transcription_api = create_transcription_api(
    model_size="base",  # tiny, base, small, medium, or large
    language=None,      # None for auto-detect
    device=None         # None for auto-detect (CPU/CUDA)
)

# Transcribe a single audio file
result = transcription_api.transcribe("path/to/audio.mp3")
print(f"Transcription: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Audio quality: {result['audio_quality']}")

# Get performance metrics
metrics = transcription_api.get_performance_metrics()
print(f"Average real-time factor: {metrics['average_real_time_factor']:.2f}x")
```

## Supported Audio Formats

The voice processing module supports:
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- FLAC (.flac)

## Performance Characteristics

- **Processing Speed**: Real-time or faster on GPU (RTF < 1.0)
- **Accuracy**: Word Error Rate < 15% on clear speech
- **Maximum Duration**: 5 minutes (300 seconds)
- **Model Sizes**:
  - tiny: ~39M parameters
  - base: ~74M parameters
  - small: ~244M parameters
  - medium: ~769M parameters
  - large: ~1550M parameters 