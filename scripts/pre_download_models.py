#!/usr/bin/env python3
"""Pre-download models for Docker build optimization."""
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Pre-download all required models."""
    # Create models directory
    os.makedirs('/app/models', exist_ok=True)

    print('🚀 Pre-downloading DeBERTa-v3 emotion model...')
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = 'duelker/samo-goemotions-deberta-v3-large'
        print(f'Downloading {model_name}...')
        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/app/models')
        _model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/app/models')
        print('✅ DeBERTa-v3 model downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading DeBERTa-v3 model: {e}')
        raise

    print('🚀 Pre-downloading T5 summarization model...')
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        t5_model = 't5-small'
        print(f'Downloading {t5_model}...')
        _t5_tokenizer = T5Tokenizer.from_pretrained(t5_model, cache_dir='/app/models')
        _t5_model_obj = T5ForConditionalGeneration.from_pretrained(t5_model, cache_dir='/app/models')
        print('✅ T5 model downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading T5 model: {e}')
        raise

    print('🚀 Pre-downloading Whisper model...')
    try:
        # Check numpy availability first
        try:
            import numpy
            print(f'✅ Numpy {numpy.__version__} available')
        except ImportError:
            print('⚠️ Installing numpy...')
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
            import numpy
            print(f'✅ Numpy {numpy.__version__} installed and available')

        import whisper
        whisper_model = 'base'
        print(f'Downloading Whisper {whisper_model}...')
        whisper.load_model(whisper_model, download_root='/app/models')
        print('✅ Whisper model downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading Whisper model: {e}')
        # Don't fail the entire build for Whisper - continue without it
        print('⚠️ Continuing without Whisper model - will be downloaded at runtime if needed')

    print('🎉 Core models pre-downloaded successfully!')

if __name__ == '__main__':
    main()
