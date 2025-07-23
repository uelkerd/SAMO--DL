# Model Optimization Scripts

This directory contains scripts for optimizing the BERT emotion classifier model in three key areas:
1. **Model Calibration** - Improving prediction accuracy through temperature scaling
2. **Model Compression** - Reducing model size and improving inference speed
3. **ONNX Conversion** - Enabling deployment on various platforms
4. **F1 Score Improvement** - Advanced techniques to boost model performance

## 1. Model Calibration

### Background
Our BERT emotion classifier was showing good training loss convergence but poor F1 scores (~7.5%) in evaluation. The root cause was identified as overconfident predictions and suboptimal threshold settings.

### Scripts
- `calibrate_model.py` - Finds optimal temperature and threshold values
- `test_calibration.py` - Tests model with optimal calibration settings
- `update_model_threshold.py` - Updates threshold in saved model

### Results
- **Before calibration**: F1 score = 0.075
- **After calibration**: F1 score = 0.132 (76% improvement)
- **Optimal settings**: Temperature = 1.0, Threshold = 0.6

## 2. Model Compression

### Background
The BERT model is large (~440MB) and computationally expensive, making it challenging to deploy in resource-constrained environments.

### Script
```bash
python scripts/compress_model.py [--input_model PATH] [--output_model PATH]
```

### Techniques
- **Dynamic Quantization**: Converts 32-bit floating-point weights to 8-bit integers
- **Linear Layer Optimization**: Focuses quantization on the most parameter-heavy layers
- **Size Measurement**: Tracks model size before and after compression

### Expected Results
- **Size reduction**: 75-80% smaller model size
- **Inference speedup**: 2-4x faster inference
- **Minimal accuracy loss**: <1% F1 score reduction

## 3. ONNX Conversion

### Background
ONNX (Open Neural Network Exchange) is an open format for representing machine learning models, enabling deployment across different frameworks and platforms.

### Script
```bash
python scripts/convert_to_onnx.py [--input_model PATH] [--output_model PATH]
```

### Features
- **Framework Interoperability**: Deploy with ONNX Runtime, TensorRT, etc.
- **Optimized Inference**: Graph optimizations for faster execution
- **Dynamic Axes**: Support for variable batch sizes and sequence lengths
- **Performance Benchmarking**: Compares PyTorch vs. ONNX inference speed

### Expected Results
- **Inference speedup**: 2-5x faster than PyTorch
- **Deployment flexibility**: Run on CPU, GPU, or specialized hardware
- **Reduced memory usage**: More efficient memory allocation

## 4. F1 Score Improvement

### Background
Multi-label emotion classification is challenging due to class imbalance, complex language patterns, and subjective annotations.

### Script
```bash
python scripts/improve_model_f1.py [--technique TECHNIQUE] [--output_model PATH]
```

### Techniques

#### Focal Loss
```bash
python scripts/improve_model_f1.py --technique focal_loss
```
- **How it works**: Reduces loss for well-classified examples, focusing on hard examples
- **Benefits**: Better handling of class imbalance
- **Expected improvement**: 10-20% F1 score increase

#### Data Augmentation
```bash
python scripts/improve_model_f1.py --technique augmentation
```
- **How it works**: Uses back-translation (English → German → English) to create paraphrased examples
- **Benefits**: Increases training data diversity
- **Expected improvement**: 5-15% F1 score increase

#### Ensemble Prediction
```bash
python scripts/improve_model_f1.py --technique ensemble
```
- **How it works**: Combines predictions from multiple models with different configurations
- **Benefits**: Reduces overfitting and improves generalization
- **Expected improvement**: 15-25% F1 score increase

## Integration with CircleCI

The model optimization pipeline is integrated into CircleCI:

```yaml
# .circleci/config.yml
model-validation:
  steps:
    # ...
    - run:
        name: Model Calibration Test
        command: |
          echo "🌡️ Testing model calibration with optimal temperature and threshold..."
          python scripts/test_calibration.py
    - run:
        name: Model Compression Test
        command: |
          echo "📦 Testing model compression..."
          python scripts/compress_model.py --input_model test_checkpoints/best_model.pt --output_model /tmp/compressed_model.pt
```

## Next Steps

1. **Deployment Pipeline**:
   - Create Docker container with ONNX Runtime
   - Set up model versioning and A/B testing
   - Implement automated performance monitoring

2. **Further Optimization**:
   - Knowledge distillation to smaller BERT models
   - Pruning to remove unnecessary connections
   - Mixed-precision training for faster training

3. **Advanced Techniques**:
   - Contrastive learning for better embeddings
   - Multi-task learning with related emotion tasks
   - Few-shot learning for rare emotions
