# Architecture Decision: ONNX-Only Production with Vertex AI Training

## Decision Made: August 6, 2025

**Architecture**: ONNX-only production deployment with Vertex AI training pipeline

## Rationale

### Why ONNX-Only Production?
- ✅ **Model Performance**: Current model achieves >90% F1 score (excellent)
- ✅ **Production Optimization**: ONNX provides 2.3x inference speedup
- ✅ **Deployment Efficiency**: Smaller footprint, faster startup, cross-platform
- ✅ **No Refactoring Needed**: Current implementation is production-ready

### Why Vertex AI Training?
- ✅ **Separation of Concerns**: Training separate from inference
- ✅ **Scalability**: GCP-managed training infrastructure
- ✅ **Cost Efficiency**: Pay-per-use training resources
- ✅ **Future-Proof**: Supports advanced training techniques

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   Conversion    │    │   Production    │
│   (Vertex AI)   │    │   (Pipeline)    │    │   (Cloud Run)   │
│                 │    │                 │    │                 │
│   PyTorch       │───▶│   PyTorch →     │───▶│   ONNX Runtime  │
│   Training      │    │   ONNX Conv     │    │   Inference     │
│                 │    │                 │    │                 │
│   • New models  │    │   • Model prep  │    │   • Fast inf    │
│   • Fine-tuning │    │   • Validation  │    │   • Optimized   │
│   • Domain adapt│    │   • Testing     │    │   • Monitoring  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Details

### Production (Cloud Run)
- **Framework**: ONNX Runtime only
- **Dependencies**: Minimal (flask, onnxruntime, numpy, gunicorn)
- **Performance**: 2.3x speedup over PyTorch
- **Deployment**: Containerized, auto-scaling

### Training (Vertex AI) - BACKLOG
- **Framework**: PyTorch
- **Infrastructure**: GCP-managed
- **Cost**: Pay-per-use
- **Capabilities**: GPU acceleration, distributed training

### Conversion Pipeline - BACKLOG
- **Input**: PyTorch model (.pt)
- **Output**: ONNX model (.onnx)
- **Validation**: Accuracy verification
- **Automation**: CI/CD integration

## Benefits

### Production Benefits
- **Performance**: 2.3x inference speedup
- **Reliability**: Optimized for production workloads
- **Scalability**: Efficient resource utilization
- **Monitoring**: Comprehensive metrics and logging

### Development Benefits
- **Flexibility**: PyTorch for research and experimentation
- **Scalability**: Vertex AI for large-scale training
- **Cost Efficiency**: Pay only for training resources used
- **Future-Proof**: Supports emerging training techniques

## Migration Path

### Current State
- ✅ ONNX-only production API (deployment/cloud-run/onnx_api_server.py)
- ✅ Production-ready with >90% F1 score
- ✅ Comprehensive monitoring and testing
- ✅ Cloud Run deployment: samo-emotion-api

### Next Steps
1. **Documentation**: Update PRD and architecture docs ✅
2. **Training Pipeline**: Set up Vertex AI training workflow (BACKLOG)
3. **Conversion Pipeline**: Implement PyTorch → ONNX conversion (BACKLOG)
4. **CI/CD Integration**: Automate model updates (BACKLOG)
5. **Monitoring**: Enhanced model drift detection (BACKLOG)

## Success Metrics

- **Production Performance**: <500ms inference latency
- **Training Efficiency**: <4 hours for model updates
- **Cost Optimization**: <$10 per training run
- **Reliability**: >99.5% uptime

## Future Considerations

- **Model Updates**: Automated retraining triggers
- **Domain Adaptation**: Journal-specific fine-tuning
- **Performance Optimization**: Further ONNX optimizations
- **Monitoring**: Advanced drift detection and alerting

## Status: ✅ APPROVED

This architecture provides optimal performance, scalability, and maintainability for the SAMO emotion detection system.
