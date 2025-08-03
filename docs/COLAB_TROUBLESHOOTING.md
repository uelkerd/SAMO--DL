# 🚀 Colab Troubleshooting Guide

## Common Issues and Solutions

### 1. **Runtime Disconnection Issues**

**Problem**: Colab disconnects during long training sessions.

**Solutions**:
- **Use Colab Pro** for longer runtime sessions
- **Enable "Keep alive" scripts**:
  ```python
  # Add this to your notebook to prevent disconnection
  import time
  import threading
  
  def keep_alive():
      while True:
          time.sleep(60)
          print("Still alive...")
  
  # Start keep-alive thread
  thread = threading.Thread(target=keep_alive, daemon=True)
  thread.start()
  ```

### 2. **GPU Runtime Issues**

**Problem**: Can't connect to GPU runtime.

**Solutions**:
- **Check GPU availability**: Runtime → Change runtime type → Hardware accelerator → GPU
- **Wait for GPU**: GPUs may be temporarily unavailable
- **Use Colab Pro** for guaranteed GPU access
- **Alternative**: Use TPU if GPU unavailable

### 3. **Memory Issues**

**Problem**: Out of memory errors during training.

**Solutions**:
- **Reduce batch size**: Change `per_device_train_batch_size` from 16 to 8
- **Enable gradient checkpointing**: Add `gradient_checkpointing=True` to TrainingArguments
- **Use mixed precision**: Ensure `fp16=True` is set
- **Clear memory**: Add `torch.cuda.empty_cache()` between cells

### 4. **Dependency Installation Issues**

**Problem**: Package installation fails.

**Solutions**:
- **Restart runtime** after installing packages
- **Use specific versions**:
  ```python
  !pip install transformers==4.35.0 torch==2.1.0 accelerate==0.26.0
  ```
- **Install one by one** if batch installation fails

### 5. **File Path Issues**

**Problem**: `FileNotFoundError` when loading data.

**Solutions**:
- **Check current directory**: `!pwd`
- **List files**: `!ls -la`
- **Use absolute paths**: `/content/SAMO--DL/data/`
- **Clone repository properly**: Ensure git clone completes

### 6. **Model Loading Issues**

**Problem**: Model fails to load or initialize.

**Solutions**:
- **Check internet connection**: Model downloads require stable connection
- **Use smaller model**: Try `distilbert-base-uncased` instead of `bert-base-uncased`
- **Clear cache**: `!rm -rf ~/.cache/huggingface/`

### 7. **Training Performance Issues**

**Problem**: Training is slow or inefficient.

**Solutions**:
- **Use GPU**: Ensure GPU runtime is active
- **Enable mixed precision**: `fp16=True`
- **Optimize batch size**: Balance between memory and speed
- **Use gradient accumulation**: `gradient_accumulation_steps=2`

### 8. **Colab Enterprise Issues**

Based on [Google Cloud documentation](https://cloud.google.com/colab/docs/troubleshooting):

**Authentication Issues**:
- Enable "Additional services without individual control" in Google Workspace
- Check browser cookie settings for `DATALAB_TUNNEL_TOKEN`
- Configure firewall rules for `*.aiplatform-notebook.cloud.google.com`

**Runtime Connection Issues**:
- Wait for runtime allocation (can take several minutes)
- Check network connectivity
- Verify service restrictions aren't blocking access

### 9. **Prevention Strategies**

**Best Practices**:
1. **Save frequently**: Download models and results regularly
2. **Use version control**: Commit important changes
3. **Monitor resources**: Check GPU memory usage
4. **Plan for disconnections**: Structure code to resume training
5. **Backup data**: Store datasets in Google Drive

### 10. **Emergency Recovery**

**If Colab disconnects during training**:
1. **Checkpoint saving**: Ensure `save_strategy="steps"` is set
2. **Resume training**: Load the latest checkpoint
3. **Reduce complexity**: Use smaller model or dataset if needed
4. **Alternative platforms**: Consider Google Cloud AI Platform or Vertex AI

## Quick Fix Commands

```python
# Check GPU availability
!nvidia-smi

# Check memory usage
!free -h

# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Check current directory
!pwd
!ls -la

# Restart runtime (if needed)
# Runtime → Restart runtime
```

## Support Resources

- [Colab Troubleshooting Guide](https://cloud.google.com/colab/docs/troubleshooting)
- [Colab Runtime Issues](https://github.com/oumaima1220/Resolve_disconnecting_googlecolab)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch GPU Guide](https://pytorch.org/docs/stable/notes/cuda.html)

---

**Remember**: Most issues can be resolved by restarting the runtime and ensuring proper setup. Always save your work frequently! 