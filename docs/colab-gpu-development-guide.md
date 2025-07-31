# SAMO Deep Learning - Colab GPU Development Guide

## 🎯 Overview

This guide provides comprehensive instructions for leveraging Google Colab's GPU resources for SAMO Deep Learning development, with a specific focus on **REQ-DL-012: Domain-Adapted Emotion Detection**. The Colab phase is critical for achieving our target of 70% F1 score on journal entries through GPU-accelerated training and domain adaptation techniques.

## 🚀 Quick Start

### 1. Colab Environment Setup

```python
# Enable GPU Runtime
# Runtime → Change runtime type → GPU (T4 or V100)

# Install dependencies
!pip install torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install transformers>=4.30.0 datasets>=2.13.0 evaluate scikit-learn pandas numpy matplotlib seaborn
!pip install accelerate wandb pydub openai-whisper jiwer

# Clone repository
!git clone https://github.com/uelkerd/SAMO--DL.git
%cd SAMO--DL

# Verify GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2. Run Domain Adaptation Notebook

```python
# Open the domain adaptation notebook
%run notebooks/domain_adaptation_gpu_training.ipynb
```

## 🎯 REQ-DL-012: Domain Adaptation Focus

### **Target Metrics**
- **Primary Goal**: 70% F1 score on journal entry test set
- **Secondary Goal**: Maintain >75% F1 on GoEmotions validation
- **Success Criteria**: Bridge domain gap between Reddit comments and journal entries

### **Domain Gap Analysis**
The critical insight driving REQ-DL-012:

| Aspect | GoEmotions (Reddit) | Journal Entries | Impact |
|--------|---------------------|-----------------|---------|
| **Length** | 10-50 words | 50-200 words | Model needs longer context |
| **Style** | Casual, social | Personal, reflective | Different language patterns |
| **Emotion Expression** | Direct, explicit | Nuanced, implicit | Harder to detect emotions |
| **Personal Pronouns** | Low frequency | High frequency | Different linguistic markers |
| **Reflection Level** | Surface-level | Deep, introspective | Requires understanding context |

## 🔧 Technical Architecture

### **GPU-Optimized Model Architecture**

```python
class DomainAdaptedEmotionClassifier(nn.Module):
    """BERT-based emotion classifier with domain adaptation capabilities."""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=12, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Domain adaptation layer
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # 2 domains: GoEmotions vs Journal
        )
    
    def forward(self, input_ids, attention_mask, domain_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Emotion classification
        emotion_logits = self.classifier(self.dropout(pooled_output))
        
        # Domain classification (for domain adaptation)
        domain_logits = self.domain_classifier(pooled_output)
        
        if domain_labels is not None:
            return emotion_logits, domain_logits
        return emotion_logits
```

### **Focal Loss Implementation**

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in emotion detection."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

## 📊 Data Pipeline

### **Journal Test Dataset**

Our custom dataset (`data/journal_test_dataset.json`) contains:
- **150+ realistic journal entries** with varied emotional content
- **12 emotion categories** matching GoEmotions taxonomy
- **Personal, reflective writing style** to test domain adaptation
- **Metadata** including word count, topics, and timestamps

### **Data Loading Strategy**

```python
# Load datasets
with open('data/journal_test_dataset.json', 'r') as f:
    journal_entries = json.load(f)

journal_df = pd.DataFrame(journal_entries)
go_emotions = load_dataset("go_emotions", "simplified")

# Prepare for training
journal_texts = journal_df['content'].tolist()
journal_labels = [label_to_id[emotion] for emotion in journal_df['emotion']]

# Create balanced training set
combined_dataset = ConcatDataset([go_dataset, journal_dataset])
```

## 🎯 Training Strategy

### **Phase 1: Domain Gap Analysis (30 minutes)**
```python
# Analyze differences between domains
def analyze_writing_style(texts, domain_name):
    avg_length = np.mean([len(text.split()) for text in texts])
    personal_pronouns = sum(['I ' in text or 'my ' in text for text in texts]) / len(texts)
    reflection_words = sum(['think' in text.lower() or 'feel' in text.lower() 
                           for text in texts]) / len(texts)
    
    print(f"{domain_name} Style Analysis:")
    print(f"  Average length: {avg_length:.1f} words")
    print(f"  Personal pronouns: {personal_pronouns:.1%}")
    print(f"  Reflection words: {reflection_words:.1%}")
```

### **Phase 2: Domain Adaptation Training (2-4 hours)**
```python
# Multi-task training with domain adaptation
for epoch in range(num_epochs):
    # Train on GoEmotions data
    for batch in go_loader:
        domain_labels = torch.zeros(batch['input_ids'].size(0), dtype=torch.long)
        losses = trainer.train_step(batch, domain_labels, lambda_domain=0.1)
    
    # Train on journal data
    for batch in journal_train_loader:
        domain_labels = torch.ones(batch['input_ids'].size(0), dtype=torch.long)
        losses = trainer.train_step(batch, domain_labels, lambda_domain=0.1)
    
    # Validate on journal test set
    val_results = trainer.evaluate(journal_val_loader)
    print(f"Epoch {epoch}: F1 = {val_results['f1_macro']:.4f}")
```

### **Phase 3: Performance Optimization (1-2 hours)**
```python
# Temperature scaling for confidence calibration
def calibrate_model(model, val_loader):
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'], batch['attention_mask'])
            logits_list.append(logits)
            labels_list.append(batch['labels'])
    
    # Fit temperature scaling
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    return temperature.item()
```

## 📈 Monitoring & Evaluation

### **Real-time Metrics Tracking**

```python
# Initialize wandb for experiment tracking
wandb.init(project="samo-domain-adaptation", name="journal-emotion-detection")

# Log metrics during training
wandb.log({
    'epoch': epoch,
    'train_loss': avg_loss,
    'val_loss': val_results['loss'],
    'val_f1_macro': val_results['f1_macro'],
    'val_f1_weighted': val_results['f1_weighted'],
    'domain_gap_reduction': domain_gap_metric
})
```

### **Success Metrics Dashboard**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Journal F1 Score** | ≥70% | TBD | 🎯 Target |
| **GoEmotions F1 Score** | ≥75% | TBD | 🎯 Target |
| **Domain Gap Reduction** | ≥20% | TBD | 🎯 Target |
| **Training Time** | <4 hours | TBD | ⏱️ Monitor |
| **GPU Memory Usage** | <80% | TBD | 💾 Monitor |

## 🔧 Troubleshooting Guide

### **Common GPU Issues**

#### **Out of Memory (OOM)**
```python
# Solution 1: Reduce batch size
batch_size = 8  # Instead of 16

# Solution 2: Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### **Slow Training**
```python
# Solution 1: Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True

# Solution 2: Use DataLoader with num_workers
dataloader = DataLoader(dataset, batch_size=16, num_workers=2, pin_memory=True)

# Solution 3: Optimize data preprocessing
def preprocess_batch(batch):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```

### **Domain Adaptation Issues**

#### **Poor Domain Transfer**
```python
# Solution 1: Adjust domain adaptation weight
lambda_domain = 0.05  # Reduce from 0.1 if overfitting

# Solution 2: Use gradient reversal layer
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return -self.alpha * grad_output

# Solution 3: Progressive domain adaptation
for epoch in range(num_epochs):
    lambda_domain = min(0.1, epoch * 0.02)  # Gradually increase
```

#### **Class Imbalance**
```python
# Solution 1: Adjust focal loss parameters
focal_loss = FocalLoss(alpha=2.0, gamma=3.0)  # More aggressive

# Solution 2: Use weighted sampling
from torch.utils.data import WeightedRandomSampler
weights = compute_class_weights(dataset)
sampler = WeightedRandomSampler(weights, len(weights))
dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

# Solution 3: Data augmentation
def augment_journal_entry(text):
    # Add noise, synonyms, or paraphrasing
    augmented = apply_synonym_replacement(text)
    return augmented
```

## 📋 Best Practices

### **GPU Memory Management**
1. **Monitor memory usage** with `nvidia-smi`
2. **Use gradient checkpointing** for large models
3. **Clear cache** between experiments: `torch.cuda.empty_cache()`
4. **Profile memory usage** with `torch.profiler`

### **Experiment Organization**
1. **Use descriptive run names** in wandb
2. **Save model checkpoints** regularly
3. **Document hyperparameters** for each experiment
4. **Version control** your notebooks and scripts

### **Data Quality Assurance**
1. **Validate journal dataset** before training
2. **Check for data leakage** between train/val sets
3. **Monitor class distribution** across domains
4. **Sanitize text data** for consistency

## 🎯 Success Criteria & Validation

### **REQ-DL-012 Validation Checklist**

- [ ] **Domain Gap Analysis Complete**
  - [ ] Writing style differences quantified
  - [ ] Emotion distribution compared
  - [ ] Linguistic patterns analyzed

- [ ] **Training Pipeline Operational**
  - [ ] GPU training running successfully
  - [ ] Focal loss implementation working
  - [ ] Domain adaptation layers functional

- [ ] **Performance Targets Met**
  - [ ] Journal F1 score ≥70%
  - [ ] GoEmotions F1 score ≥75%
  - [ ] Training time <4 hours

- [ ] **Model Export Ready**
  - [ ] Best model saved and validated
  - [ ] Performance metrics documented
  - [ ] Deployment artifacts created

### **Final Validation Script**

```python
def validate_req_dl_012():
    """Comprehensive validation for REQ-DL-012."""
    
    # Load best model
    model.load_state_dict(torch.load('best_domain_adapted_model.pth'))
    
    # Test on journal dataset
    journal_results = evaluate_on_journal_dataset(model)
    
    # Test on GoEmotions dataset
    go_emotions_results = evaluate_on_go_emotions_dataset(model)
    
    # Validate requirements
    journal_f1 = journal_results['f1_macro']
    go_emotions_f1 = go_emotions_results['f1_macro']
    
    print("🎯 REQ-DL-012 Validation Results:")
    print(f"  Journal F1 Score: {journal_f1:.4f} (Target: ≥0.70)")
    print(f"  GoEmotions F1 Score: {go_emotions_f1:.4f} (Target: ≥0.75)")
    print(f"  Journal Target Met: {'✅' if journal_f1 >= 0.7 else '❌'}")
    print(f"  GoEmotions Target Met: {'✅' if go_emotions_f1 >= 0.75 else '❌'}")
    
    return journal_f1 >= 0.7 and go_emotions_f1 >= 0.75
```

## 🚀 Next Steps After Colab Phase

### **Immediate Actions**
1. **Model Integration**: Integrate trained model into SAMO-DL pipeline
2. **API Enhancement**: Update emotion detection API with domain-aware scoring
3. **Performance Testing**: Validate model performance in production environment
4. **Documentation**: Update PRD with achieved metrics

### **Long-term Improvements**
1. **Continuous Training**: Set up automated retraining pipeline
2. **Domain Expansion**: Apply techniques to other domains (social media, emails)
3. **Model Compression**: Optimize for edge deployment
4. **Advanced Techniques**: Implement adversarial training, contrastive learning

## 📞 Support & Resources

### **Key Files**
- `notebooks/domain_adaptation_gpu_training.ipynb` - Main training notebook
- `data/journal_test_dataset.json` - Journal test dataset
- `scripts/create_journal_test_dataset.py` - Dataset generation script
- `docs/samo-dl-prd.md` - Product requirements document

### **Useful Commands**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check available memory
!free -h

# Download trained model
from google.colab import files
files.download('best_domain_adapted_model.pth')

# Save notebook to GitHub
!git add notebooks/domain_adaptation_gpu_training.ipynb
!git commit -m "feat: Add domain adaptation training results"
!git push
```

### **Troubleshooting Resources**
- [PyTorch GPU Guide](https://pytorch.org/docs/stable/notes/cuda.html)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Weights & Biases Guide](https://docs.wandb.ai/)

---

**Last Updated**: July 31, 2025  
**Version**: 1.0.0  
**Status**: Ready for Colab Development 🚀  
**Target**: REQ-DL-012 Domain Adaptation Success ✅ 