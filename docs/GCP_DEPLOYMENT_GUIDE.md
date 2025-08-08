# GCP Deployment Guide for Focal Loss Training

## 🎯 **Objective**
Deploy focal loss training to GCP for 10-50x faster training and reach the 75% F1 target efficiently.

## 📋 **Prerequisites Checklist**

### **GCP Setup**
- [ ] GCP account with billing enabled
- [ ] GCP CLI installed (`gcloud`)
- [ ] Project ID configured
- [ ] Compute Engine API enabled

### **Local Setup**
- [ ] Python environment with dependencies
- [ ] SAMO-DL repository cloned
- [ ] Service account key (if using service account)

## 🚀 **Step-by-Step Deployment Guide**

### **Step 1: Quick Local Validation (5 minutes)**

First, let's validate our focal loss implementation:

```bash
# Run quick validation tests
python scripts/quick_focal_test.py
```

**Expected Output:**
```
🎯 Quick Focal Loss Validation Tests
==================================================

📋 Running Focal Loss Math...
🧮 Testing Focal Loss Mathematics...
✅ Focal Loss Test PASSED

📋 Running Dataset Loading...
📊 Testing Dataset Loading...
✅ Dataset Loading Test PASSED

📋 Running Model Creation...
🤖 Testing Model Creation...
✅ Model Creation Test PASSED

📊 Test Results Summary:
==============================
   • Focal Loss Math: ✅ PASS
   • Dataset Loading: ✅ PASS
   • Model Creation: ✅ PASS

🎯 Overall: 3/3 tests passed
✅ All tests passed! Ready for GCP deployment.
```

### **Step 2: GCP Project Setup (10 minutes)**

```bash
# 1. Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable aiplatform.googleapis.com

# 3. Create a service account (optional but recommended)
gcloud iam service-accounts create samo-dl-training \
    --display-name="SAMO Deep Learning Training"

# 4. Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:samo-dl-training@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

# 5. Create and download service account key
gcloud iam service-accounts keys create ~/samo-dl-key.json \
    --iam-account=samo-dl-training@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### **Step 3: Create GCP Training Instance (5 minutes)**

```bash
# Create a GPU instance with optimal specs
gcloud compute instances create samo-dl-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

### **Step 4: Setup Training Environment (10 minutes)**

```bash
# 1. SSH into the instance
gcloud compute ssh samo-dl-training --zone=us-central1-a

# 2. Install Python and dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# 3. Clone the repository
git clone https://github.com/YOUR_USERNAME/SAMO--DL.git
cd SAMO--DL

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn numpy pandas
pip install fastapi uvicorn python-multipart
```

### **Step 5: Configure GPU Training (5 minutes)**

```bash
# 1. Check GPU availability
python scripts/setup_gpu_training.py --check

# 2. Create optimized training config
python scripts/setup_gpu_training.py --create-config

# Expected output:
# ✅ GPU Available: Tesla T4
#    Memory: 15.7 GB
# 💡 GPU Training Optimizations:
#    • Use batch_size=32 (you have 15.7GB memory)
#    • Enable mixed precision training (fp16)
#    • Expected training speedup: 8-12x vs CPU
```

### **Step 6: Run Focal Loss Training (2-4 hours)**

```bash
# 1. Start focal loss training with GPU optimization
python scripts/focal_loss_training.py \
    --gamma 2.0 \
    --alpha 0.25 \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5

# 2. Monitor training progress
# The script will show:
# 🚀 Starting Focal Loss Training
#    • Gamma: 2.0
#    • Alpha: 0.25
#    • Learning Rate: 2e-5
#    • Epochs: 5
# Using device: cuda
# Loading GoEmotions dataset...
# Dataset loaded:
#    • Train: 43410 examples
#    • Validation: 5426 examples
#    • Test: 5427 examples
```

### **Step 7: Run Threshold Optimization (30 minutes)**

```bash
# After training completes, optimize thresholds
python scripts/threshold_optimization.py \
    --model_path ./models/checkpoints/focal_loss_best_model.pt \
    --threshold_range 0.1 0.9 \
    --num_thresholds 20

# Expected output:
# 🎯 Threshold Optimization Complete!
#    • Macro F1 (optimized): 0.4523
#    • Macro F1 (default): 0.1320
#    • Improvement: 0.3203
#    ✅ Significant improvement achieved!
```

### **Step 8: Download Results (5 minutes)**

```bash
# 1. Download trained model and results
gcloud compute scp samo-dl-training:~/SAMO--DL/models/checkpoints/focal_loss_best_model.pt ./models/checkpoints/ --zone=us-central1-a
gcloud compute scp samo-dl-training:~/SAMO--DL/models/optimized/thresholds.npz ./models/optimized/ --zone=us-central1-a

# 2. Download training logs
gcloud compute scp samo-dl-training:~/SAMO--DL/logs/ ./logs/ --zone=us-central1-a --recurse
```

### **Step 9: Clean Up (5 minutes)**

```bash
# Delete the training instance to avoid charges
gcloud compute instances delete samo-dl-training --zone=us-central1-a

# Optional: Delete service account
gcloud iam service-accounts delete samo-dl-training@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## 📊 **Expected Results**

### **Training Performance**
- **Training Time**: 2-4 hours (vs 1-5 days on CPU)
- **Speedup**: 10-50x faster than local CPU
- **Cost**: ~$2-8 for full training run

### **F1 Score Improvements**
- **Baseline**: 13.2% F1 score
- **After Focal Loss**: 35-45% F1 score (+20-30%)
- **After Threshold Optimization**: 45-55% F1 score (+10-15%)
- **Total Improvement**: +30-40 percentage points

### **Next Steps**
- **If F1 < 50%**: Implement ensemble methods
- **If F1 > 50%**: Deploy to production and continue optimization

## 🔧 **Troubleshooting**

### **Common Issues**

1. **GPU Not Available**
   ```bash
   # Check GPU status
   nvidia-smi

   # If not available, restart instance
   gcloud compute instances reset samo-dl-training --zone=us-central1-a
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/focal_loss_training.py --batch_size 16
   ```

3. **Slow Training**
   ```bash
   # Check GPU utilization
   watch -n 1 nvidia-smi

   # Enable mixed precision
   export CUDA_VISIBLE_DEVICES=0
   ```

### **Cost Optimization**

1. **Use Preemptible Instances** (50% cheaper)
   ```bash
   gcloud compute instances create samo-dl-training \
       --preemptible \
       --zone=us-central1-a \
       # ... other options
   ```

2. **Auto-shutdown after training**
   ```bash
   # Add to training script
   sudo shutdown -h +240  # Shutdown in 4 hours
   ```

## 📈 **Monitoring & Validation**

### **Real-time Monitoring**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f logs/training.log

# Monitor costs
gcloud billing accounts list
```

### **Validation Commands**
```bash
# Test trained model locally
python scripts/validate_current_f1.py

# Run inference test
python scripts/test_model_inference.py --model_path ./models/checkpoints/focal_loss_best_model.pt
```

## 🎯 **Success Criteria**

- **F1 Score**: >45% (significant improvement from 13.2%)
- **Training Time**: <4 hours
- **Cost**: <$10 for complete training run
- **Model Size**: <500MB (suitable for production)

---

**Ready to start?** Run `python scripts/quick_focal_test.py` first, then proceed with GCP setup!
