import torch
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Load checkpoint with weights_only=False
checkpoint_path = Path("test_checkpoints/best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Create model
model = create_bert_emotion_classifier()
model.load_state_dict(checkpoint['model_state_dict'])

print("✅ Model loaded successfully!")
print(f"Checkpoint keys: {list(checkpoint.keys())}")
print(f"Best F1 score: {checkpoint['best_score']:.4f}")
