"""
Unit tests for BERT emotion detection model.
Tests model initialization, forward pass, and emotion classification logic.
"""

from unittest.mock import Mock, patch

import pytest
import torch

try:
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
except ImportError as e:
    raise RuntimeError(
        "Failed to import BERTEmotionClassifier. Ensure all model dependencies are installed."
    ) from e


class TestBertEmotionClassifier:
    """Test suite for BERT emotion detection classifier."""

    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        num_emotions = 28
        model = BERTEmotionClassifier(num_emotions=num_emotions)

        assert model.num_emotions == num_emotions
        assert hasattr(model, "bert")
        assert hasattr(model, "classifier")
        assert hasattr(model, "dropout")

    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        model = BERTEmotionClassifier(num_emotions=28)
        total_params = sum(p.numel() for p in model.parameters())

        # BERT-base has ~110M parameters, with classifier should be ~110M+
        assert total_params > 100_000_000
        assert total_params < 200_000_000

    @patch("src.models.emotion_detection.bert_classifier.BertModel")
    def test_forward_pass(self, mock_bert):
        """Test model forward pass with mock BERT."""
        # Setup mock
        mock_bert_instance = Mock()
        mock_bert_instance.config.hidden_size = 768
        mock_bert_output = Mock()
        mock_bert_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_bert_instance.return_value = mock_bert_output
        mock_bert.from_pretrained.return_value = mock_bert_instance

        model = BERTEmotionClassifier(num_emotions=28)

        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = model(input_ids, attention_mask)

        assert output.shape == (2, 28)
        assert torch.all(torch.isfinite(output))

    def test_predict_emotions(self):
        """Test emotion prediction with threshold."""
        # Mock model output
        with patch.object(BERTEmotionClassifier, "forward") as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.8, 0.2, 0.9]])

            model = BERTEmotionClassifier(num_emotions=4)
            model.eval()

            # Mock tokenizer
            with patch("src.models.emotion_detection.bert_classifier.BertTokenizer"):
                predicted = model.predict_emotions("test text", threshold=0.5)

                # Should predict indices 1 and 3 (values 0.8 and 0.9)
                assert len(predicted) == 2
                assert 1 in predicted
                assert 3 in predicted

    def test_device_compatibility(self):
        """Test model works on both CPU and GPU (if available)."""
        model = BERTEmotionClassifier(num_emotions=28)

        # Test CPU
        device = torch.device("cpu")
        model = model.to(device)
        assert next(model.parameters()).device == device

        # Test GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            assert next(model.parameters()).device == device

    def test_training_mode(self):
        """Test model switches between training and evaluation modes."""
        model = BERTEmotionClassifier(num_emotions=28)

        # Test training mode
        model.train()
        assert model.training

        # Test evaluation mode
        model.eval()
        assert not model.training

    def test_class_weights_handling(self):
        """Test model handles class weights for imbalanced dataset."""
        from src.models.emotion_detection.bert_classifier import WeightedBCELoss

        # Test with sample class weights
        class_weights = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        criterion = WeightedBCELoss(class_weights)

        # Test loss computation
        predictions = torch.sigmoid(torch.randn(2, 5))
        targets = torch.randint(0, 2, (2, 5)).float()

        loss = criterion(predictions, targets)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    @pytest.mark.slow
    def test_emotion_label_mapping(self):
        """Test emotion label mapping matches GoEmotions dataset."""
        model = BERTEmotionClassifier(num_emotions=28)

        expected_emotions = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        # Test that we expect 28 emotions (27 + neutral)
        assert model.num_emotions == len(expected_emotions)
        # Note: Actual label mapping would be tested in integration tests
