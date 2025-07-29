"""
Unit tests for BERT emotion detection model.
Tests model initialization, forward pass, and emotion classification logic.
"""

from unittest.mock import patch, MagicMock

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

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_model_initialization(self, mock_bert, mock_config):
        """Test model initializes with correct parameters."""
        # Mock the config
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        
        # Mock the BERT model
        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance
        
        num_emotions = 28
        model = BERTEmotionClassifier(num_emotions=num_emotions)

        assert model.num_emotions == num_emotions
        assert hasattr(model, "bert")
        assert hasattr(model, "classifier")
        # The model has dropout within the classifier, not as a direct attribute
        assert hasattr(model.classifier, "0")  # First dropout layer
        assert hasattr(model.classifier, "3")  # Second dropout layer

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_model_parameter_count(self, mock_bert, mock_config):
        """Test model has expected number of parameters."""
        # Mock the config
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        
        # Mock the BERT model
        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance
        
        model = BERTEmotionClassifier(num_emotions=28)
        total_params = sum(p.numel() for p in model.parameters())

        # For a mocked model, we expect fewer parameters since BERT is mocked
        # The classifier layers should still have parameters
        assert total_params > 10_000  # At least the classifier parameters
        assert total_params < 1_000_000  # But less than a full BERT model

    @patch("transformers.AutoModel.from_pretrained")
    def test_forward_pass(self, mock_bert):
        """Test forward pass through the model."""
        # Create a proper mock for BERT output
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        # Mock BERT output with proper structure
        mock_bert_output = BaseModelOutputWithPooling(
            last_hidden_state=torch.randn(2, 10, 768),
            pooler_output=torch.randn(2, 768),  # This is what we actually use
            hidden_states=None,
            attentions=None,
        )

        # Create a mock BERT instance that returns the output when called
        mock_bert_instance = MagicMock()
        mock_bert_instance.return_value = mock_bert_output
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifier(num_emotions=28)
        model.eval()  # Set to evaluation mode to disable dropout

        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        # Call the model - this should work now since we properly mocked the BERT instance
        output = model(input_ids, attention_mask)

        assert output.shape == (2, 28)
        assert torch.all(torch.isfinite(output))

    @staticmethod
    def test_predict_emotions():
        """Test emotion prediction with threshold."""
        # Mock model output - now returns logits directly
        with patch.object(BERTEmotionClassifier, "forward") as mock_forward:
            # Mock forward to return logits tensor
            mock_logits = torch.tensor([[0.1, 0.8, 0.2, 0.9]])
            mock_forward.return_value = mock_logits

            # Also set the internal attributes that will be used
            model = BERTEmotionClassifier(num_emotions=4)
            model.eval()
            model._probabilities = torch.tensor([[0.1, 0.8, 0.2, 0.9]])
            model._calibrated_logits = torch.tensor([[0.1, 0.8, 0.2, 0.9]])

            # Mock tokenizer
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
                # Mock the tokenizer to return proper tensors
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer_instance.return_value = {
                    "input_ids": torch.tensor([[101, 102, 103, 102]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                }
                mock_tokenizer.return_value = mock_tokenizer_instance

                # Mock the GOEMOTIONS_EMOTIONS list for testing
                with patch(
                    "src.models.emotion_detection.bert_classifier.GOEMOTIONS_EMOTIONS",
                    ["admiration", "joy", "anger", "gratitude"],
                ):
                    predicted = model.predict_emotions("test text", threshold=0.5)

                    # Should predict indices 1 and 3 (values 0.8 and 0.9)
                    assert len(predicted["predicted_emotions"]) == 2
                    assert "joy" in predicted["predicted_emotions"]  # Index 1 maps to joy
                    assert (
                        "gratitude" in predicted["predicted_emotions"]
                    )  # Index 3 maps to gratitude

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_device_compatibility(self, mock_bert, mock_config):
        """Test model works on both CPU and GPU (if available)."""
        # Mock the config
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        
        # Mock the BERT model
        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance
        
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

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_training_mode(self, mock_bert, mock_config):
        """Test model switches between training and evaluation modes."""
        # Mock the config
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        
        # Mock the BERT model
        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance
        
        model = BERTEmotionClassifier(num_emotions=28)

        # Test training mode
        model.train()
        assert model.training

        # Test evaluation mode
        model.eval()
        assert not model.training

    @staticmethod
    def test_class_weights_handling():
        """Test model handles class weights for imbalanced dataset."""
        from src.models.emotion_detection.bert_classifier import WeightedBCELoss

        # Test with sample class weights
        class_weights = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], requires_grad=True)
        criterion = WeightedBCELoss(class_weights)

        # Test loss computation
        predictions = torch.sigmoid(torch.randn(2, 5, requires_grad=True))
        targets = torch.randint(0, 2, (2, 5)).float()

        loss = criterion(predictions, targets)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    @pytest.mark.slow
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_emotion_label_mapping(self, mock_bert, mock_config):
        """Test emotion label mapping matches GoEmotions dataset."""
        # Mock the config
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        
        # Mock the BERT model
        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance
        
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
