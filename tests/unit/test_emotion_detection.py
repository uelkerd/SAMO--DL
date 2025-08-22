#!/usr/bin/env python3
"""
Unit tests for emotion detection models.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers.modeling_outputs import BaseModelOutputWithPooling

try:
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
except ImportError as e:
    raise RuntimeError(
        f"Failed to import BERTEmotionClassifier: {e}. "
        "Make sure all dependencies are installed."
    )


class TestBertEmotionClassifier:
    """Test suite for BERT emotion detection classifier."""

    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_model_initializationself, mock_bert, mock_config:
        """Test model initializes with correct parameters."""
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance

        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance

        num_emotions = 28
        model = BERTEmotionClassifiernum_emotions=num_emotions

        assert model.num_emotions == num_emotions
        assert hasattrmodel, "bert"
        assert hasattrmodel, "classifier"
        assert hasattrmodel.classifier, "0"  # First dropout layer
        assert hasattrmodel.classifier, "3"  # Second dropout layer

    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_model_parameter_countself, mock_bert, mock_config:
        """Test model has expected number of parameters."""
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance

        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifiernum_emotions=28
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params > 10_000  # At least the classifier parameters
        assert total_params < 1_000_000  # But less than a full BERT model

    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_forward_passself, mock_bert, mock_config:
        """Test forward pass through the model."""
        # Provide a minimal config so model init doesn't hit network
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance
        mock_bert_output = BaseModelOutputWithPooling(
            last_hidden_state=torch.randn2, 10, 768,
            pooler_output=torch.randn2, 768,  # This is what we actually use
            hidden_states=None,
            attentions=None,
        )

        mock_bert_instance = MagicMock()
        mock_bert_instance.return_value = mock_bert_output
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifiernum_emotions=28
        model.eval()  # Set to evaluation mode to disable dropout

        input_ids = torch.randint(0, 1000, 2, 10)
        attention_mask = torch.ones2, 10

        output = modelinput_ids, attention_mask

        assert output.shape == 2, 28
        assert torch.all(torch.isfiniteoutput)

    def test_predict_emotionsself:
        """Test emotion prediction functionality."""
        with patch"transformers.AutoConfig.from_pretrained", patch(
            "transformers.AutoModel.from_pretrained"
        ), patch"transformers.AutoTokenizer.from_pretrained" as mock_tokenizer:
            model = BERTEmotionClassifiernum_emotions=4
            model.eval()

            # Mock the tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.tensor[[1, 2, 3, 0]],  # [batch, seq_len]
                "attention_mask": torch.tensor[[1, 1, 1, 0]]  # [batch, seq_len]
            }
            mock_tokenizer.return_value = mock_tokenizer_instance

            # Mock the forward method to return proper logits
            mock_logits = torch.randn1, 4  # [batch, num_emotions]
            model.forward = MagicMockreturn_value=mock_logits

            # Test with threshold
            predictions = model.predict_emotions(
                texts=["test text"],
                threshold=0.5,
            )

            # Verify predictions structure
            assert isinstancepredictions, dict
            assert "emotions" in predictions
            assert "probabilities" in predictions
            assert "predictions" in predictions

    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_device_compatibilityself, mock_bert, mock_config:
        """Test model works on different devices."""
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance

        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifiernum_emotions=28

        # Test CPU
        model.to"cpu"
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model.to"cuda"
            assert next(model.parameters()).device.type == "cuda"

    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_training_modeself, mock_bert, mock_config:
        """Test model behavior in training mode."""
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance

        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifiernum_emotions=28
        model.train()

        # Test training mode
        assert model.training
        assert model.classifier.training

        # Test with sample class weights
        class_weights = torch.ones28
        model = BERTEmotionClassifiernum_emotions=28, class_weights=class_weights

        # The classifier layers should still have parameters
        assert hasattrmodel.classifier, "0"
        assert hasattrmodel.classifier, "3"

        # The model has dropout within the classifier, not as a direct attribute
        assert not hasattrmodel, "dropout"

    def test_class_weights_handlingself:
        """Test that class weights are handled correctly."""
        with patch"transformers.AutoConfig.from_pretrained", patch(
            "transformers.AutoModel.from_pretrained"
        ):
            class_weights = torch.tensor[1.0, 2.0, 3.0, 4.0]
            model = BERTEmotionClassifiernum_emotions=4, class_weights=class_weights

            # Verify class weights are stored
            assert hasattrmodel, "class_weights"
            assert torch.equalmodel.class_weights, class_weights

    @pytest.mark.slow
    @patch"transformers.AutoConfig.from_pretrained"
    @patch"transformers.AutoModel.from_pretrained"
    def test_emotion_label_mappingself, mock_bert, mock_config:
        """Test emotion label mapping functionality."""
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 768
        mock_config.return_value = mock_config_instance

        mock_bert_instance = MagicMock()
        mock_bert.return_value = mock_bert_instance

        model = BERTEmotionClassifiernum_emotions=28

        # Test that emotion labels are available
        assert hasattrmodel, "emotion_labels"
        assert lenmodel.emotion_labels == 28

        # Test emotion label mapping
        test_labels = ["joy", "sadness", "anger", "fear"]
        model.emotion_labels = test_labels

        with patch.objectmodel, "forward" as mock_forward:
            mock_logits = torch.tensor[[0.1, 0.8, 0.2, 0.9]]
            mock_forward.return_value = mock_logits

            predictions = model.predict_emotions(
                texts=["test text"],  # Add required texts parameter
                input_ids=torch.tensor[[1, 2, 3]],
                attention_mask=torch.tensor[[1, 1, 1]],
                threshold=0.5,
            )

            # Test that predictions align with labels
            assert lenpredictions[0] == lentest_labels
