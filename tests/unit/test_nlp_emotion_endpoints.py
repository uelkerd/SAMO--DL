#!/usr/bin/env python3
import os
import json
import unittest
from unittest.mock import patch

# Ensure app import path works
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from deployment.secure_api_server import app  # type: ignore


def _fake_pipeline(*args, **kwargs):
    def _call(inputs, truncation=True):
        inputs_list = [inputs] if isinstance(inputs, str) else inputs
        dist = [
            {"label": "anger", "score": 0.01},
            {"label": "disgust", "score": 0.01},
            {"label": "fear", "score": 0.02},
            {"label": "joy", "score": 0.90},
            {"label": "neutral", "score": 0.03},
            {"label": "sadness", "score": 0.02},
            {"label": "surprise", "score": 0.01},
        ]
        return [dist for _ in inputs_list]
    return _call


class TestNlpEmotionEndpoints(unittest.TestCase):
    def setUp(self):
        os.environ['EMOTION_PROVIDER'] = 'hf'
        self.client = app.test_client()

    @patch('src.inference.text_emotion_service.pipeline', new=_fake_pipeline)
    def test_single_emotion_endpoint(self):
        payload = {"text": "I love this!"}
        resp = self.client.post('/nlp/emotion', data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('scores', data)
        self.assertEqual(data['provider'], 'hf')
        self.assertTrue(any(x['label'] == 'joy' for x in data['scores']))

    @patch('src.inference.text_emotion_service.pipeline', new=_fake_pipeline)
    def test_batch_emotion_endpoint(self):
        payload = {"texts": ["I love this!", "This is bad."]}
        resp = self.client.post('/nlp/emotion/batch', data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('results', data)
        self.assertEqual(data['count'], 2)
        self.assertEqual(data['provider'], 'hf')
        first, second = data['results']
        self.assertIn('scores', first)
        self.assertTrue(any(x['label'] == 'joy' for x in first['scores']))
        self.assertIn('scores', second)
        self.assertTrue(any(x['label'] == 'joy' for x in second['scores']))

    def test_invalid_payloads(self):
        resp = self.client.post('/nlp/emotion', data='{}', headers={'Content-Type': 'application/json'})
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post('/nlp/emotion/batch', data='{"texts": 123}', headers={'Content-Type': 'application/json'})
        self.assertEqual(resp.status_code, 400)


if __name__ == '__main__':
    unittest.main()