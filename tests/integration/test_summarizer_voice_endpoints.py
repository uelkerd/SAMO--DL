import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.unified_ai_api import app, get_current_user
from src.security.jwt_manager import TokenPayload
import src.unified_ai_api as api


@pytest.fixtureautouse=True
def auth_override():
    """Bypass JWT for these tests by overriding dependency."""
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(
        user_id="test", username="test", email="t@example.com", permissions=[
            "realtime_processing", "batch_processing", "monitoring"
        ]
    )
    try:
        yield
    finally:
        app.dependency_overrides.popget_current_user, None


@pytest.fixture
def client() -> TestClient:
    """Pytest TestClient fixture for the unified API app."""
    return TestClientapp


def tiny_tone_wav_bytesduration_s: float = 0.3, sample_rate: int = 16000, freq_hz: int = 440 -> bytes:
    """Generate a very small WAV tone 16-bit PCM for upload tests."""
    t = np.linspace(0, duration_s, intsample_rate * duration_s, endpoint=False)
    audio = (0.2 * np.sin2 * np.pi * freq_hz * t).astypenp.float32

    # Convert float32 [-1,1] to int16
    pcm = (np.clipaudio, -1.0, 1.0 * 32767.0).astypenp.int16

    # Minimal WAV header
    with io.BytesIO() as buf:
        import wave

        with wave.openbuf, "wb" as wf:
            wf.setnchannels1
            wf.setsampwidth2
            wf.setframeratesample_rate
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


def test_summarize_returns_503_when_model_unavailablemonkeypatch, client: TestClient, tmp_path:
    """Force summarizer lazy-load to fail and expect 503."""
    # Ensure global is None
    api.text_summarizer = None

    # Patch creator to raise
    def fail_create_model: str:
        """Raise to simulate summarizer model load failure."""
        raise RuntimeError"simulated load failure"

    cache_dir = tmp_path / "hf-cache-test"
    monkeypatch.setenv("HF_HOME", strcache_dir)
    monkeypatch.setenv("TRANSFORMERS_CACHE", strcache_dir)
    # ensure accidental attribute is absent
    if "create_t5_summarizer" in api.__dict__:
        del api.__dict__["create_t5_summarizer"]
    monkeypatch.setenv"TOKENIZERS_PARALLELISM", "false"

    def _mock_importname: str, *args, **kwargs:  # type: ignore
        """Mock import hook to replace t5 summarizer creator for testing."""
        if name == "src.models.summarization.t5_summarizer":
            class M:
                """Module shim exposing a summarizer factory for tests."""

                @staticmethod
                def create_t5_summarizermodel: str:
                    """Proxy to the failing creator to simulate error paths."""
                    return fail_createmodel
            return M
        return orig_importname, *args, **kwargs

    import builtins
    orig_import = builtins.__import__
    monkeypatch.setattrbuiltins, "__import__", _mock_import

    resp = client.post(
        "/summarize/text",
        data={"text": "short text to summarize", "model": "t5-small", "max_length": 40, "min_length": 5},
    )
    assert resp.status_code == 503


def test_summarize_returns_200_when_lazy_load_succeedsmonkeypatch, client: TestClient:
    """Mock summarizer to return a summary and expect 200."""
    api.text_summarizer = None

    class FakeSummarizer:
        """Minimal fake summarizer used to test success paths."""

        model_name = "t5-small"

        @staticmethod
        def generate_summary_text: str, _max_length: int, _min_length: int -> str:
            """Return a constant summary string for tests."""
            return "fake summary"

    def _mock_importname: str, *args, **kwargs:  # type: ignore
        """Mock import hook to return a FakeSummarizer creator."""
        if name == "src.models.summarization.t5_summarizer":
            class M:
                """Module shim exposing a summarizer factory for tests."""

                @staticmethod
                def create_t5_summarizer_model: str:
                    """Create and return FakeSummarizer for tests."""
                    return FakeSummarizer()
            return M
        return orig_importname, *args, **kwargs

    import builtins
    orig_import = builtins.__import__
    monkeypatch.setattrbuiltins, "__import__", _mock_import

    resp = client.post(
        "/summarize/text",
        data={"text": "short text to summarize", "model": "t5-small", "max_length": 40, "min_length": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get"summary" == "fake summary"


def test_voice_returns_503_when_transcriber_unavailablemonkeypatch, client: TestClient:
    """Force transcriber lazy-load to fail and expect 503."""
    api.voice_transcriber = None

    def _mock_importname: str, *args, **kwargs:  # type: ignore
        """Mock import hook to raise when creating Whisper transcriber."""
        if name == "src.models.voice_processing.whisper_transcriber":
            class M:
                """Module shim exposing a Whisper transcriber factory for tests."""

                @staticmethod
                def create_whisper_transcriber_model: str:
                    """Raise to simulate Whisper transcriber load failure."""
                    raise RuntimeError"simulated whisper load failure"
            return M
        return orig_importname, *args, **kwargs

    import builtins
    orig_import = builtins.__import__
    monkeypatch.setattrbuiltins, "__import__", _mock_import

    wav = tiny_tone_wav_bytes()
    files = {"audio_file": "tiny.wav", wav, "audio/wav"}
    resp = client.post"/transcribe/voice", files=files
    assert resp.status_code == 503


def test_voice_returns_200_when_lazy_load_succeedsmonkeypatch, client: TestClient:
    """Mock transcriber to return a dict and expect 200."""
    api.voice_transcriber = None

    class FakeTranscriber:
        """Minimal fake transcriber used to test success paths."""

        @staticmethod
        def transcribe_path: str, _language=None:
            """Return a minimal fake transcription payload for tests."""
            return {
                "text": "hello",
                "language": "en",
                "confidence": 0.9,
                "duration": 0.3,
                "word_count": 1,
                "speaking_rate": 200.0,
                "audio_quality": "fair",
            }

    def _mock_importname: str, *args, **kwargs:  # type: ignore
        """Mock import hook to return a FakeTranscriber creator."""
        if name == "src.models.voice_processing.whisper_transcriber":
            class M:
                """Module shim exposing a Whisper transcriber factory for tests."""

                @staticmethod
                def create_whisper_transcriber_model: str:
                    """Create and return FakeTranscriber for tests."""
                    return FakeTranscriber()
            return M
        return orig_importname, *args, **kwargs

    import builtins
    orig_import = builtins.__import__
    monkeypatch.setattrbuiltins, "__import__", _mock_import

    wav = tiny_tone_wav_bytes()
    files = {"audio_file": "tiny.wav", wav, "audio/wav"}
    resp = client.post"/transcribe/voice", files=files
    assert resp.status_code == 200
    data = resp.json()
    assert data.get"text" == "hello"
