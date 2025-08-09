#!/usr/bin/env python3
"""
Sequential smoke test for local and Cloud Run deployments.

Usage:
  BASE_URL="http://127.0.0.1:8000" python scripts/testing/smoke_local.py
  python scripts/testing/smoke_local.py --base-url https://YOUR-SERVICE-url.run.app

Notes:
- Sets User-Agent: testclient to bypass local rate limiting.
- Mints a short-lived elevated JWT for endpoints requiring special permissions.
- Keeps output minimal: endpoint, status, brief detail.
"""

import argparse
import io
import json
import os
import time
import wave
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Tuple

import numpy as np
import requests
import asyncio  # ensure available for ws async path

# WebSocket client: prefer websocket-client if available; fallback to websockets (async)
try:
    import websocket  # type: ignore
    WEBSOCKET_BACKEND = "websocket-client"
except Exception:
    websocket = None  # type: ignore
    WEBSOCKET_BACKEND = "websockets"

# websockets backend is optional; ensure symbol exists for checks
try:
    import websockets  # type: ignore
except Exception:
    websockets = None  # type: ignore

try:
    import jwt  # PyJWT
except ImportError as import_err:
    print("PyJWT is required to mint a local elevated token; install PyJWT.")
    raise


def tiny_tone_wav_bytes(
    duration_s: float = 0.3, sample_rate: int = 16000, freq_hz: int = 440
) -> bytes:
    """Generate a tiny in-memory WAV tone for voice endpoint tests."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    # Convert to 16-bit PCM
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def mint_elevated_dev_token(secret: str, minutes: int = 5) -> str:
    """Mint a short-lived elevated JWT for endpoints needing extra permissions."""
    now = datetime.utcnow()
    payload = {
        "user_id": "user_test_elevated",
        "username": "dev",
        "email": "dev@example.com",
        "permissions": [
            "read",
            "write",
            "admin",
            "monitoring",
            "batch_processing",
            "realtime_processing",
        ],
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def p(endpoint: str, status: Optional[int], msg: str):
    """Print a minimal one-line result for each endpoint call."""
    print(f"{endpoint} -> {status} {msg}")


def phase_basic_gets(session: requests.Session, url: Callable[[str], str], pause: Callable[[], None]) -> None:
    """Check basic unauthenticated endpoints return 200 and brief status."""
    for ep in ["/", "/health", "/models/status"]:
        try:
            r = session.get(url(ep), timeout=10)
            brief = ""
            try:
                data = r.json()
                brief = data.get("status") or data.get("message") or "ok"
            except Exception:
                brief = r.text[:60]
            p(ep, r.status_code, brief)
        except Exception as e:
            p(ep, None, f"error: {e}")
        pause()


def login_and_get_access_token(
    session: requests.Session, url: Callable[[str], str]
) -> Optional[str]:
    """Attempt login and return an access token; return None on failure."""
    try:
        r = session.post(
            url("/auth/login"),
            json={"username": "tester@example.com", "password": "secret123"},
            timeout=10,
        )
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json().get("access_token")
    except Exception:
        pass
    return None


def phase_auth_login_refresh_logout(
    session: requests.Session, url: Callable[[str], str], pause: Callable[[], None]
) -> Tuple[Optional[str], Optional[str]]:
    """Exercise login, profile, refresh, and logout flow sequentially."""
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    try:
        r = session.post(
            url("/auth/login"),
            json={"username": "tester@example.com", "password": "secret123"},
            timeout=10,
        )
        data = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        p("/auth/login", r.status_code, "token" if access_token else r.text[:60])
    except Exception as exc:
        p("/auth/login", None, f"error: {exc}")
    pause()
    if access_token:
        try:
            r = session.get(
                url("/auth/profile"),
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            msg = "ok"
            try:
                msg = r.json().get("username", "ok")
            except Exception:
                pass
            p("/auth/profile", r.status_code, msg)
        except Exception as exc:
            p("/auth/profile", None, f"error: {exc}")
        pause()
        if refresh_token:
            try:
                r = session.post(
                    url("/auth/refresh"),
                    json={"refresh_token": refresh_token},
                    timeout=10,
                )
                new = (
                    r.json()
                    if r.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {}
                )
                access_token = new.get("access_token", access_token)
                p(
                    "/auth/refresh",
                    r.status_code,
                    "refreshed" if new.get("access_token") else r.text[:60],
                )
            except Exception as exc:
                p("/auth/refresh", None, f"error: {exc}")
            pause()
        try:
            r = session.post(
                url("/auth/logout"),
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            p("/auth/logout", r.status_code, "logged out")
        except Exception as exc:
            p("/auth/logout", None, f"error: {exc}")
        access_token = None
        pause()
    return access_token, refresh_token


def phase_analyze_journal(
    session: requests.Session,
    url: Callable[[str], str],
    pause: Callable[[], None],
) -> None:
    """POST a small journal entry and report primary emotion."""
    try:
        r = session.post(
            url("/analyze/journal"),
            json={
                "text": "A tiny note for quick smoke.",
                "generate_summary": False,
            },
            timeout=20,
        )
        msg = "ok"
        try:
            d = r.json()
            msg = d.get("emotion_analysis", {}).get("primary_emotion", "ok")
        except Exception:
            msg = r.text[:60]
        p("/analyze/journal", r.status_code, msg)
    except Exception as exc:
        p("/analyze/journal", None, f"error: {exc}")
    pause()


def phase_summarize_text(
    session: requests.Session,
    url: Callable[[str], str],
    pause: Callable[[], None],
    access_token: Optional[str],
) -> None:
    """POST summarization; if service unavailable, report it."""
    if not access_token:
        access_token = login_and_get_access_token(session, url)
    if access_token:
        try:
            form = {
                "text": "This is a very small text that should summarize okay.",
                "model": "t5-small",
                "max_length": 40,
                "min_length": 5,
            }
            r = session.post(
                url("/summarize/text"),
                headers={"Authorization": f"Bearer {access_token}"},
                data=form,
                timeout=20,
            )
            if r.status_code == 503:
                p("/summarize/text", r.status_code, "service unavailable")
            else:
                msg = "ok"
                try:
                    msg = (r.json().get("summary", "ok") or "ok")[:40]
                except Exception:
                    msg = r.text[:60]
                p("/summarize/text", r.status_code, msg)
        except Exception as exc:
            p("/summarize/text", None, f"error: {exc}")
        pause()


def phase_transcribe_voice(
    session: requests.Session,
    url: Callable[[str], str],
    pause: Callable[[], None],
    access_token: Optional[str],
    wav1: bytes,
) -> None:
    """POST single-file transcription using a tiny WAV."""
    if not access_token:
        access_token = login_and_get_access_token(session, url)
    if access_token:
        try:
            files = {"audio_file": ("tiny.wav", wav1, "audio/wav")}
            r = session.post(
                url("/transcribe/voice"),
                headers={"Authorization": f"Bearer {access_token}"},
                files=files,
                data={"language": "en"},
                timeout=30,
            )
            msg = "ok"
            try:
                if r.status_code == 503:
                    msg = "service unavailable"
                else:
                    msg = (r.json().get("text", "ok") or "ok")[:40]
            except Exception:
                msg = r.text[:60]
            p("/transcribe/voice", r.status_code, msg)
        except Exception as exc:
            p("/transcribe/voice", None, f"error: {exc}")
        pause()


def phase_batch_transcribe(
    session: requests.Session,
    url: Callable[[str], str],
    pause: Callable[[], None],
    elevated: str,
    wav1: bytes,
    wav2: bytes,
) -> None:
    """Call batch transcription with two tiny WAVs and report successes."""
    try:
        files = [
            ("audio_files", ("a.wav", wav1, "audio/wav")),
            ("audio_files", ("b.wav", wav2, "audio/wav")),
        ]
        r = session.post(
            url("/transcribe/batch"),
            headers={"Authorization": f"Bearer {elevated}"},
            files=files,
            timeout=45,
        )
        msg = "ok"
        try:
            msg = f"ok:{r.json().get('successful_transcriptions', 0)}"
        except Exception:
            msg = r.text[:60]
        p("/transcribe/batch", r.status_code, msg)
    except Exception as exc:
        p("/transcribe/batch", None, f"error: {exc}")
    pause()


def phase_monitoring(
    session: requests.Session,
    url: Callable[[str], str],
    pause: Callable[[], None],
    elevated: str,
) -> None:
    """Fetch monitoring endpoints and print their status fields."""
    for ep in ["/monitoring/performance", "/monitoring/health/detailed"]:
        try:
            r = session.get(
                url(ep),
                headers={"Authorization": f"Bearer {elevated}"},
                timeout=15,
            )
            brief = "ok"
            try:
                brief = r.json().get("status", "ok")
            except Exception:
                brief = r.text[:60]
            p(ep, r.status_code, brief)
        except Exception as exc:
            p(ep, None, f"error: {exc}")
        pause()


def phase_websocket(
    base_url: str,
    url: Callable[[str], str],
    elevated: str,
    wav1: bytes,
) -> None:
    """Attempt a minimal WS exchange if HTTPS → WSS; otherwise print skipped."""
    if base_url.startswith("https://"):
        ws_url = (
            url("/ws/realtime").replace("https://", "wss://")
            + f"?token={elevated}"
        )
    else:
        ws_url = None
    if ws_url and websocket is not None and WEBSOCKET_BACKEND == "websocket-client":
        try:
            ws = websocket.create_connection(  # type: ignore
                ws_url, timeout=10, header=["User-Agent: testclient"]
            )
            ws.send_binary(wav1)
            raw = ws.recv()
            ws.close()
            msg = "ok"
            try:
                jd = json.loads(raw)
                msg = jd.get("type", "ok")
            except Exception:
                msg = str(raw)[:40]
            p("WS /ws/realtime", 101, msg)
        except Exception as exc:
            p("WS /ws/realtime", None, f"error: {exc}")
    elif ws_url and WEBSOCKET_BACKEND == "websockets" and websockets is not None:
        async def ws_run():
            try:
                async with websockets.connect(
                    ws_url, extra_headers={"User-Agent": "testclient"}
                ) as ws:
                    await ws.send(wav1)
                    raw = await ws.recv()
                    msg = "ok"
                    try:
                        jd = json.loads(raw)
                        msg = jd.get("type", "ok")
                    except Exception:
                        msg = str(raw)[:40]
                    p("WS /ws/realtime", 101, msg)
            except Exception as exc:
                p("WS /ws/realtime", None, f"error: {exc}")

        asyncio.run(ws_run())
    else:
        reason = (
            "skipped: base_url is not https"
            if not ws_url
            else "skipped: no websocket client available"
        )
        p("WS /ws/realtime", None, reason)


def run_smoke(base_url: str, pause_ms: int = 200):
    """Run the sequential smoke test against the provided base URL."""
    headers = {"User-Agent": "testclient", "Accept": "application/json"}
    session = requests.Session()
    session.headers.update(headers)

    def url(path: str) -> str:
        """Build a full URL from base and relative path."""
        return base_url.rstrip("/") + path

    # Discover server secret for local signing (use default if env not set)
    jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

    def pause():
        """Sleep briefly between requests to reduce rate-limit noise."""
        if pause_ms and pause_ms > 0:
            time.sleep(pause_ms / 1000.0)

    # Run phases
    phase_basic_gets(session, url, pause)

    access_token, refresh_token = phase_auth_login_refresh_logout(session, url, pause)

    phase_analyze_journal(session, url, pause)

    # Phase 4: Summarize text (auth required)
    # Ensure we have a valid token for protected endpoints
    phase_summarize_text(session, url, pause, access_token)

    # Prepare tiny wavs
    wav1 = tiny_tone_wav_bytes()
    wav2 = tiny_tone_wav_bytes(freq_hz=554)

    # Phase 5: Transcribe voice (auth required)
    phase_transcribe_voice(session, url, pause, access_token, wav1)

    # Elevated token for batch + monitoring + WS
    elevated = mint_elevated_dev_token(jwt_secret)

    phase_batch_transcribe(session, url, pause, elevated, wav1, wav2)

    phase_monitoring(session, url, pause, elevated)

    # Phase 8: WebSocket realtime (elevated)
    phase_websocket(base_url, url, elevated, wav1)


def main():
    """CLI entrypoint: parse args and run smoke tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=os.getenv("BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--pause-ms",
        dest="pause_ms",
        type=int,
        default=int(os.getenv("SMOKE_PAUSE_MS", "200")),
    )
    args = parser.parse_args()

    run_smoke(args.base_url, pause_ms=args.pause_ms)


if __name__ == "__main__":
    main()
