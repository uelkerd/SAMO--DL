#!/usr/bin/env python3
import os
import json
import time
import sys
from typing import List, Tuple

import requests

HF_REPO = os.getenv("HF_REPO", "0xmnrv/samo")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{HF_REPO}"
HEADERS = {"Content-Type": "application/json"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

TEST_CASES: List[Tuple[str, bool]] = [
    ("I felt calm after writing it all down.", True),
    ("I am frustrated but hopeful.", True),
    ("Today was overwhelming but I'm proud of getting through it.", True),
    ("", False),  # empty input may be accepted or return an error; don't enforce schema
]

MAX_RETRIES = 3
RETRY_WAIT_S = 10


def _is_valid_output_schema(obj) -> bool:
    # Accept either [ {label, score}, ... ] or [ [ {label, score}, ... ] ]
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    first = obj[0]
    if isinstance(first, dict):
        return "label" in first and "score" in first
    if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
        return "label" in first[0] and "score" in first[0]
    return False


def _post_with_retries(payload: dict) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
        except Exception as e:
            last_exc = e
            r = None
        else:
            # If model is still loading, HF returns 503 with estimated_time
            if r.status_code != 503:
                return r
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_S)
    if r is None and last_exc is not None:
        raise last_exc
    return r


def main() -> int:
    print(f"🔎 Serverless smoke test → {API_URL}")
    all_ok = True
    for text, enforce_schema in TEST_CASES:
        payload = {"inputs": text}
        t0 = time.time()
        r = _post_with_retries(payload)
        dt = (time.time() - t0) * 1000
        print("—" * 40)
        print(f"Input: {repr(text)}")
        print(f"Status: {r.status_code} ({dt:.1f} ms)")
        try:
            obj = r.json()
        except Exception:
            print("Raw:", r.text)
            obj = None

        # Status expectations: 200 is success. 503 acceptable only during warm-up (we retried above).
        if r.status_code != 200:
            print(f"❌ Unexpected status: {r.status_code}; body={obj}")
            all_ok = False
            continue

        if enforce_schema:
            if not _is_valid_output_schema(obj):
                print(f"❌ Unexpected schema: {obj}")
                all_ok = False
            else:
                print("✅ Schema OK")
        else:
            print("ℹ️  Skipped schema enforcement for edge case input")

    print("—" * 40)
    if not all_ok:
        print("⚠️ Some checks failed")
        return 1
    print("✅ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
