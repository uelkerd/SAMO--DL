#!/usr/bin/env python3
import os
import json
import time
import sys
from typing import List

import requests

HF_REPO = os.getenv("HF_REPO", "0xmnrv/samo")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

API_URL = f"https://api-inference.huggingface.co/models/{HF_REPO}"
HEADERS = {"Content-Type": "application/json"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

TEST_CASES: List[str] = [
    "I felt calm after writing it all down.",
    "I am frustrated but hopeful.",
    "Today was overwhelming but I'm proud of getting through it.",
    "",
]

def main() -> int:
    print(f"🔎 Serverless smoke test → {API_URL}")
    ok = True
    for text in TEST_CASES:
        payload = {"inputs": text}
        t0 = time.time()
        try:
            r = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
        except Exception as e:
            print(f"❌ Request failed for input='{text[:40]}...': {e}")
            ok = False
            continue
        dt = (time.time() - t0) * 1000
        print("—" * 40)
        print(f"Input: {repr(text)}")
        print(f"Status: {r.status_code} ({dt:.1f} ms)")
        try:
            print("Output:", r.json())
        except Exception:
            print("Raw:", r.text)
            ok = False
    print("—" * 40)
    print("✅ All requests succeeded" if ok else "⚠️ Some requests failed")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())