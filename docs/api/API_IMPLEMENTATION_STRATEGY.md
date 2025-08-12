### SAMO-DL Secure API — Deep Learning Feature Implementation Strategy

This document is the single source of truth for implementing (and verifying) the API so it reflects ~100% of the deep learning capabilities in this repository. We will iterate against this plan until the API is feature-complete, stable, and production-ready.

---

### Objectives
- Deliver a single, consistent HTTP API for deep learning features (text, voice, analysis, monitoring).
- Maintain robust security (headers, rate limiting, input sanitization) and observability (logs, metrics, health).
- Provide reliable docs (`/docs`) sourced from a curated `openapi.yaml` (no dynamic Swagger generation pitfalls).
- Ensure fast cold starts via lazy model loading; degrade gracefully when optional components are missing.
- Back the API with comprehensive tests (unit, integration, smoke) to prevent regressions.

---

### Out of Scope (Handled by Core Backend)
- Authentication and user management (register, login, refresh, logout, profile) will be provided by the core backend.
  - Temporary stubbed endpoints may exist in this branch for testing but will be disabled/removed prior to production cutover.
- Role-based authorization beyond API-key header.

---

### Architecture Decisions
- Framework: Flask + Flask-RESTX (routing, OpenAPI models). Swagger UI served via blueprint + static `openapi.yaml`.
- Namespacing: All feature routes live under `/api/*`. Root `/` returns API info. Docs at `/docs`.
- Docs: Flask-RESTX Swagger UI disabled (instability). Instead: `docs_blueprint` serves `/docs` and `/openapi.yaml`.
- Model lifecycle: Lazy, thread-safe loading at first request. Errors are non-fatal at startup; endpoints return 503 if a component is unavailable.
- Security: API-key via `X-API-Key`; strict security headers; centralized input sanitization; rate limiting per minute (configurable via env).
- Observability: Structured logs; request IDs + duration headers; health/metrics endpoints; readiness through `/api/health`.
- Performance: CPU-first behavior with optional GPU. Keep summarization beams conservative to avoid long latencies on CPU.

---

### Endpoint Map (Scope and Status)
- Core Text (implemented)
  - GET `/api/health` — service and model readiness
  - POST `/api/predict` — single text emotion detection
  - POST `/api/predict_batch` — batch emotion detection
  - GET `/api/emotions` — supported labels

- Text Processing & Summarization (implemented)
  - POST `/api/summarize` — T5-based summary for input text
  - POST `/api/analyze/journal` — combined pipeline: emotion + optional summarization

- Voice Processing (implemented)
  - POST `/api/transcribe` — Whisper speech-to-text
  - POST `/api/transcribe_batch` — batch audio transcription
  - POST `/api/analyze/voice_journal` — transcription then journal analysis

- Monitoring & System (implemented)
  - GET `/api/monitoring/performance` — CPU/mem/disk snapshot
  - GET `/api/monitoring/health/detailed` — component-level health
  - GET `/api/models/status` — load status for each component

- Real-time/Streaming (planned)
  - WebSocket `/ws/realtime` for live audio processing (Design Decision pending):
    - Option A: Flask-Sock (Flask native)
    - Option B: Extract to a dedicated FastAPI/ASGI microservice (recommended for production scale)

- Documentation (implemented)
  - GET `/docs` — static UI pointing to `openapi.yaml`
  - GET `/openapi.yaml` — spec source

---

### Error and Response Contracts
- Unified error body:
  - `{ error: string, status_code: int, request_id: string, timestamp: float }`
- Success responses strictly follow schemas in `deployment/cloud-run/openapi.yaml`.
- All endpoints return appropriate HTTP status codes: 2xx (success), 4xx (client errors), 5xx (server errors), 503 (service unavailable for lazy components).

---

### Security & Limits
- API-Key required for all stateful/expensive endpoints via `X-API-Key`.
- Rate limiting per-IP and per-key (env-configurable), returning 429 on breach.
- Input sanitization on text fields; length caps via env; batching limits to prevent abuse.
- Security headers applied globally.

---

### Dependencies & Models
- Emotion: HF Transformers classification model; optional local fine-tuned weights when present.
- Summarization: T5 family (small by default), adjustable via request; beam search, conservative defaults.
- Voice: OpenAI Whisper; requires ffmpeg in runtime image.
- All model modules loaded lazily, guarded with detailed logs; endpoints degrade to 503 when not available.

---

### Testing Strategy
- Unit tests
  - Lazy loader behavior and error paths
  - Error helpers and input sanitization
- Integration tests
  - JSON endpoints: happy paths + missing headers + bad input
  - Multipart endpoints: smoke tests for single and batch uploads
  - Health and monitoring endpoints
- E2E (planned)
  - Real-time streaming mock
  - Long-text stress tests for summarization

Execution
- Local integration harness spins up Flask app on an ephemeral port during tests.
- CI to run unit + integration suites; allow 503 for optional components absent in CI.

---

### Deployment & Environments
- Docs blueprint defaults point to `deployment/cloud-run/openapi.yaml` for /docs.
- Cloud Run Dockerfiles available (with ffmpeg for Whisper). Build-time caching can pre-pull T5/Whisper weights in certain images.
- Env vars control limits and behavior (e.g., `ADMIN_API_KEY`, `RATE_LIMIT_PER_MINUTE`, `MODEL_PATH`, summarizer model override).

---

### Versioning & Compatibility
- Current base path: `/api/*` (no version segment). Future-compatible plan: alias under `/api/v1/*` once we stabilize.
- Backwards compatibility maintained across minor updates; breaking changes guarded behind new paths or feature flags.

---

### Work Plan (Backlog)
- Done
  - Stable `/docs` using blueprint and static `openapi.yaml`.
  - Implemented: summarize, journal analysis, transcribe, transcribe_batch, analyze/voice_journal.
  - Monitoring endpoints; model status; core emotion endpoints.
  - Lazy loaders and error-hardening; integration + unit tests; pytest server fixture.
- In Progress
  - Expand tests (edge cases, larger payloads, rate limit assertions, monitoring schema checks)
- Planned
  - Real-time streaming endpoint (flask-sock) or dedicated ASGI microservice
  - Performance tuning (thread pools, batching, caching hot tokenizers)
  - Observability: Prometheus metrics endpoint for Cloud Run scraping
  - Harden multipart validation (MIME checks, size guards) and audio pre-processing fallbacks
  - Document resource limits, timeouts, and SLA per endpoint
- Defer/Remove before Production
  - Stubbed auth endpoints (to be removed once core backend integration is wired)

---

### Definition of Done
- All listed endpoints implemented and documented in `openapi.yaml`.
- Green test suite locally and in CI for unit + integration (with optional components tolerated via 503).
- Manual smoke on `/docs`, `/api/health`, summarization, journal, voice endpoints.
- Cloud Run image build successful; deploy to staging; monitor health and latency.

---

### How We Work This Plan
- Keep this file updated as the single source of truth for this branch.
- Small, focused PRs; commit frequently; extend tests alongside changes.
- Defer non-essential features (auth suite) to the core backend integration.
- Prefer simple, robust solutions; remove temporary stubs when replaced.