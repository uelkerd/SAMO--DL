# Surgical CI Debugging Plan

This guide documents a fast, low-noise approach to isolate and fix CI failures with minimal iteration time. It emphasizes short OODA loops, binary-search style instrumentation, and early-failure diagnostics.

## Goals

- Fail fast with high-signal diagnostics (environment, paths, interpreters, deps).
- Shrink the search space quickly using binary search instrumentation between steps.
- Keep changes small and reversible; prefer parameters/toggles over large edits.

## Method (OODA + Binary Search)

1. Observe
   - Add early checks in the very first job step:
     - `echo $SHELL`, `uname -a`, `whoami`, `pwd`, `ls -la`, `df -h`.
     - `env | sort | head -n 100` (never dump secrets).
     - Verify required files: `test -f environment.yml || (echo 'missing environment.yml' && exit 10)`.
   - After conda install: list conda bin and the active Python:
     - `ls -la "$HOME/miniconda/bin" || true`
     - `"$HOME/miniconda/bin/conda" info --envs || true`
     - `"$HOME/miniconda/bin/conda" list | head -n 50 || true`
     - `"$HOME/miniconda/bin/conda" run -n samo-dl-stable python -c "import sys; print(sys.executable)"`

2. Orient
   - Classify failures into: bootstrap (conda missing/paths), dependency gaps (packages not installed in the env actually running tests), cache poisoning (stale env reused), working-directory/path issues (files not found), or test-level errors.

3. Decide
   - Binary search steps: disable everything except “conda bootstrap + doctor”, then re-enable steps until failure reappears.
   - Toggle caches off temporarily. If green, reintroduce with a new cache key.
   - Rerun failing job with SSH to validate live: confirm conda path exists, `conda run` works, and pytest uses the expected interpreter.
   - Parameterize workflows to run only “bootstrap + doctor” for quick iteration.

4. Act
   - Add a lightweight “doctor” step at the start of every job.
   - Make all Python invocations explicit via conda-run to avoid PATH dependence:
     - `"$HOME/miniconda/bin/conda" run -n samo-dl-stable python ...`
   - Upload artifacts (doctor logs, conda info, pip freeze) on failure.

## Doctor step (snippet)

Use a single step or script that never blocks the pipeline and always prints the essentials.

```bash
#!/usr/bin/env bash
set -euxo pipefail

echo "SHELL=${SHELL}"
uname -a || true
whoami || true
echo "PATH=${PATH}"
pwd
ls -la
df -h || true
env | sort | head -n 100 || true

ls -la "$HOME/miniconda/bin" || true
"$HOME/miniconda/bin/conda" --version || true
"$HOME/miniconda/bin/conda" info --envs || true
"$HOME/miniconda/bin/conda" list | head -n 50 || true

"$HOME/miniconda/bin/conda" run -n samo-dl-stable python - <<'PY'
import sys
try:
    import jwt
    jwt_ver = getattr(jwt, "__version__", "n/a")
except Exception as exc:  # noqa: BLE001
    jwt_ver = f"import failed: {exc}"
print("python:", sys.executable)
print("jwt:", jwt_ver)
PY
```

## CircleCI examples (fragments)

Add an initial diagnostic step and store artifacts for later inspection:

```yaml
steps:
  - run:
      name: Doctor
      command: |
        bash -c 'set -euxo pipefail; echo "SHELL=$SHELL"; uname -a; whoami; pwd; ls -la; df -h || true'
        ls -la "$HOME/miniconda/bin" || true
        "$HOME/miniconda/bin/conda" --version || true
        "$HOME/miniconda/bin/conda" info --envs || true | tee doctor.conda.info.txt
        "$HOME/miniconda/bin/conda" list | tee doctor.conda.list.txt
        "$HOME/miniconda/bin/conda" run -n samo-dl-stable python -c "import sys, jwt; print(sys.executable); print(getattr(jwt, '__version__', 'n/a'))" | tee doctor.python_jwt.txt

  - store_artifacts:
      path: doctor.conda.info.txt
  - store_artifacts:
      path: doctor.conda.list.txt
  - store_artifacts:
      path: doctor.python_jwt.txt
```

Introduce a quick “bootstrap-only” workflow gated by a parameter to iterate in ~1 minute:

```yaml
parameters:
  bootstrap_only:
    type: boolean
    default: false

workflows:
  version: 2
  build-test:
    when: << pipeline.parameters.bootstrap_only >>
    jobs:
      - setup_python_env
      - unit_tests:
          requires:
            - setup_python_env
```

Temporarily disable caches to rule out poisoning; if green, bump cache keys (e.g., `v4-…` → `v5-…`).

## Rerun with SSH (when stuck)

Validate live:

```bash
ls -la "$HOME/miniconda/bin/conda"
"$HOME/miniconda/bin/conda" run -n samo-dl-stable python -c "import jwt; print(jwt.__version__)"
"$HOME/miniconda/bin/conda" run -n samo-dl-stable python -c "import sys; print(sys.executable)"
```

## References

- Socratic ducking and OODA loops: [LessWrong article](https://www.lesswrong.com/s/KAv8z6oJCTxjR8vdR/p/CJGKkTjWLGCwhkjGY)
- Unusually effective debugging (binary search mindset): [Carlos Bueno](https://carlos.bueno.org/2013/09/effective-debugging.html)
- Debugging via binary search: [Medium guide](https://medium.com/codecastpublication/debugging-tools-and-techniques-binary-search-2da5bb4282c7)
