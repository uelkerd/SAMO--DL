# CircleCI Pipeline Debugging Prompt for SAMO Deep Learning Project

## Context
You are debugging a CircleCI pipeline for the SAMO Deep Learning project. The pipeline has 3 stages with quality gates and must run end-to-end without failures.

## Debugging Process

### 1. Initial Failure Analysis
When encountering a pipeline failure:

```
Please analyze the following CircleCI pipeline failure:

Pipeline: samo-ci-cd
Branch: [BRANCH_NAME]
Failed Job: [JOB_NAME]
Stage: [1/2/3]
Error Output:
"""
[PASTE ERROR LOGS HERE]
"""

Perform root cause analysis:
1. Identify the exact error message
2. Determine if it's a:
   - Dependency issue
   - Code quality failure (linting/formatting)
   - Test failure
   - Resource constraint
   - Configuration error
   - Model loading/inference issue
3. Check for patterns in similar failures
```

### 2. Systematic Debug Approach

#### For Each Failed Job:

**A. Stage 1 Failures (Fast Feedback)**
```
If lint-and-format fails:
- Check Ruff configuration in pyproject.toml
- Verify import sorting rules
- Check for formatting inconsistencies
- Run locally: ruff check src/ tests/ scripts/

If unit-tests fail:
- Check test coverage threshold (currently 5%, target 70%)
- Verify PYTHONPATH is set correctly
- Check for missing test dependencies
- Examine pytest configuration
```

**B. Stage 2 Failures (Integration & Security)**
```
If security-scan fails:
- Review Bandit security findings
- Check for hardcoded secrets
- Verify dependency vulnerabilities with Safety

If model-validation fails:
- Check model file paths
- Verify HuggingFace cache permissions
- Test CUDA availability (if GPU job)
- Validate model loading scripts:
  * bert_model_test.py
  * model_calibration_test.py
  * model_compression_test.py
  * onnx_conversion_test.py
  * t5_summarization_test.py
```

**C. Stage 3 Failures (Comprehensive Testing)**
```
If e2e-tests fail:
- Check timeout settings (current: 300s)
- Verify API endpoints are accessible
- Review integration points

If performance-benchmarks fail:
- Check response time thresholds (CI: <2s, Prod target: <500ms)
- Verify resource allocation
- Review memory usage
```

### 3. Iterative Fix Protocol

For each identified issue:

1. **Hypothesis Formation**
   ```
   Based on error: [ERROR]
   Hypothesis: [WHAT YOU THINK IS WRONG]
   Evidence: [SUPPORTING FACTS FROM LOGS/CODE]
   ```

2. **Solution Implementation**
   ```
   Fix Type: [code/config/dependency/resource]
   Files to modify:
   - [FILE_PATH]
   
   Changes needed:
   [SPECIFIC CHANGES]
   ```

3. **Validation Steps**
   ```
   Local test command: [COMMAND]
   Expected outcome: [WHAT SHOULD HAPPEN]
   CircleCI job to re-run: [JOB_NAME]
   ```

### 4. Common Issues Checklist

- [ ] All dependencies in pyproject.toml are compatible
- [ ] System dependencies (portaudio19-dev, python3-pyaudio) installed
- [ ] PYTHONPATH correctly set to include src/
- [ ] Model files accessible in CI environment
- [ ] Sufficient resources allocated (especially for GPU jobs)
- [ ] API rate limiting not causing timeouts
- [ ] Docker layer caching enabled for faster builds
- [ ] Test database/fixtures properly initialized

### 5. Monitoring Progress

Track resolution progress:
```
Pipeline Run #[NUMBER]
Fixed Issues:
- ✅ [ISSUE 1]
- ✅ [ISSUE 2]

Remaining Issues:
- ❌ [ISSUE 3] - [STATUS]
- ❌ [ISSUE 4] - [STATUS]

Next Steps:
1. [IMMEDIATE ACTION]
2. [FOLLOW-UP ACTION]
```

### 6. Final Validation

Once all jobs pass:
```
✅ Stage 1: All quality checks pass
✅ Stage 2: Security clean, models validated
✅ Stage 3: Performance meets targets
✅ Deployment: Docker image builds successfully

Lessons Learned:
- [KEY INSIGHT 1]
- [KEY INSIGHT 2]

Documentation Updates Needed:
- [UPDATE 1]
- [UPDATE 2]
```

## Usage Instructions

1. Copy this template
2. Fill in the placeholders with actual error information
3. Work through each section systematically
4. Document all changes made
5. Re-run failed jobs after each fix
6. Continue until full pipeline passes

## Important Notes

- Focus only on Deep Learning track concerns
- Avoid scope creep into other tracks (Web Dev, UX)
- Maintain separation of concerns
- Document all fixes for future reference
- Update this template with new patterns discovered