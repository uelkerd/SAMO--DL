#!/bin/bash

# 🚀 Create Focused PRs Script
# This script creates focused PR branches from the massive cloud-run-deployment-focus branch

set -euo pipefail

echo "🚀 Creating focused PR branches..."

# PR 3: Deployment Scripts
echo "📦 Creating PR3: Deployment Scripts..."
git checkout -b pr3-deployment-scripts main
git checkout cloud-run-deployment-focus -- \
    scripts/deployment/convert_model_to_onnx.py \
    scripts/deployment/convert_model_to_onnx_simple.py \
    scripts/deployment/deploy_minimal_cloud_run.sh \
    scripts/deployment/deploy_onnx_cloud_run.sh \
    scripts/deployment/fix_model_loading_issues.py

git add .
git commit -m "📦 PR3: Deployment Scripts

- Add convert_model_to_onnx.py for model optimization
- Add convert_model_to_onnx_simple.py for simplified conversion
- Add deploy_minimal_cloud_run.sh for automated deployment
- Add deploy_onnx_cloud_run.sh for ONNX deployment
- Add fix_model_loading_issues.py for troubleshooting

This PR focuses on deployment automation and model optimization scripts."

# PR 4: Testing & Validation
echo "🧪 Creating PR4: Testing & Validation..."
git checkout -b pr4-testing-validation main
git checkout cloud-run-deployment-focus -- \
    scripts/testing/check_model_health.py \
    scripts/testing/debug_model_loading.py \
    scripts/testing/test_cloud_run_api_endpoints.py \
    scripts/testing/test_model_status.py

git add .
git commit -m "🧪 PR4: Testing & Validation

- Add check_model_health.py for model validation
- Add debug_model_loading.py for troubleshooting
- Add test_cloud_run_api_endpoints.py for API testing
- Add test_model_status.py for status monitoring

This PR focuses on comprehensive testing and validation infrastructure."

# PR 5: ONNX Optimization
echo "⚡ Creating PR5: ONNX Optimization..."
git checkout -b pr5-onnx-optimization main
git checkout cloud-run-deployment-focus -- \
    deployment/cloud-run/onnx_api_server.py \
    dependencies/requirements_onnx.txt

# Verify the file exists in the source branch to avoid a silent failure later
if ! git show cloud-run-deployment-focus:dependencies/requirements_onnx.txt > /dev/null 2>&1; then
  echo "❌ Missing dependencies/requirements_onnx.txt in branch cloud-run-deployment-focus" >&2
  exit 1
fi

git add .
git commit -m "⚡ PR5: ONNX Optimization

- Add onnx_api_server.py for optimized model serving
- Add requirements_onnx.txt for ONNX dependencies

This PR focuses on ONNX model optimization for improved performance."

echo "✅ All focused PR branches created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Push all branches: git push origin pr1-core-cloud-run-deployment pr2-documentation-updates pr3-deployment-scripts pr4-testing-validation pr5-onnx-optimization"
echo "2. Create GitHub PRs for each branch"
echo "3. Review and merge in order: PR1 → PR2 → PR3 → PR4 → PR5"
echo ""
echo "🎯 Remember: Each PR is now focused and reviewable (<1,200 lines each)"
