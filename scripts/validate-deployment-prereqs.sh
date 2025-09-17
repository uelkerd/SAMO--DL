#!/bin/bash

# SAMO-DL Pre-flight Deployment Validation Script
# Prevents "this kinda shit" from happening by validating ALL prerequisites

set -e

echo "🚀 SAMO-DL Deployment Pre-flight Validation"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Validation results
VALIDATION_ERRORS=0

print_check() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_info() {
    echo -e "ℹ️  $1"
}

# 1. Check Google Cloud Authentication
echo
echo "1. Checking Google Cloud Authentication..."
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q ".*"; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    print_check "Authenticated as: $ACCOUNT"
else
    print_error "Not authenticated to Google Cloud. Run: gcloud auth login"
fi

# 2. Check Project Configuration
echo
echo "2. Checking Project Configuration..."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [[ -n "$PROJECT_ID" ]]; then
    print_check "Project set to: $PROJECT_ID"
else
    print_error "No project configured. Run: gcloud config set project YOUR_PROJECT_ID"
fi

# 3. Check Required APIs
echo
echo "3. Checking Required Google Cloud APIs..."

REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "secretmanager.googleapis.com"
)

OPTIONAL_APIS=(
    "ondemandscanning.googleapis.com"
    "containerscanning.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_check "Required API enabled: $api"
    else
        print_error "Required API NOT enabled: $api"
        echo "         Enable with: gcloud services enable "$api""
    fi
done

for api in "${OPTIONAL_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_check "Optional API enabled: $api"
    else
        print_warning "Optional API NOT enabled: $api (vulnerability scanning will be disabled)"
    fi
done

# 4. Check Artifact Registry Repository
echo
echo "4. Checking Artifact Registry Repository..."
REPO_NAME="samo-dl"
REPO_LOCATION="us-central1"

if gcloud artifacts repositories describe "$REPO_NAME" --location="$REPO_LOCATION" >/dev/null 2>&1; then
    print_check "Artifact Registry repository exists: $REPO_NAME"

    # Check repository permissions
    if gcloud artifacts repositories get-iam-policy "$REPO_NAME" --location="$REPO_LOCATION" >/dev/null 2>&1; then
        print_check "Have access to repository: $REPO_NAME"
    else
        print_error "No access to repository: $REPO_NAME"
    fi
else
    print_error "Artifact Registry repository NOT found: $REPO_NAME"
    echo "         Create with: gcloud artifacts repositories create \"$REPO_NAME\" --repository-format=docker --location=\"$REPO_LOCATION\""
fi

# 5. Check Required Secrets
echo
echo "5. Checking Required Secrets..."
REQUIRED_SECRETS=(
    "SAMO_EMOTION_API_KEY"
)

for secret in "${REQUIRED_SECRETS[@]}"; do
    if gcloud secrets describe "$secret" >/dev/null 2>&1; then
        print_check "Secret exists: $secret"

        # Check if we can access the latest version
        if gcloud secrets versions access latest --secret="$secret" >/dev/null 2>&1; then
            print_check "Can access secret: $secret"
        else
            print_error "Cannot access secret: $secret (check IAM permissions)"
        fi
    else
        print_error "Secret NOT found: $secret"
        echo "         Create with: gcloud secrets create "$secret" --data-file=<path-to-secret-file>"
    fi
done

# 6. Check Cloud Build Configuration Files
echo
echo "6. Checking Build Configuration Files..."
BUILD_CONFIGS=(
    "cloudbuild-samo-unified-api.yaml"
    "Dockerfile.optimized"
)

for config in "${BUILD_CONFIGS[@]}"; do
    if [[ -f "$config" ]]; then
        print_check "Build config exists: $config"

        # Validate YAML syntax for build configs
        if [[ "$config" == *.yaml ]] || [[ "$config" == *.yml ]]; then
            if python3 -c "import yaml; yaml.safe_load(open('$config', 'r'))" 2>/dev/null; then
                print_check "YAML syntax valid: $config"
            else
                print_error "YAML syntax invalid: $config"
            fi
        fi
    else
        print_error "Build config NOT found: $config"
    fi
done

# 7. Check Cloud Build Permissions
echo
echo "7. Checking Cloud Build Permissions..."
if gcloud builds list --limit=1 >/dev/null 2>&1; then
    print_check "Can access Cloud Build"
else
    print_error "Cannot access Cloud Build (check IAM permissions)"
fi

# 8. Check Cloud Run Permissions
echo
echo "8. Checking Cloud Run Permissions..."
if gcloud run services list --region=us-central1 >/dev/null 2>&1; then
    print_check "Can access Cloud Run in us-central1"
else
    print_error "Cannot access Cloud Run in us-central1 (check IAM permissions)"
fi

# 9. Check Docker Configuration
echo
echo "9. Checking Docker Configuration..."
if command -v docker >/dev/null 2>&1; then
    print_check "Docker is installed"

    if docker info >/dev/null 2>&1; then
        print_check "Docker daemon is running"
    else
        print_warning "Docker daemon not running (Cloud Build will handle this)"
    fi
else
    print_warning "Docker not installed locally (Cloud Build will handle this)"
fi

# Summary
echo
echo "=========================================="
if [[ "$VALIDATION_ERRORS" -eq 0 ]]; then
    print_check "ALL VALIDATIONS PASSED! Ready for deployment."
    echo
    echo "Deploy with:"
    echo "  gcloud builds submit --config cloudbuild-samo-unified-api.yaml ."
    echo
    exit 0
else
    print_error "$VALIDATION_ERRORS validation error(s) found!"
    echo
    echo "Fix the errors above before deploying to prevent failures."
    echo
    exit 1
fi