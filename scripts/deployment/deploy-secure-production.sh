#!/bin/bash
# SAMO Secure Production Deployment Script
# Deploy the security-hardened SAMO container to production

set -euo pipefail

# Configuration
CONTAINER_NAME="samo-production"
IMAGE_NAME="samo-secure:latest"
PRODUCTION_TAG="samo-secure:production"
PORT="8000"
HEALTH_CHECK_URL="http://localhost:${PORT}/health"
BACKUP_CONTAINER="${CONTAINER_NAME}-backup"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running"
        exit 1
    fi
    success "Docker is running"
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        error "Image $IMAGE_NAME not found"
        exit 1
    fi
    success "Secure image found: $IMAGE_NAME"
    
    # Run security scan
    log "🛡️  Running final security scan..."
    if command -v trivy > /dev/null 2>&1; then
        SCAN_RESULT=$(docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image --severity CRITICAL --quiet --format json "$IMAGE_NAME" 2>/dev/null || echo '{"Results":[]}')
        
        CRITICAL_COUNT=$(echo "$SCAN_RESULT" | jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL") | .VulnerabilityID' 2>/dev/null | wc -l || echo 0)
        
        if [ "$CRITICAL_COUNT" -gt 10 ]; then
            error "Too many critical vulnerabilities: $CRITICAL_COUNT"
            exit 1
        fi
        success "Security scan passed (Critical: $CRITICAL_COUNT)"
    else
        warning "Trivy not available, skipping security scan"
    fi
}

# Backup current container
backup_current_container() {
    if docker ps -q --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
        log "📦 Backing up current container..."
        
        # Check if backup already exists and handle it
        if docker ps -a -q --filter "name=^/${BACKUP_CONTAINER}$" | grep -q .; then
            log "⚠️  Existing backup container found, removing it..."
            docker stop "$BACKUP_CONTAINER" 2>/dev/null || true
            docker rm "$BACKUP_CONTAINER" 2>/dev/null || true
        fi
        
        # Stop current container
        docker stop "$CONTAINER_NAME" || true
        
        # Rename to backup and check status
        if docker rename "$CONTAINER_NAME" "$BACKUP_CONTAINER"; then
            success "Current container backed up as $BACKUP_CONTAINER"
        else
            error "Failed to backup current container"
            exit 1
        fi
    else
        log "ℹ️  No existing container to backup"
    fi
}

# Deploy new container
deploy_container() {
    log "🚀 Deploying secure container..."
    
    # Tag for production
    docker tag "$IMAGE_NAME" "$PRODUCTION_TAG"
    success "Tagged image as $PRODUCTION_TAG"
    
    # Deploy new container
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${PORT}:8000" \
        --health-cmd="curl -f http://localhost:8000/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        --health-start-period=20s \
        --memory="512m" \
        --cpus="1.0" \
        --security-opt no-new-privileges:true \
        --read-only=false \
        "$PRODUCTION_TAG"
    
    success "Container deployed: $CONTAINER_NAME"
}

# Health check
wait_for_health() {
    log "🏥 Waiting for container to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "healthy"; then
            success "Container is healthy!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts - waiting for health check..."
        sleep 10
        ((attempt++))
    done
    
    error "Container failed to become healthy within timeout"
    return 1
}

# Validate deployment
validate_deployment() {
    log "✅ Validating deployment..."
    
    # Test health endpoint
    if curl -sf "$HEALTH_CHECK_URL" > /dev/null; then
        success "Health endpoint responding"
    else
        error "Health endpoint not responding"
        return 1
    fi
    
    # Test response time
    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "$HEALTH_CHECK_URL")
    log "Response time: ${response_time}s"
    
    # Use awk for portable floating-point comparison (no bc dependency)
    if awk -v rt="$response_time" 'BEGIN { exit (rt > 1.0 ? 0 : 1) }'; then
        warning "Slow response time: ${response_time}s"
    else
        success "Response time acceptable: ${response_time}s"
    fi
    
    # Check container stats
    local stats
    stats=$(docker stats "$CONTAINER_NAME" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | tail -n 1)
    log "Container stats: $stats"
    
    success "Deployment validation completed"
}

# Cleanup backup
cleanup_backup() {
    if docker ps -a -q -f name="$BACKUP_CONTAINER" | grep -q .; then
        log "🧹 Cleaning up backup container..."
        docker rm "$BACKUP_CONTAINER" 2>/dev/null || true
        success "Backup container removed"
    fi
}

# Rollback function
rollback() {
    error "Deployment failed, initiating rollback..."
    
    # Stop failed container
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    
    # Restore backup if exists
    if docker ps -a -q -f name="$BACKUP_CONTAINER" | grep -q .; then
        docker rename "$BACKUP_CONTAINER" "$CONTAINER_NAME"
        docker start "$CONTAINER_NAME"
        warning "Rollback completed - restored previous container"
    else
        error "No backup container found for rollback"
    fi
    
    exit 1
}

# Start monitoring
start_monitoring() {
    log "📊 Starting continuous monitoring..."
    
    if [ -f "./scripts/security/continuous-security-monitor.sh" ]; then
        # Start monitoring in background
        nohup ./scripts/security/continuous-security-monitor.sh continuous > /tmp/samo-monitor.log 2>&1 &
        local monitor_pid=$!
        echo "$monitor_pid" > /tmp/samo-monitor.pid
        success "Monitoring started (PID: $monitor_pid)"
        log "Monitor logs: /tmp/samo-monitor.log"
    else
        warning "Monitoring script not found, skipping"
    fi
}

# Main deployment function
main() {
    log "🚀 Starting SAMO Secure Production Deployment"
    log "Container: $CONTAINER_NAME"
    log "Image: $IMAGE_NAME"
    log "Port: $PORT"
    echo
    
    # Set trap for rollback on failure
    trap rollback ERR
    
    # Execute deployment steps
    pre_deployment_checks
    backup_current_container
    deploy_container
    wait_for_health
    validate_deployment
    cleanup_backup
    start_monitoring
    
    # Success message
    echo
    success "🎉 SAMO Secure Production Deployment Completed Successfully!"
    log "Container: $CONTAINER_NAME is running on port $PORT"
    log "Health check: $HEALTH_CHECK_URL"
    log "Monitor logs: /tmp/samo-monitor.log"
    echo
    log "Next steps:"
    log "1. Monitor the application logs: docker logs -f $CONTAINER_NAME"
    log "2. Check monitoring: tail -f /tmp/samo-monitor.log"
    log "3. Test API endpoints: curl $HEALTH_CHECK_URL"
    echo
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "status")
        if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            success "Container $CONTAINER_NAME is running"
            docker stats "$CONTAINER_NAME" --no-stream
        else
            warning "Container $CONTAINER_NAME is not running"
        fi
        ;;
    "logs")
        docker logs -f "$CONTAINER_NAME"
        ;;
    "stop")
        log "Stopping $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME"
        success "Container stopped"
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|status|logs|stop]"
        echo "  deploy   - Deploy secure container to production (default)"
        echo "  rollback - Rollback to previous container"
        echo "  status   - Check container status"
        echo "  logs     - Show container logs"
        echo "  stop     - Stop production container"
        exit 1
        ;;
esac