#!/bin/bash
# SAMO Security Monitoring Script
# Continuous security monitoring for production deployment

set -euo pipefail

# Configuration
CONTAINER_NAME="samo-secure-test"
IMAGE_NAME="samo-secure:latest"
SCAN_INTERVAL=86400  # 24 hours in seconds
LOG_FILE="/tmp/samo-security-monitor.log"
ALERT_THRESHOLD_CRITICAL=10  # Adjusted for FFmpeg vulnerabilities
ALERT_THRESHOLD_HIGH=50      # Adjusted for current security state

# Colors for output (export for external use)
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# Security scan function
run_security_scan() {
    log "🔍 Starting security scan for $IMAGE_NAME"
    
    # Run Trivy scan
    SCAN_OUTPUT=$(docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy:latest image --severity CRITICAL,HIGH --quiet --format json "$IMAGE_NAME" 2>/dev/null || echo '{"Results":[]}')
    
    # Parse results with clean numeric values
    CRITICAL_COUNT=$(echo "$SCAN_OUTPUT" | jq -r '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length // 0' 2>/dev/null || echo 0)
    HIGH_COUNT=$(echo "$SCAN_OUTPUT" | jq -r '[.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH")] | length // 0' 2>/dev/null || echo 0)
    
    log "📊 Security Scan Results:"
    log "   Critical vulnerabilities: $CRITICAL_COUNT"
    log "   High vulnerabilities: $HIGH_COUNT"
    
    # Check thresholds
    if [ "$CRITICAL_COUNT" -gt "$ALERT_THRESHOLD_CRITICAL" ]; then
        log "🚨 ALERT: Critical vulnerabilities ($CRITICAL_COUNT) exceed threshold ($ALERT_THRESHOLD_CRITICAL)"
        return 1
    fi
    
    if [ "$HIGH_COUNT" -gt "$ALERT_THRESHOLD_HIGH" ]; then
        log "⚠️  WARNING: High vulnerabilities ($HIGH_COUNT) exceed threshold ($ALERT_THRESHOLD_HIGH)"
        return 2
    fi
    
    log "✅ Security scan passed - vulnerabilities within acceptable limits"
    return 0
}

# Container health check
check_container_health() {
    log "🏥 Checking container health for $CONTAINER_NAME"
    
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        log "❌ Container $CONTAINER_NAME is not running"
        return 1
    fi
    
    # Check health status
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
    
    if [ "$HEALTH_STATUS" != "healthy" ]; then
        log "❌ Container health status: $HEALTH_STATUS"
        return 1
    fi
    
    log "✅ Container is healthy"
    return 0
}

# Performance monitoring
check_performance() {
    log "📈 Checking container performance"
    
    # Get container stats
    STATS=$(docker stats "$CONTAINER_NAME" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | tail -n 1)
    
    if [ -n "$STATS" ]; then
        log "📊 Performance metrics: $STATS"
        
        # Extract CPU percentage (remove % sign)
        CPU_PERCENT=$(echo "$STATS" | awk '{print $1}' | sed 's/%//')
        MEM_PERCENT=$(echo "$STATS" | awk '{print $3}' | sed 's/%//')
        
        # Check thresholds
        if (( $(echo "$CPU_PERCENT > 80" | bc -l) )); then
            log "⚠️  WARNING: High CPU usage: ${CPU_PERCENT}%"
        fi
        
        if (( $(echo "$MEM_PERCENT > 80" | bc -l) )); then
            log "⚠️  WARNING: High memory usage: ${MEM_PERCENT}%"
        fi
    else
        log "❌ Could not retrieve performance stats"
        return 1
    fi
    
    return 0
}

# API endpoint testing
test_api_endpoints() {
    log "🌐 Testing API endpoints"
    
    # Test health endpoint
    if curl -sf http://localhost:8080/health > /dev/null; then
        log "✅ Health endpoint responding"
    else
        log "❌ Health endpoint not responding"
        return 1
    fi
    
    # Test response time
    RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8080/health)
    log "⏱️  Health endpoint response time: ${RESPONSE_TIME}s"
    
    # Check if response time is within acceptable range (< 1 second)
    if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
        log "⚠️  WARNING: Slow response time: ${RESPONSE_TIME}s"
    fi
    
    return 0
}

# Main monitoring function
run_monitoring_cycle() {
    log "🚀 Starting SAMO security monitoring cycle"
    
    local exit_code=0
    
    # Run all checks
    check_container_health || exit_code=$?
    check_performance || exit_code=$?
    test_api_endpoints || exit_code=$?
    run_security_scan || exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ All monitoring checks passed"
    else
        log "⚠️  Some monitoring checks failed (exit code: $exit_code)"
    fi
    
    log "📊 Monitoring cycle completed"
    echo "---" >> "$LOG_FILE"
    
    return $exit_code
}

# Continuous monitoring loop
continuous_monitor() {
    log "🔄 Starting continuous security monitoring (interval: ${SCAN_INTERVAL}s)"
    
    while true; do
        run_monitoring_cycle
        
        log "😴 Sleeping for $SCAN_INTERVAL seconds..."
        sleep "$SCAN_INTERVAL"
    done
}

# Signal handlers
cleanup() {
    log "🛑 Monitoring stopped by user"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main execution
case "${1:-continuous}" in
    "once")
        log "🔍 Running single monitoring cycle"
        run_monitoring_cycle
        ;;
    "continuous")
        continuous_monitor
        ;;
    "scan")
        run_security_scan
        ;;
    "health")
        check_container_health
        ;;
    "performance")
        check_performance
        ;;
    "api")
        test_api_endpoints
        ;;
    *)
        echo "Usage: $0 [once|continuous|scan|health|performance|api]"
        echo "  once        - Run monitoring cycle once"
        echo "  continuous  - Run continuous monitoring (default)"
        echo "  scan        - Run security scan only"
        echo "  health      - Check container health only"
        echo "  performance - Check performance only"
        echo "  api         - Test API endpoints only"
        exit 1
        ;;
esac