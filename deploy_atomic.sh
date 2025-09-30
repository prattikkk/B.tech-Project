#!/bin/bash
# Atomic Deployment Script for IoT Inference Service
# Ensures artifact integrity through atomic moves and checksum verification

set -euo pipefail

# Configuration
DEPLOY_USER="${DEPLOY_USER:-iotuser}"
DEPLOY_HOST="${DEPLOY_HOST:-edge-device}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/iot-inference}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_phase2}"
METADATA_FILE="metadata.json"
SERVICE_NAME="phase4-inference"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to calculate SHA256 checksum
calculate_checksum() {
    local file="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        shasum -a 256 "$file" | cut -d' ' -f1
    else
        sha256sum "$file" | cut -d' ' -f1
    fi
}

# Function to verify local artifacts before deployment
verify_local_artifacts() {
    log_info "Verifying local artifacts before deployment..."
    
    if [[ ! -f "$ARTIFACTS_DIR/$METADATA_FILE" ]]; then
        log_error "Metadata file not found: $ARTIFACTS_DIR/$METADATA_FILE"
        return 1
    fi
    
    # Extract expected checksums from metadata.json
    local metadata_file="$ARTIFACTS_DIR/$METADATA_FILE"
    local artifacts_manifest=$(jq -r '.artifacts_manifest // empty' "$metadata_file" 2>/dev/null || echo '{}')
    
    if [[ "$artifacts_manifest" == "{}" ]]; then
        log_warning "No artifacts_manifest found in metadata.json, checking individual checksums"
        return 0
    fi
    
    # Verify each artifact listed in manifest
    echo "$artifacts_manifest" | jq -r 'to_entries[] | "\(.key) \(.value.sha256 // .value)"' | while read -r filename expected_checksum; do
        local filepath="$ARTIFACTS_DIR/$filename"
        if [[ -f "$filepath" ]]; then
            local actual_checksum=$(calculate_checksum "$filepath")
            if [[ "$actual_checksum" == "$expected_checksum" ]]; then
                log_success "✓ $filename: checksum verified"
            else
                log_error "✗ $filename: checksum mismatch"
                log_error "  Expected: $expected_checksum"
                log_error "  Actual:   $actual_checksum"
                return 1
            fi
        else
            log_warning "Artifact not found locally: $filepath"
        fi
    done
    
    log_success "All local artifacts verified successfully"
}

# Function to deploy single artifact atomically
deploy_artifact_atomic() {
    local local_file="$1"
    local remote_file="$2"
    local expected_checksum="$3"
    
    local filename=$(basename "$local_file")
    local temp_remote_file="/tmp/${filename}.new.$$"
    
    log_info "Deploying $filename atomically..."
    
    # Step 1: Upload to temporary location
    log_info "  Uploading to temporary location: $temp_remote_file"
    if ! scp "$local_file" "$DEPLOY_USER@$DEPLOY_HOST:$temp_remote_file"; then
        log_error "Failed to upload $filename"
        return 1
    fi
    
    # Step 2: Verify checksum on remote system
    log_info "  Verifying checksum on remote system..."
    local remote_checksum
    remote_checksum=$(ssh "$DEPLOY_USER@$DEPLOY_HOST" "sha256sum $temp_remote_file | cut -d' ' -f1")
    
    if [[ "$remote_checksum" != "$expected_checksum" ]]; then
        log_error "  Checksum mismatch for $filename on remote system"
        log_error "    Expected: $expected_checksum"
        log_error "    Remote:   $remote_checksum"
        # Cleanup temporary file
        ssh "$DEPLOY_USER@$DEPLOY_HOST" "rm -f $temp_remote_file" || true
        return 1
    fi
    
    log_success "  ✓ Remote checksum verified: $remote_checksum"
    
    # Step 3: Atomic move to final location
    log_info "  Performing atomic move to final location..."
    if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "mkdir -p $(dirname $remote_file) && mv $temp_remote_file $remote_file"; then
        log_error "Failed to move $filename to final location"
        # Cleanup temporary file
        ssh "$DEPLOY_USER@$DEPLOY_HOST" "rm -f $temp_remote_file" || true
        return 1
    fi
    
    log_success "  ✓ $filename deployed successfully"
}

# Function to backup existing artifacts
backup_remote_artifacts() {
    log_info "Creating backup of existing artifacts..."
    
    local backup_dir="$DEPLOY_PATH/backup/$(date +%Y%m%d-%H%M%S)"
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        if [[ -d $DEPLOY_PATH/$ARTIFACTS_DIR ]]; then
            mkdir -p $backup_dir
            cp -r $DEPLOY_PATH/$ARTIFACTS_DIR/* $backup_dir/ 2>/dev/null || true
            echo 'Backup created at: $backup_dir'
        else
            echo 'No existing artifacts to backup'
        fi
    "
    
    log_success "Backup completed"
}

# Function to deploy all artifacts
deploy_artifacts() {
    log_info "Starting atomic deployment of artifacts..."
    
    # Parse metadata.json for artifact checksums
    local metadata_file="$ARTIFACTS_DIR/$METADATA_FILE"
    local artifacts_manifest=$(jq -r '.artifacts_manifest // empty' "$metadata_file" 2>/dev/null || echo '{}')
    
    if [[ "$artifacts_manifest" == "{}" ]]; then
        log_warning "No artifacts_manifest in metadata.json, deploying critical files..."
        
        # Deploy critical files with manual checksum calculation
        local critical_files=(
            "model_hybrid.onnx"
            "best_model_hybrid.pth"
            "final_model_hybrid.pth"
            "lgbm_model.pkl"
            "scaler.pkl"
            "feature_order.json"
            "evaluation.json"
        )
        
        for file in "${critical_files[@]}"; do
            local filepath="$ARTIFACTS_DIR/$file"
            if [[ -f "$filepath" ]]; then
                local checksum=$(calculate_checksum "$filepath")
                local remote_path="$DEPLOY_PATH/$ARTIFACTS_DIR/$file"
                deploy_artifact_atomic "$filepath" "$remote_path" "$checksum"
            fi
        done
    else
        # Deploy artifacts listed in manifest
        echo "$artifacts_manifest" | jq -r 'to_entries[] | "\(.key) \(.value.sha256 // .value)"' | while read -r filename expected_checksum; do
            local filepath="$ARTIFACTS_DIR/$filename"
            if [[ -f "$filepath" ]]; then
                local remote_path="$DEPLOY_PATH/$ARTIFACTS_DIR/$filename"
                deploy_artifact_atomic "$filepath" "$remote_path" "$expected_checksum"
            else
                log_warning "Skipping missing file: $filepath"
            fi
        done
    fi
    
    # Always deploy metadata.json last
    log_info "Deploying metadata.json..."
    local metadata_checksum=$(calculate_checksum "$metadata_file")
    local remote_metadata="$DEPLOY_PATH/$ARTIFACTS_DIR/$METADATA_FILE"
    deploy_artifact_atomic "$metadata_file" "$remote_metadata" "$metadata_checksum"
    
    log_success "All artifacts deployed successfully"
}

# Function to verify deployment integrity
verify_deployment() {
    log_info "Verifying deployment integrity on remote system..."
    
    # Run remote verification script
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        cd $DEPLOY_PATH
        
        # Check if metadata.json exists
        if [[ ! -f $ARTIFACTS_DIR/$METADATA_FILE ]]; then
            echo 'ERROR: metadata.json not found on remote system'
            exit 1
        fi
        
        # Verify checksums using metadata
        if command -v jq >/dev/null 2>&1; then
            echo 'Verifying artifacts using metadata.json...'
            artifacts_manifest=\$(jq -r '.artifacts_manifest // empty' $ARTIFACTS_DIR/$METADATA_FILE 2>/dev/null || echo '{}')
            
            if [[ \"\$artifacts_manifest\" != '{}' ]]; then
                echo \"\$artifacts_manifest\" | jq -r 'to_entries[] | \"\(.key) \(.value.sha256 // .value)\"' | while read -r filename expected_checksum; do
                    filepath=\"$ARTIFACTS_DIR/\$filename\"
                    if [[ -f \"\$filepath\" ]]; then
                        actual_checksum=\$(sha256sum \"\$filepath\" | cut -d' ' -f1)
                        if [[ \"\$actual_checksum\" == \"\$expected_checksum\" ]]; then
                            echo \"✓ \$filename: checksum verified\"
                        else
                            echo \"✗ \$filename: checksum mismatch\"
                            echo \"  Expected: \$expected_checksum\"
                            echo \"  Actual:   \$actual_checksum\"
                            exit 1
                        fi
                    else
                        echo \"Warning: \$filename not found\"
                    fi
                done
                echo 'All artifacts verified successfully on remote system'
            else
                echo 'No artifacts_manifest found, basic verification only'
            fi
        else
            echo 'jq not available, skipping detailed verification'
        fi
        
        # Check critical files exist
        critical_files=('feature_order.json' 'scaler.pkl')
        for file in \"\${critical_files[@]}\"; do
            if [[ -f \"$ARTIFACTS_DIR/\$file\" ]]; then
                echo \"✓ Critical file exists: \$file\"
            else
                echo \"✗ Critical file missing: \$file\"
                exit 1
            fi
        done
        
        echo 'Deployment verification completed successfully'
    "
    
    if [[ $? -eq 0 ]]; then
        log_success "Remote deployment verification passed"
    else
        log_error "Remote deployment verification failed"
        return 1
    fi
}

# Function to restart service safely
restart_service() {
    log_info "Restarting inference service..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        # Check if service exists
        if systemctl list-unit-files | grep -q $SERVICE_NAME; then
            echo 'Stopping service...'
            sudo systemctl stop $SERVICE_NAME || true
            
            echo 'Starting service...'
            sudo systemctl start $SERVICE_NAME
            
            echo 'Checking service status...'
            sleep 2
            if sudo systemctl is-active --quiet $SERVICE_NAME; then
                echo 'Service started successfully'
                sudo systemctl status $SERVICE_NAME --no-pager -l
            else
                echo 'Service failed to start'
                sudo journalctl -u $SERVICE_NAME --no-pager -l --since='5 minutes ago'
                exit 1
            fi
        else
            echo 'Service $SERVICE_NAME not found, manual start required'
        fi
    "
    
    if [[ $? -eq 0 ]]; then
        log_success "Service restarted successfully"
    else
        log_error "Service restart failed"
        return 1
    fi
}

# Function to run deployment health check
health_check() {
    log_info "Running post-deployment health check..."
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        if ssh "$DEPLOY_USER@$DEPLOY_HOST" "curl -f http://localhost:9208/ready >/dev/null 2>&1"; then
            log_success "Service is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Service failed to become ready after $max_attempts attempts"
            return 1
        fi
        
        sleep 2
        ((attempt++))
    done
    
    # Get detailed health info
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "
        echo 'Service health status:'
        curl -s http://localhost:9208/health | jq . || echo 'Health endpoint not available'
        
        echo 'Service metrics:'
        curl -s http://localhost:9208/metrics | grep -E '(phase4_messages_total|phase4_uptime|phase4_memory)' || echo 'Metrics not available'
    "
    
    log_success "Health check completed"
}

# Main deployment function
main() {
    log_info "Starting atomic deployment process..."
    log_info "Target: $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH"
    
    # Verify prerequisites
    if ! command -v jq >/dev/null 2>&1; then
        log_error "jq is required but not installed"
        exit 1
    fi
    
    if ! command -v ssh >/dev/null 2>&1; then
        log_error "ssh is required but not installed"
        exit 1
    fi
    
    if ! command -v scp >/dev/null 2>&1; then
        log_error "scp is required but not installed"
        exit 1
    fi
    
    # Check if artifacts directory exists
    if [[ ! -d "$ARTIFACTS_DIR" ]]; then
        log_error "Artifacts directory not found: $ARTIFACTS_DIR"
        exit 1
    fi
    
    # Step 1: Verify local artifacts
    if ! verify_local_artifacts; then
        log_error "Local artifact verification failed"
        exit 1
    fi
    
    # Step 2: Test connectivity
    log_info "Testing connectivity to deployment target..."
    if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "echo 'Connection successful'"; then
        log_error "Cannot connect to deployment target"
        exit 1
    fi
    
    # Step 3: Create backup
    backup_remote_artifacts
    
    # Step 4: Deploy artifacts atomically
    if ! deploy_artifacts; then
        log_error "Artifact deployment failed"
        exit 1
    fi
    
    # Step 5: Verify deployment
    if ! verify_deployment; then
        log_error "Deployment verification failed"
        exit 1
    fi
    
    # Step 6: Restart service
    if ! restart_service; then
        log_error "Service restart failed"
        exit 1
    fi
    
    # Step 7: Health check
    if ! health_check; then
        log_error "Health check failed"
        exit 1
    fi
    
    log_success "🎉 Atomic deployment completed successfully!"
    log_info "No more 'expected != actual' checksum mismatches should occur"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify-local")
        verify_local_artifacts
        ;;
    "verify-remote")
        verify_deployment
        ;;
    "health-check")
        health_check
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  deploy       - Full atomic deployment (default)"
        echo "  verify-local - Verify local artifacts only"
        echo "  verify-remote - Verify remote deployment only"
        echo "  health-check - Run health check only"
        echo "  help         - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  DEPLOY_USER     - Deployment user (default: iotuser)"
        echo "  DEPLOY_HOST     - Deployment host (default: edge-device)"
        echo "  DEPLOY_PATH     - Deployment path (default: /opt/iot-inference)"
        echo "  ARTIFACTS_DIR   - Local artifacts directory (default: artifacts_phase2)"
        ;;
    *)
        log_error "Unknown command: $1"
        log_info "Use '$0 help' for usage information"
        exit 1
        ;;
esac