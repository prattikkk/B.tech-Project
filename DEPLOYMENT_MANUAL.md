# Atomic Deployment Manual

## Overview

This guide provides step-by-step instructions for atomic deployment of IoT inference artifacts to prevent checksum mismatches and ensure deployment integrity.

## Problem Statement

**Issue**: Partial/corrupt writes during deployment cause checksum mismatches and service failures.
**Error**: "expected != actual" checksum mismatches reported in Phase_Four.py
**Solution**: Atomic deployment with temporary files and checksum verification.

## Prerequisites

### Local Environment
- Bash shell (Linux/macOS) or PowerShell (Windows)
- SSH access to target deployment host
- `scp` and `ssh` commands available
- `jq` for JSON processing (optional but recommended)

### Target Host Requirements
- SSH daemon running
- `sha256sum` command available
- Directory write permissions
- `systemd` for service management (optional)

### Required Files
- `deploy_atomic.sh` (Bash version) or `deploy_atomic.ps1` (PowerShell version)
- `artifacts_phase2/` directory with model files
- `artifacts_phase2/metadata.json` with checksum manifest

## Configuration

### Environment Variables
```bash
export DEPLOY_USER="iotuser"           # Target username  
export DEPLOY_HOST="edge-device"       # Target hostname/IP
export DEPLOY_PATH="/opt/iot-inference" # Target directory
export ARTIFACTS_DIR="artifacts_phase2" # Local artifacts directory
```

### SSH Key Setup
```bash
# Generate SSH key if needed
ssh-keygen -t rsa -b 4096 -f ~/.ssh/iot-deployment

# Copy public key to target
ssh-copy-id -i ~/.ssh/iot-deployment.pub $DEPLOY_USER@$DEPLOY_HOST

# Test connection
ssh $DEPLOY_USER@$DEPLOY_HOST "echo 'Connection successful'"
```

## Deployment Methods

### Method 1: Automated Script Deployment

#### Using Bash (Linux/macOS)
```bash
# Make script executable
chmod +x deploy_atomic.sh

# Run full deployment
./deploy_atomic.sh deploy

# Or run individual steps
./deploy_atomic.sh verify-local    # Verify local artifacts
./deploy_atomic.sh verify-remote   # Verify remote deployment  
./deploy_atomic.sh health-check    # Run health check only
```

#### Using PowerShell (Windows)
```powershell
# Set execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run full deployment
.\deploy_atomic.ps1 deploy

# Or run individual steps
.\deploy_atomic.ps1 verify-local
.\deploy_atomic.ps1 verify-remote  
.\deploy_atomic.ps1 health-check
```

### Method 2: Manual Step-by-Step Deployment

#### Step 1: Pre-deployment Verification
```bash
# Verify local artifacts exist and have valid checksums
cd artifacts_phase2

# Check metadata.json exists
if [[ ! -f metadata.json ]]; then
    echo "ERROR: metadata.json not found"
    exit 1
fi

# Verify checksums (if jq available)
if command -v jq >/dev/null; then
    jq -r '.artifacts_manifest // {} | to_entries[] | "\(.key) \(.value.sha256 // .value)"' metadata.json | while read -r filename expected_checksum; do
        if [[ -f "$filename" ]]; then
            actual_checksum=$(sha256sum "$filename" | cut -d' ' -f1)
            if [[ "$actual_checksum" == "$expected_checksum" ]]; then
                echo "✓ $filename: checksum verified"
            else
                echo "✗ $filename: checksum mismatch"
                echo "  Expected: $expected_checksum"
                echo "  Actual:   $actual_checksum"
                exit 1
            fi
        fi
    done
fi
```

#### Step 2: Create Remote Backup
```bash
# Create timestamped backup on remote host
BACKUP_DIR="/opt/iot-inference/backup/$(date +%Y%m%d-%H%M%S)"
ssh $DEPLOY_USER@$DEPLOY_HOST "
    if [[ -d /opt/iot-inference/artifacts_phase2 ]]; then
        mkdir -p $BACKUP_DIR
        cp -r /opt/iot-inference/artifacts_phase2/* $BACKUP_DIR/
        echo 'Backup created at: $BACKUP_DIR'
    fi
"
```

#### Step 3: Deploy Critical Artifacts Atomically
```bash
# Deploy each artifact using atomic operations
ARTIFACTS=("model_hybrid.onnx" "best_model_hybrid.pth" "scaler.pkl" "feature_order.json")

for artifact in "${ARTIFACTS[@]}"; do
    if [[ -f "artifacts_phase2/$artifact" ]]; then
        echo "Deploying $artifact..."
        
        # Calculate expected checksum
        expected_checksum=$(sha256sum "artifacts_phase2/$artifact" | cut -d' ' -f1)
        
        # Upload to temporary location
        temp_file="/tmp/$artifact.new.$$"
        scp "artifacts_phase2/$artifact" "$DEPLOY_USER@$DEPLOY_HOST:$temp_file"
        
        # Verify checksum on remote
        remote_checksum=$(ssh $DEPLOY_USER@$DEPLOY_HOST "sha256sum $temp_file | cut -d' ' -f1")
        
        if [[ "$remote_checksum" != "$expected_checksum" ]]; then
            echo "ERROR: Checksum mismatch for $artifact"
            echo "Expected: $expected_checksum"
            echo "Remote:   $remote_checksum"
            ssh $DEPLOY_USER@$DEPLOY_HOST "rm -f $temp_file"
            exit 1
        fi
        
        # Atomic move to final location
        final_path="/opt/iot-inference/artifacts_phase2/$artifact"
        ssh $DEPLOY_USER@$DEPLOY_HOST "mkdir -p $(dirname $final_path) && mv $temp_file $final_path"
        
        echo "✓ $artifact deployed successfully"
    fi
done
```

#### Step 4: Deploy metadata.json Last
```bash
echo "Deploying metadata.json..."

# Calculate checksum
metadata_checksum=$(sha256sum "artifacts_phase2/metadata.json" | cut -d' ' -f1)

# Upload to temporary location
temp_metadata="/tmp/metadata.json.new.$$"
scp "artifacts_phase2/metadata.json" "$DEPLOY_USER@$DEPLOY_HOST:$temp_metadata"

# Verify and move
remote_checksum=$(ssh $DEPLOY_USER@$DEPLOY_HOST "sha256sum $temp_metadata | cut -d' ' -f1")
if [[ "$remote_checksum" == "$metadata_checksum" ]]; then
    ssh $DEPLOY_USER@$DEPLOY_HOST "mv $temp_metadata /opt/iot-inference/artifacts_phase2/metadata.json"
    echo "✓ metadata.json deployed successfully"
else
    echo "ERROR: metadata.json checksum mismatch"
    exit 1
fi
```

#### Step 5: Verify Deployment Integrity
```bash
# Run remote verification
ssh $DEPLOY_USER@$DEPLOY_HOST "
    cd /opt/iot-inference
    
    # Verify metadata.json exists
    if [[ ! -f artifacts_phase2/metadata.json ]]; then
        echo 'ERROR: metadata.json not found'
        exit 1
    fi
    
    # Verify artifacts using metadata (if jq available)
    if command -v jq >/dev/null; then
        jq -r '.artifacts_manifest // {} | to_entries[] | \"\(.key) \(.value.sha256 // .value)\"' artifacts_phase2/metadata.json | while read -r filename expected_checksum; do
            filepath=\"artifacts_phase2/\$filename\"
            if [[ -f \"\$filepath\" ]]; then
                actual_checksum=\$(sha256sum \"\$filepath\" | cut -d' ' -f1)
                if [[ \"\$actual_checksum\" == \"\$expected_checksum\" ]]; then
                    echo \"✓ \$filename: verified\"
                else
                    echo \"✗ \$filename: checksum mismatch\"
                    exit 1
                fi
            fi
        done
    fi
    
    echo 'Deployment verification completed successfully'
"
```

#### Step 6: Restart Service
```bash
# Restart the inference service
ssh $DEPLOY_USER@$DEPLOY_HOST "
    if systemctl list-unit-files | grep -q phase4-inference; then
        echo 'Restarting service...'
        sudo systemctl stop phase4-inference
        sudo systemctl start phase4-inference
        
        # Verify service started
        sleep 5
        if sudo systemctl is-active --quiet phase4-inference; then
            echo 'Service restarted successfully'
        else
            echo 'Service failed to start'
            sudo journalctl -u phase4-inference --no-pager -l --since='2 minutes ago'
            exit 1
        fi
    else
        echo 'Service not found - manual start required'
    fi
"
```

#### Step 7: Health Check
```bash
# Wait for service to be ready
echo "Running health check..."
for i in {1..10}; do
    if ssh $DEPLOY_USER@$DEPLOY_HOST "curl -f http://localhost:9208/ready >/dev/null 2>&1"; then
        echo "✓ Service is ready"
        break
    fi
    echo "Waiting for service... ($i/10)"
    sleep 3
done

# Get health status
ssh $DEPLOY_USER@$DEPLOY_HOST "
    echo 'Health status:'
    curl -s http://localhost:9208/health | jq . || echo 'Health endpoint not available'
    
    echo 'Service metrics:'
    curl -s http://localhost:9208/metrics | grep -E '(phase4_uptime|phase4_memory_rss)' || echo 'Metrics not available'
"
```

## Troubleshooting

### Common Issues

#### 1. Checksum Mismatch During Upload
```
ERROR: Checksum mismatch for model_hybrid.onnx
Expected: abc123...
Remote:   def456...
```

**Solution**:
- Check network stability
- Verify source file integrity
- Retry upload
- Check disk space on target

#### 2. SSH Connection Failed
```
ssh: connect to host edge-device port 22: Connection refused
```

**Solution**:
- Verify target host is reachable: `ping $DEPLOY_HOST`
- Check SSH service: `ssh $DEPLOY_USER@$DEPLOY_HOST`
- Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`

#### 3. Permission Denied on Target
```
mv: cannot move '/tmp/model.onnx.new': Permission denied
```

**Solution**:
- Check target directory permissions
- Verify user has write access
- Use `sudo` if needed for system directories

#### 4. Service Restart Failed
```
Job for phase4-inference.service failed
```

**Solution**:
- Check service logs: `sudo journalctl -u phase4-inference -f`
- Verify configuration files
- Check for missing dependencies
- Validate artifact integrity

### Recovery Procedures

#### Rollback to Previous Version
```bash
# Find latest backup
LATEST_BACKUP=$(ssh $DEPLOY_USER@$DEPLOY_HOST "ls -t /opt/iot-inference/backup/ | head -1")

# Restore from backup
ssh $DEPLOY_USER@$DEPLOY_HOST "
    if [[ -d /opt/iot-inference/backup/$LATEST_BACKUP ]]; then
        echo 'Rolling back to: $LATEST_BACKUP'
        rm -rf /opt/iot-inference/artifacts_phase2/*
        cp -r /opt/iot-inference/backup/$LATEST_BACKUP/* /opt/iot-inference/artifacts_phase2/
        sudo systemctl restart phase4-inference
    fi
"
```

#### Clean Up Failed Deployment
```bash
# Remove temporary files
ssh $DEPLOY_USER@$DEPLOY_HOST "rm -f /tmp/*.new.*"

# Reset service if needed
ssh $DEPLOY_USER@$DEPLOY_HOST "
    sudo systemctl stop phase4-inference
    sudo systemctl reset-failed phase4-inference
"
```

## Validation

### Pre-deployment Checklist
- [ ] Local artifacts exist and have valid checksums
- [ ] SSH connectivity to target host verified
- [ ] Target directory exists and is writable  
- [ ] Backup of existing artifacts created
- [ ] Service restart permissions available

### Post-deployment Checklist
- [ ] All artifacts deployed successfully
- [ ] Remote checksum verification passed
- [ ] Service restarted without errors
- [ ] Health check endpoints responding
- [ ] No "expected != actual" errors in logs

## Automation Integration

### CI/CD Pipeline Integration
```yaml
# Add to GitHub Actions workflow
- name: Atomic Deployment
  run: |
    chmod +x deploy_atomic.sh
    ./deploy_atomic.sh deploy
  env:
    DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
    DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
    DEPLOY_PATH: ${{ secrets.DEPLOY_PATH }}
```

### Monitoring Integration
```bash
# Add monitoring webhook
curl -X POST https://monitoring.example.com/webhook \
  -H "Content-Type: application/json" \
  -d '{"status": "deployed", "version": "'$(git rev-parse HEAD)'", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
```

## Expected Results

After successful atomic deployment:

1. **No Checksum Mismatches**: Eliminates "expected != actual" errors
2. **Atomic Operations**: Either complete success or complete rollback
3. **Integrity Verification**: All artifacts verified before service restart
4. **Service Reliability**: Clean service restarts with health validation
5. **Audit Trail**: Complete deployment logs and backup history

## Summary

This atomic deployment process ensures:
- **Data Integrity**: Checksums verified at every step
- **Atomicity**: Complete success or failure, no partial states
- **Recovery**: Automated backup and rollback capabilities
- **Validation**: Comprehensive pre and post-deployment checks
- **Monitoring**: Health endpoints and service status verification

The result is **zero tolerance for checksum mismatches** and reliable, predictable deployments.