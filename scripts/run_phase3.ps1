param(
  [string]$ArtifactsPhase2 = "$PSScriptRoot\..\artifacts_phase2",
  [string]$Broker = $env:MQTT_BROKER,
  [int]$Port = [int]($env:MQTT_PORT | ForEach-Object { if ($_){$_} else {1883} }),
  [string]$Topic = $env:MQTT_TOPIC
)
$ErrorActionPreference = 'Stop'
$python = 'python'

Write-Host "Running Phase 3 (inference service)..." -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $ArtifactsPhase2)) {
  throw "Artifacts directory not found: $ArtifactsPhase2"
}

# Export minimal env if provided via parameters
if ($Broker) { $env:MQTT_BROKER = $Broker }
if ($Port) { $env:MQTT_PORT = $Port }
if ($Topic) { $env:MQTT_TOPIC = $Topic }

& $python "$PSScriptRoot\..\Phase_Three.py"
if ($LASTEXITCODE -ne 0) { throw "Phase 3 failed with exit code $LASTEXITCODE" }
Write-Host "Phase 3 exited." -ForegroundColor Green
