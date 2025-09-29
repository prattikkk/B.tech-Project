param(
  [string]$ArtifactsPhase2 = "$PSScriptRoot\..\artifacts_phase2"
)
$ErrorActionPreference = 'Stop'
$python = 'python'

Write-Host "Running Phase 2..." -ForegroundColor Cyan

# Ensure optimization config exists (optional)
if (-not (Test-Path -LiteralPath "$ArtifactsPhase2")) {
  New-Item -ItemType Directory -Path "$ArtifactsPhase2" | Out-Null
}

& $python "$PSScriptRoot\..\Phase_Two.py"
if ($LASTEXITCODE -ne 0) { throw "Phase 2 failed with exit code $LASTEXITCODE" }
Write-Host "Phase 2 complete." -ForegroundColor Green
