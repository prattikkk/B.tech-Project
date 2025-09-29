param(
  [string]$CsvPath = "$PSScriptRoot\..\IDSAI.csv",
  [string]$Artifacts = "$PSScriptRoot\..\artifacts_phase1",
  [int]$Seed = 42,
  [double]$Val = 0.10,
  [double]$Test = 0.10,
  [int]$BatchSize = 64,
  [switch]$NoCsv,
  [string]$StrictFeaturesJson = ''
)

$ErrorActionPreference = 'Stop'
$python = 'python'

Write-Host "Running Phase 1..." -ForegroundColor Cyan

$strictArg = @()
if ($StrictFeaturesJson -and (Test-Path -LiteralPath $StrictFeaturesJson)) {
  $strictArg = @('--strict-features-json', $StrictFeaturesJson)
}

& $python "$PSScriptRoot\..\Phase_One.py" `
  --csv $CsvPath `
  --artifacts $Artifacts `
  --seed $Seed `
  --val $Val `
  --test $Test `
  --batch-size $BatchSize `
  $(if ($NoCsv) { '--no-csv' }) `
  @strictArg

if ($LASTEXITCODE -ne 0) { throw "Phase 1 failed with exit code $LASTEXITCODE" }
Write-Host "Phase 1 complete." -ForegroundColor Green
