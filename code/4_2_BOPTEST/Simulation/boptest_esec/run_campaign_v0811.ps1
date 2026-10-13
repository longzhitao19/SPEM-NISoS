param(
    [string]$Python = 'python',
    [string]$BaseUrl = 'http://127.0.0.1',
    [string]$WorkRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Workers = 12,
    [int]$MaxAttempts = 3,
    [switch]$ResumeExisting
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'

Write-Host "=== v0.8.11 stage 1: N=1 full-factorial screening (272 trials) ==="
$screenArgs = @(
    '-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'run_screening_v0811_N1.ps1'),
    '-Python',$Python,
    '-BaseUrl',$BaseUrl,
    '-WorkRoot',$WorkRoot,
    '-Workers',"$Workers",
    '-MaxAttempts',"$MaxAttempts"
)
if ($ResumeExisting) { $screenArgs += '-ResumeExisting' }
& powershell @screenArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "N=1 screening did not pass. Formal N=20 campaign will NOT start."
    exit $LASTEXITCODE
}

Write-Host "=== v0.8.11 stage 2: N=20 formal campaign (5136 trials) ==="
$formalArgs = @(
    '-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'run_formal_v0811_N20.ps1'),
    '-Python',$Python,
    '-BaseUrl',$BaseUrl,
    '-WorkRoot',$WorkRoot,
    '-Workers',"$Workers",
    '-MaxAttempts',"$MaxAttempts"
)
if ($ResumeExisting) { $formalArgs += '-ResumeExisting' }
& powershell @formalArgs
exit $LASTEXITCODE
