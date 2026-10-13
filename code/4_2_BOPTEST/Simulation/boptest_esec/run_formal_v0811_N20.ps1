param(
    [string]$Python = 'python',
    [string]$BaseUrl = 'http://127.0.0.1',
    [string]$WorkRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputRelative = 'boptest_esec\results\formal_v0811_N20',
    [int]$Workers = 12,
    [int]$MaxAttempts = 3,
    [int]$PollSeconds = 60,
    [switch]$ResumeExisting
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$outputPath = Join-Path $WorkRoot $OutputRelative
$logRoot = Join-Path $PSScriptRoot 'results\formal_v0811_N20_supervisor'
$analysisPath = Join-Path $outputPath 'analysis'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$supervisorLog = Join-Path $logRoot 'supervisor.log'

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff K') $Message"
    Add-Content -LiteralPath $supervisorLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-State {
    foreach ($name in @('run_manifest.txt','checkpoint_manifest.txt')) {
        $path = Join-Path $outputPath $name
        if (Test-Path -LiteralPath $path) {
            try { return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json } catch { }
        }
    }
    return $null
}

Write-Log "formal_v0811_supervisor_start work_root=$WorkRoot output=$OutputRelative workers=$Workers"

$preflightJson = Join-Path $logRoot 'formal_preflight.txt'
& $Python -m boptest_esec.formal_preflight --base-url $BaseUrl --design-seed 20260915 --formal-replicates 20 --step-s 300 --trial-duration-s 21600 --output $preflightJson
if ($LASTEXITCODE -ne 0) {
    Write-Log "preflight_failed exit_code=$LASTEXITCODE report=$preflightJson"
    exit 2
}
Write-Log "preflight_passed report=$preflightJson"

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $stdoutPath = Join-Path $logRoot ("runner_attempt_{0}.stdout.log" -f $attempt)
    $stderrPath = Join-Path $logRoot ("runner_attempt_{0}.stderr.log" -f $attempt)
    $arguments = @(
        '-m','boptest_esec.runner',
        '--base-url',$BaseUrl,
        '--output',$OutputRelative,
        '--split','formal',
        '--design','v2',
        '--formal-replicates','20',
        '--design-seed','20260915',
        '--step-s','300',
        '--trial-duration-s','21600',
        '--attack-delivery-probability','1.0',
        '--workers',"$Workers",
        '--retries','2'
    )
    $fingerprintPath = Join-Path $outputPath 'design_fingerprint.txt'
    if ($ResumeExisting -or $attempt -gt 1 -or (Test-Path -LiteralPath $fingerprintPath)) { $arguments += '--resume' }

    Write-Log "runner_start attempt=$attempt"
    $process = Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $arguments -WorkingDirectory $WorkRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    while (-not $process.HasExited) {
        $state = Get-State
        if ($null -eq $state) {
            Write-Log "heartbeat attempt=$attempt pid=$($process.Id) manifest=absent"
        } else {
            Write-Log "heartbeat attempt=$attempt pid=$($process.Id) status=$($state.status) completed=$($state.completed) planned=$($state.planned)"
        }
        Start-Sleep -Seconds $PollSeconds
        $process.Refresh()
    }
    $state = Get-State
    Write-Log "runner_exit attempt=$attempt exit_code=$($process.ExitCode)"
    if ($null -ne $state -and $state.status -eq 'complete_raw_boptest_esec_execution' -and [int]$state.completed -eq 5136) {
        break
    }
    if ($attempt -eq $MaxAttempts) {
        Write-Log 'runner_failed_max_attempts'
        exit 3
    }
    Write-Log 'runner_incomplete_retry_with_resume'
}

Write-Log 'raw_run_complete_starting_postflight_analysis'
& $Python -m boptest_esec.formal_postflight --input $outputPath --output $analysisPath --expected-trials 5136
if ($LASTEXITCODE -ne 0) {
    Write-Log "postflight_failed exit_code=$LASTEXITCODE; inspect analysis diagnostics before freezing results"
    exit 4
}
Write-Log 'formal_campaign_complete_postflight_passed'
exit 0
