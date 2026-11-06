#Requires -Version 5.1
<#
.SYNOPSIS
    One-command Windows PowerShell setup for cc-agy-plugin behind an enterprise AI gateway or Cloud Run proxy.

.DESCRIPTION
    Verifies Python 3.10+ (skipping Windows Store app-execution stubs), git, and optional Google
    Cloud SDK (`gcloud`) OIDC sign-in when `-GatewayAudience` is configured, then runs
    `python -m server.install` to register cc-agy-plugin across Claude Code, Codex, Cursor,
    OpenCode, and/or Antigravity without writing secrets or static API keys to disk.

.EXAMPLE
    .\scripts\bootstrap-gateway.ps1 -GatewayUrl "https://ai-gateway.example.com" -GatewayAudience "https://ai-gateway.example.com"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$GatewayUrl = "",

    [Parameter(Mandatory = $false)]
    [string]$GatewayAudience = "",

    [Parameter(Mandatory = $false)]
    [ValidateSet("auto", "vertex", "gemini", "litellm-vertex", "litellm-gemini", "bifrost")]
    [string]$GatewayWire = "auto",

    [Parameter(Mandatory = $false)]
    [Alias("HostTarget")]
    [ValidateSet("all", "claude", "codex", "cursor", "opencode", "agy")]
    [string]$HostClient = "all",

    [Parameter(Mandatory = $false)]
    [Alias("Profile")]
    [ValidateSet("economy", "balanced", "quality")]
    [string]$RoutingProfile = "balanced",

    [Parameter(Mandatory = $false)]
    [double]$MaxDailyCents = 0,

    [Parameter(Mandatory = $false)]
    [string]$TelemetrySink = "",

    [Parameter(Mandatory = $false)]
    [switch]$NoAutoApply,

    [Parameter(Mandatory = $false)]
    [switch]$NoBootstrap,

    [Parameter(Mandatory = $false)]
    [switch]$NonInteractive,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$GatewayUrlFromGeminiEnv = $false
if (-not $GatewayUrl) {
    foreach ($EnvPair in @(
        @{ Url = $env:CC_AGY_VERTEX_BASE_URL; IsGemini = $false },
        @{ Url = $env:GOOGLE_VERTEX_BASE_URL; IsGemini = $false },
        @{ Url = $env:CC_AGY_GEMINI_BASE_URL; IsGemini = $true },
        @{ Url = $env:GOOGLE_GEMINI_BASE_URL; IsGemini = $true }
    )) {
        if ($EnvPair.Url -and $EnvPair.Url.Trim()) {
            $GatewayUrl = $EnvPair.Url.Trim()
            $GatewayUrlFromGeminiEnv = [bool]$EnvPair.IsGemini
            break
        }
    }
}
if (-not $GatewayUrl) {
    throw "GatewayUrl is required: pass -GatewayUrl 'https://...' or set CC_AGY_VERTEX_BASE_URL / CC_AGY_GEMINI_BASE_URL."
}

if ($GatewayUrl -match '[\s\x00-\x1f\x7f&|;<>`$"''\\%^!()]' -or $GatewayUrl -notmatch '^https://|^http://(localhost|127\.0\.0\.1|\[::1\])(:|/|$)') {
    throw "GatewayUrl must be a clean https:// URL (or loopback http://localhost / http://127.0.0.1) without whitespace or shell metacharacters. Got: $GatewayUrl"
}
$DecodedGatewayUrl = try { [uri]::UnescapeDataString($GatewayUrl) } catch { $GatewayUrl }
$SecretParamPattern = '(?i)(?<![a-z0-9])(key|apikey|xapikey|token|[a-z0-9]*(?:token|secret|password|passwd|apikey|accesskey|authkey|privatekey|secretkey|virtualkey|signature|credential|credentials)(?-i:(?:[A-Z][a-zA-Z0-9]*)?)|pwd|pass|pw|jwt|sid|assertion|sig|sigv[0-9]+|auth|authorization|bearer)[0-9]*(?![a-z0-9])'
if ($DecodedGatewayUrl -match '^[a-zA-Z][a-zA-Z0-9+.-]*://[^/?#]*@' -or $DecodedGatewayUrl -match "[?&#;][^=#&;]*$SecretParamPattern[^=#&;]*(=|$|[#&;])" -or $DecodedGatewayUrl -match "#[^=]*$SecretParamPattern") {
    throw "GatewayUrl must not embed username/password credentials or secret query/fragment parameters; use -GatewayAudience (OIDC) or export CC_AGY_VERTEX_BASE_URL_TOKEN / CC_AGY_BIFROST_VIRTUAL_KEY at runtime."
}

if ($GatewayAudience) {
    if ($GatewayAudience.StartsWith("-") -or $GatewayAudience -match '[\s\x00-\x1f\x7f&|;<>`$"''\\%^!()]') {
        throw "GatewayAudience must not start with '-' or contain whitespace or shell metacharacters. Got: $GatewayAudience"
    }
    if ($GatewayAudience -notmatch '^(auto|https://|http://(localhost|127\.0\.0\.1|\[::1\])|iap://|api://)') {
        throw "GatewayAudience must start with https://, iap://, api://, loopback http://localhost, or 'auto'. Got: $GatewayAudience"
    }
}

if ($MaxDailyCents -lt 0 -or [double]::IsNaN($MaxDailyCents) -or [double]::IsInfinity($MaxDailyCents)) {
    throw "MaxDailyCents must be a finite non-negative number of cents. Got: $MaxDailyCents"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$PythonCmd = $null
$PythonPrefixArgs = @()
foreach ($Cand in @("python", "py")) {
    $FoundList = @(Get-Command $Cand -All -ErrorAction SilentlyContinue)
    foreach ($Found in $FoundList) {
        if (-not $Found.Source -or $Found.Source -like "*\WindowsApps\*") {
            continue
        }
        $TryPrefix = if ($Cand -eq "py") { @("-3") } else { @() }
        $PrevEap = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $Found.Source @TryPrefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $Found.Source
                $PythonPrefixArgs = $TryPrefix
                break
            }
        } catch {
            # Ignore unusable interpreter candidate and continue probing
        } finally {
            $ErrorActionPreference = $PrevEap
        }
    }
    if ($PythonCmd) {
        break
    }
}
if (-not $PythonCmd) {
    throw "Python 3.10+ was not found on PATH (excluding WindowsApps Store stubs). Install Python from https://www.python.org/downloads/windows/ and re-run."
}

if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    throw "git was not found on PATH. Install Git for Windows from https://git-scm.com/download/win and re-run."
}

if ($GatewayAudience) {
    $HasAmbientCreds = [bool]($env:GOOGLE_APPLICATION_CREDENTIALS -or $env:CC_AGY_VERTEX_BASE_URL_TOKEN -or $env:CC_AGY_BIFROST_VIRTUAL_KEY)
    $GcloudCmd = Get-Command "gcloud" -ErrorAction SilentlyContinue
    if (-not $GcloudCmd) {
        if (-not $HasAmbientCreds) {
            Write-Warning "gcloud CLI not found on PATH. Install Google Cloud SDK, configure GOOGLE_APPLICATION_CREDENTIALS (WIF/SA), or set CC_AGY_VERTEX_BASE_URL_TOKEN at runtime."
        }
    } elseif (-not $DryRun -and -not $HasAmbientCreds) {
        $PrevEap = $ErrorActionPreference
        $ActiveAccount = ""
        try {
            $ErrorActionPreference = "Continue"
            $ActiveAccount = (& $GcloudCmd.Source auth list --filter="status:ACTIVE" --format="value(account)" 2>$null | Out-String).Trim()
        } finally {
            $ErrorActionPreference = $PrevEap
        }
        if (-not $ActiveAccount) {
            if ($NonInteractive) {
                throw "No active gcloud session detected in -NonInteractive mode. Run 'gcloud auth login', configure GOOGLE_APPLICATION_CREDENTIALS, or set CC_AGY_VERTEX_BASE_URL_TOKEN."
            }
            Write-Host "No active gcloud session detected. Running 'gcloud auth login'..."
            & $GcloudCmd.Source auth login
            if ($LASTEXITCODE -ne 0) {
                throw "gcloud auth login failed with exit code $LASTEXITCODE."
            }
        }
    }
}

$InstallArgs = @() + $PythonPrefixArgs + @(
    "-m", "server.install",
    "--host", $HostClient,
    "--auth", "gateway",
    "--profile", $RoutingProfile,
    "--base-url", $GatewayUrl,
    "--non-interactive"
)

$NormalizedGatewayWire = switch ($GatewayWire) {
    "litellm-vertex" { "vertex" }
    "litellm-gemini" { "gemini" }
    "bifrost"        { "gemini" }
    default          { $GatewayWire }
}
$EffectiveGatewayWire = if ($NormalizedGatewayWire -and $NormalizedGatewayWire -ne "auto") { $NormalizedGatewayWire } elseif ($GatewayUrlFromGeminiEnv) { "gemini" } else { "" }
if ($EffectiveGatewayWire) {
    $InstallArgs += @("--gateway-wire", $EffectiveGatewayWire)
}
if ($GatewayAudience) {
    $InstallArgs += @("--gateway-audience", $GatewayAudience)
}
if ($MaxDailyCents -gt 0) {
    $InstallArgs += @("--max-daily-cents", "$MaxDailyCents")
}
if ($TelemetrySink) {
    $InstallArgs += @("--telemetry-sink", $TelemetrySink)
}
if ($NoAutoApply) {
    $InstallArgs += "--no-auto-apply"
}
if ($NoBootstrap) {
    $InstallArgs += "--no-bootstrap"
}
if ($DryRun) {
    $InstallArgs += "--dry-run"
}

$DoctorArgs = @() + $PythonPrefixArgs + @("-m", "server.install", "--doctor")

Push-Location $RepoRoot
try {
    & $PythonCmd @InstallArgs
    if ($LASTEXITCODE -ne 0) {
        throw "server.install exited with code $LASTEXITCODE."
    }
    if (-not $DryRun) {
        $env:CC_AGY_VERTEX_BASE_URL = $GatewayUrl
        if ($EffectiveGatewayWire) {
            $env:CC_AGY_GATEWAY_WIRE = $EffectiveGatewayWire
        }
        if ($GatewayAudience) {
            $env:CC_AGY_VERTEX_BASE_URL_AUDIENCE = $GatewayAudience
        }
        if ($TelemetrySink) {
            $env:CC_AGY_TELEMETRY_SINK = $TelemetrySink
        }
        & $PythonCmd @DoctorArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Post-install doctor check exited with code $LASTEXITCODE."
        }
    }
} finally {
    Pop-Location
}
