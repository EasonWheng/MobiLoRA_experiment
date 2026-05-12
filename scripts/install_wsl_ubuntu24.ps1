param(
    [string]$LocalImagePath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Administrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Please re-run this script in an elevated PowerShell window."
    }
}

function Get-LatestUbuntuWslUrl {
    $releasePage = "https://releases.ubuntu.com/noble/"
    $response = Invoke-WebRequest -UseBasicParsing -Uri $releasePage
    $matches = [regex]::Matches($response.Content, 'ubuntu-24\.04\.\d+-wsl-amd64\.wsl')
    if ($matches.Count -eq 0) {
        throw "Could not locate a Ubuntu 24.04 WSL image on $releasePage"
    }

    $latestName = $matches[$matches.Count - 1].Value
    return [pscustomobject]@{
        Page = $releasePage
        FileName = $latestName
        Url = "$releasePage$latestName"
    }
}

function Ensure-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Find-ExistingUbuntuWslImage([string]$DirectoryPath) {
    if (-not (Test-Path $DirectoryPath)) {
        return $null
    }

    $candidates = @(
        Get-ChildItem -LiteralPath $DirectoryPath -Filter "ubuntu-24.04.*-wsl-amd64.wsl" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending
    )
    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }
    return $null
}

function Download-File([string]$Url, [string]$DestinationPath) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        Write-Host "Downloading with curl.exe to $DestinationPath"
        & $curl.Source -L --retry 8 --retry-delay 5 --connect-timeout 30 -o $DestinationPath $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE while downloading $Url"
        }
        return
    }

    Write-Host "Downloading with Invoke-WebRequest to $DestinationPath"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $DestinationPath -TimeoutSec 0
}

function Test-PendingReboot {
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
    )
    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $true
        }
    }
    return $false
}

function Invoke-DismFeatureEnable([string]$FeatureName) {
    Write-Host "Enabling Windows feature: $FeatureName"
    & dism.exe /online /enable-feature /featurename:$FeatureName /all /norestart | Out-Host
    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw "DISM failed while enabling feature '$FeatureName' with exit code $LASTEXITCODE."
    }
    return ($LASTEXITCODE -eq 3010)
}

Assert-Administrator

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$wslRoot = "D:\WSL"
$distroPath = Join-Path $wslRoot "Ubuntu-24.04"
$downloadDir = Join-Path $wslRoot "downloads"

Ensure-Directory $wslRoot
Ensure-Directory $distroPath
Ensure-Directory $downloadDir

Write-Host "Enabling required Windows features for WSL2..."
$restartRequested = $false
$restartRequested = (Invoke-DismFeatureEnable "Microsoft-Windows-Subsystem-Linux") -or $restartRequested
$restartRequested = (Invoke-DismFeatureEnable "VirtualMachinePlatform") -or $restartRequested

if ($restartRequested -or (Test-PendingReboot)) {
    Write-Host ""
    Write-Host "A reboot is required before WSL can finish installing."
    Write-Host "Please restart Windows, then rerun this same script from an elevated terminal:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSScriptRoot\install_wsl_ubuntu24.ps1`""
    exit 0
}

Write-Host "Installing the WSL runtime without a default distro..."
& wsl.exe --install --no-distribution | Out-Host
if ($LASTEXITCODE -notin @(0, 3010)) {
    throw "wsl --install --no-distribution failed with exit code $LASTEXITCODE."
}
if ($LASTEXITCODE -eq 3010 -or (Test-PendingReboot)) {
    Write-Host ""
    Write-Host "WSL runtime installation requested a reboot."
    Write-Host "Please restart Windows, then rerun this same script."
    exit 0
}

& wsl.exe --set-default-version 2 | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "wsl --set-default-version 2 failed with exit code $LASTEXITCODE."
}

$downloadPath = $null

if ($LocalImagePath) {
    $resolvedLocalImagePath = Resolve-Path -LiteralPath $LocalImagePath -ErrorAction Stop
    $downloadPath = $resolvedLocalImagePath.Path
    Write-Host "Using manually provided Ubuntu WSL image at $downloadPath"
} else {
    $existingImagePath = Find-ExistingUbuntuWslImage $downloadDir
    if ($existingImagePath) {
        $downloadPath = $existingImagePath
        Write-Host "Using existing Ubuntu WSL image at $downloadPath"
    } else {
        $imageInfo = Get-LatestUbuntuWslUrl
        $downloadPath = Join-Path $downloadDir $imageInfo.FileName
        Write-Host "Downloading Ubuntu WSL image from $($imageInfo.Url)"
        Download-File -Url $imageInfo.Url -DestinationPath $downloadPath
    }
}

if (wsl.exe -l -q | Select-String -SimpleMatch "Ubuntu-24.04") {
    Write-Host "A distro named Ubuntu-24.04 already exists. Skipping import."
    exit 0
}

Write-Host "Installing Ubuntu-24.04 from local .wsl image into $distroPath"
& wsl.exe --install --from-file $downloadPath --name Ubuntu-24.04 --location $distroPath --no-launch | Out-Host
if ($LASTEXITCODE -notin @(0, 3010)) {
    throw "wsl --install --from-file failed with exit code $LASTEXITCODE."
}
if ($LASTEXITCODE -eq 3010 -or (Test-PendingReboot)) {
    Write-Host ""
    Write-Host "Ubuntu-24.04 installation requested a reboot."
    Write-Host "Please restart Windows, then rerun this same script."
    exit 0
}

Write-Host ""
Write-Host "Ubuntu-24.04 installation finished."
Write-Host "Next steps:"
Write-Host "  1. Launch the distro: wsl.exe -d Ubuntu-24.04"
Write-Host "  2. Create your Linux user."
Write-Host "  3. Run: bash /mnt/d/learn_pytorch/SEU_DEEPLEARNING_LESSON_EXPERIMENTS/MobiLoRA_experiment/scripts/bootstrap_wsl_env.sh"
