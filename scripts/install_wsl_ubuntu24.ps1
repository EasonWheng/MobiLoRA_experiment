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
$tempExtract = Join-Path $downloadDir "ubuntu-import"

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

$imageInfo = Get-LatestUbuntuWslUrl
$downloadPath = Join-Path $downloadDir $imageInfo.FileName

if (-not (Test-Path $downloadPath)) {
    Write-Host "Downloading Ubuntu WSL image from $($imageInfo.Url)"
    Invoke-WebRequest -UseBasicParsing -Uri $imageInfo.Url -OutFile $downloadPath
} else {
    Write-Host "Using existing Ubuntu WSL image at $downloadPath"
}

if (wsl.exe -l -q | Select-String -SimpleMatch "Ubuntu-24.04") {
    Write-Host "A distro named Ubuntu-24.04 already exists. Skipping import."
    exit 0
}

if (Test-Path $tempExtract) {
    Remove-Item -LiteralPath $tempExtract -Recurse -Force
}
Ensure-Directory $tempExtract

Copy-Item -LiteralPath $downloadPath -Destination (Join-Path $tempExtract "ubuntu-24.04.wsl") -Force
Push-Location $tempExtract
tar.exe -xf ".\ubuntu-24.04.wsl"
Pop-Location

$rootFsCandidates = Get-ChildItem -LiteralPath $tempExtract -Filter "*.tar.gz" -Recurse
if ($rootFsCandidates.Count -eq 0) {
    throw "Failed to extract a rootfs tarball from the downloaded WSL image."
}

$rootFsPath = $rootFsCandidates[0].FullName
Write-Host "Importing Ubuntu-24.04 into $distroPath"
wsl.exe --import Ubuntu-24.04 $distroPath $rootFsPath --version 2 | Out-Host

Write-Host ""
Write-Host "WSL import finished."
Write-Host "Next steps:"
Write-Host "  1. Launch the distro: wsl.exe -d Ubuntu-24.04"
Write-Host "  2. Create your Linux user."
Write-Host "  3. Run: bash /mnt/d/learn_pytorch/SEU_DEEPLEARNING_LESSON_EXPERIMENTS/MobiLoRA_experiment/scripts/bootstrap_wsl_env.sh"
