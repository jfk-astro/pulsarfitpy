# Build script for Windows

param(
    [switch]$Clean,
    [switch]$Run,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$BIN_DIR = "bin"
$OUTPUT = "$BIN_DIR/pulsar-cli.exe"
$MAIN_PATH = "./cmd/pulsar-cli"

function Show-Help {
    Write-Host @"
pulsarfitpy Build Script

Usage:
  .\build.ps1              Build the TUI application
  .\build.ps1 -Run         Build and run the application
  .\build.ps1 -Clean       Clean build artifacts
  .\build.ps1 -Help        Show this help message

Examples:
  .\build.ps1              # Just build
  .\build.ps1 -Run         # Build and run
  .\build.ps1 -Clean       # Clean bin directory

"@
}

function Clean-Build {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path $BIN_DIR) {
        Remove-Item -Recurse -Force $BIN_DIR
        Write-Host "Cleaned $BIN_DIR" -ForegroundColor Green
    } else {
        Write-Host "Nothing to clean" -ForegroundColor Gray
    }
}

function Build-App {
    Write-Host "Building pulsarfitpy TUI..." -ForegroundColor Cyan
    
    if (-not (Test-Path $BIN_DIR)) {
        New-Item -ItemType Directory -Path $BIN_DIR | Out-Null
    }
    
    Write-Host "Checking dependencies..." -ForegroundColor Gray
    go mod download
    
    Write-Host "Compiling..." -ForegroundColor Gray
    go build -o $OUTPUT $MAIN_PATH
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful: $OUTPUT" -ForegroundColor Green
        
        $size = (Get-Item $OUTPUT).Length / 1MB
        Write-Host "Binary size: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
        
        return $true
    } else {
        Write-Host "Build failed" -ForegroundColor Red
        return $false
    }
}

function Run-App {
    if (Test-Path $OUTPUT) {
        Write-Host "`nStarting application..." -ForegroundColor Cyan
        Write-Host "----------------------------------------" -ForegroundColor Gray
        & $OUTPUT
    } else {
        Write-Host "Executable not found: $OUTPUT" -ForegroundColor Red
        Write-Host "Run without -Run flag to build first" -ForegroundColor Yellow
        exit 1
    }
}

if ($Help) {
    Show-Help
    exit 0
}

if ($Clean) {
    Clean-Build
    exit 0
}

$buildSuccess = Build-App

if ($buildSuccess -and $Run) {
    Run-App
}

if (-not $buildSuccess) {
    exit 1
}
