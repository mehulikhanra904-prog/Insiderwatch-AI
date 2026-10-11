$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== InsiderWatch AI Windows Repair ===" -ForegroundColor Cyan

if (-not (Test-Path "app.py")) {
    throw "app.py not found. Open PowerShell in the InsiderWatch AI project folder."
}

# Preserve existing files before syncing the known-good version from GitHub.
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Copy-Item "app.py" "app.py.backup-$stamp" -Force
if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" "requirements.txt.backup-$stamp" -Force
}

# Get the latest repository version so stale local code (including df-before-load)
# cannot continue to trigger the old error.
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "git fetch failed. Check your internet connection and GitHub access." }
git checkout origin/main -- app.py requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Could not sync app.py and requirements.txt from origin/main." }

# Keep the old virtual environment as a backup; create a clean one to remove
# broken/partial installs such as the '~ip' invalid-distribution warning.
if (Test-Path ".venv") {
    $backupVenv = ".venv-broken-$stamp"
    Move-Item ".venv" $backupVenv
    Write-Host "Old virtual environment preserved as $backupVenv" -ForegroundColor Yellow
}

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 was not found; trying the default Python installation..." -ForegroundColor Yellow
    py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create a virtual environment. Install Python 3.12 or 3.13 and retry." }
}

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Copy the full pip error and send it to ChatGPT." }

& $python -c "import streamlit, pandas, numpy, plotly, sklearn; print('SUCCESS: all required imports work')"
if ($LASTEXITCODE -ne 0) { throw "One or more imports still fail. Copy the full error output." }

if (-not (Test-Path "activity_data.csv")) {
    throw "activity_data.csv is missing from this folder. Keep your existing dataset here; this script will not invent or overwrite it."
}

Write-Host ""
Write-Host "Repair checks passed. Starting InsiderWatch AI..." -ForegroundColor Green
& $python -m streamlit run app.py
