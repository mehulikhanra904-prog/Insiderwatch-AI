$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== InsiderWatch AI Setup / Repair ===" -ForegroundColor Cyan

if (-not (Test-Path ".\app.py")) {
    throw "app.py not found. Open PowerShell in the InsiderWatch AI project directory."
}
if (-not (Test-Path ".\requirements.txt")) {
    throw "requirements.txt not found in the current directory."
}

# Never move or delete the existing .venv. Use a separate environment instead.
$envFolder = ".venv-insiderwatch"
if (-not (Test-Path "$envFolder\Scripts\python.exe")) {
    Write-Host "Creating a separate Python environment..." -ForegroundColor Cyan
    py -3.12 -m venv $envFolder
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create Python 3.12 environment. Run 'py -0p' to see installed versions."
    }
}

$python = Join-Path (Resolve-Path $envFolder).Path "Scripts\python.exe"

Write-Host "Installing project dependencies..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Checking imports..." -ForegroundColor Cyan
& $python -c "import streamlit, pandas, numpy, plotly, sklearn; import detection; print('SUCCESS: imports are working')"
if ($LASTEXITCODE -ne 0) { throw "Import check failed." }

Write-Host "Running unit tests..." -ForegroundColor Cyan
& $python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "One or more tests failed. Review the output above." }

if (-not (Test-Path ".\activity_data.csv")) {
    Write-Host "Note: activity_data.csv is not present. You can upload a compatible CSV in the dashboard." -ForegroundColor Yellow
}

Write-Host "Starting InsiderWatch AI..." -ForegroundColor Green
& $python -m streamlit run app.py
