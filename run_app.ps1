$ErrorActionPreference = "Continue"

Set-Location -Path $PSScriptRoot

Write-Host "Starting quant-factor-system Streamlit App..."

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating local virtual environment..."
    & $venvActivate
} else {
    Write-Host "No .venv detected. Using current Python environment."
}

streamlit run app/streamlit_app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start Streamlit. Please check whether dependencies are installed."
}
