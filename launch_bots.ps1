# launch_bots.ps1
# Launches EURUSD, US500 and XAUUSD trading bots in separate terminal windows.
# Run from the project root:  .\launch_bots.ps1
# Run files are not tracked. See Engine/run_multiclass.py for run template.

$root   = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Warning ".venv Python not found at '$python' - falling back to system 'python'."
    $python = "python"
}

Write-Host "Launching EURUSD bot..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "& { `$host.UI.RawUI.WindowTitle = 'BOT - EURUSD'; Set-Location '$root'; & '$python' 'Engine/.run_EURUSD.py' }"

Write-Host "Launching XAUUSD bot..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "& { `$host.UI.RawUI.WindowTitle = 'BOT - XAUUSD'; Set-Location '$root'; & '$python' 'Engine/.run_XAUUSD.py' }"

Write-Host "Launching US500 bot..." -ForegroundColor Red
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "& { `$host.UI.RawUI.WindowTitle = 'BOT - US500'; Set-Location '$root'; & '$python' 'Engine/.run_US500.py' }"

Write-Host "All bots launched in separate windows." -ForegroundColor Green
