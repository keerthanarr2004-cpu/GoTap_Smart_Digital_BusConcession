Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting GoTap Smart Student Transportation System" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Set-Location -Path $PSScriptRoot
& ".\.venv\Scripts\python.exe" manage.py runserver
