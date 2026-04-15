@echo off
set "CLAUDE_ADAPTER_ROOT=%~dp0.."
start "Claude Adapter Env" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8787'; $env:ANTHROPIC_AUTH_TOKEN='claude_adapter'; Set-Location '%CLAUDE_ADAPTER_ROOT%'; Write-Host 'Claude adapter environment loaded.'; Write-Host 'ANTHROPIC_BASE_URL=' $env:ANTHROPIC_BASE_URL; Write-Host 'ANTHROPIC_AUTH_TOKEN=' $env:ANTHROPIC_AUTH_TOKEN"
