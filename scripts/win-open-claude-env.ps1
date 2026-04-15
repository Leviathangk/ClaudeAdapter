$ErrorActionPreference = "Stop"
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN = "claude_adapter"

$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

Write-Host "Claude adapter environment loaded."
Write-Host "ANTHROPIC_BASE_URL=$env:ANTHROPIC_BASE_URL"
Write-Host "ANTHROPIC_AUTH_TOKEN=$env:ANTHROPIC_AUTH_TOKEN"
