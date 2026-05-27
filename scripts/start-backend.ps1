# Remove old data (data folder, except file data/README.md)
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataPath = Join-Path $RepoRoot "data"

Write-Host "Cleaning up old data..."
if (Test-Path -LiteralPath $DataPath) {
	Get-ChildItem -LiteralPath $DataPath -Force | Where-Object { $_.Name -ne "README.md" } | ForEach-Object {
		Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
	}
} else {
	Write-Host "Data folder not found, skipping cleanup."
}

Write-Host "Starting backend server..."
$BackendPath = Join-Path $RepoRoot "backend"
if (-not (Test-Path -LiteralPath $BackendPath)) {
	throw "Backend folder not found: $BackendPath"
}

Push-Location $BackendPath
try {
	# Reload env
	Write-Host "Reloading environment variables..."
	$EnvFile = Join-Path $BackendPath ".env"
	if (Test-Path -LiteralPath $EnvFile) {
		Get-Content -LiteralPath $EnvFile | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
			$parts = $_ -split '=', 2
			[System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim().Trim('"'), 'Process')
		}
	} else {
		Write-Host ".env not found, skipping env reload."
	}

	# Run
	Write-Host "Running backend server..."
	uv run uvicorn app.robot_agent.main:app --reload --port 8000
} finally {
	# Redirect to root when stopping
	Write-Host "Stopping backend server..."
	Pop-Location
}