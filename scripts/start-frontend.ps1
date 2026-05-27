
$RepoRoot = Split-Path -Parent $PSScriptRoot
$FrontendPath = Join-Path $RepoRoot "frontend"

Write-Host "Starting frontend dev server..."
if (-not (Test-Path -LiteralPath $FrontendPath)) {
	throw "Frontend folder not found: $FrontendPath"
}

Push-Location $FrontendPath
try {
	# Reload env
	Write-Host "Reloading environment variables..."
	$EnvFile = Join-Path $FrontendPath ".env"
	if (Test-Path -LiteralPath $EnvFile) {
		Get-Content -LiteralPath $EnvFile | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
			$parts = $_ -split '=', 2
			[System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim().Trim('"'), 'Process')
		}
	} else {
		Write-Host ".env not found, skipping env reload."
	}

	# Run
	Write-Host "Running frontend dev server..."
	npm run dev
} finally {
	Write-Host "Stopping frontend dev server..."
	Pop-Location
}
