param(
    [string]$Config = "config/v2.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$dotenvPath = Join-Path $repoRoot ".env"
if ((-not $env:DEEPSEEK_API_KEY) -and (Test-Path $dotenvPath)) {
    $dotenvLine = Get-Content $dotenvPath |
        Where-Object { $_ -match '^\s*DEEPSEEK_API_KEY\s*=' } |
        Select-Object -First 1
    if ($dotenvLine) {
        $env:DEEPSEEK_API_KEY = ($dotenvLine -replace '^\s*DEEPSEEK_API_KEY\s*=\s*', '').Trim().Trim('"').Trim("'")
    }
}

$srcPath = Join-Path $repoRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $srcPath
}

$python = $null
if ($env:RAG_V2_PYTHON) {
    $python = $env:RAG_V2_PYTHON
} elseif (Test-Path (Join-Path $repoRoot ".venv\Scripts\python.exe")) {
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
} else {
    $python = "python"
}

$pythonVersion = & $python --version 2>&1
if ($pythonVersion -notmatch "Python 3\.11") {
    throw "RAG v2 requires Python 3.11; found $pythonVersion. Set RAG_V2_PYTHON to the correct interpreter."
}

$configPath = (Resolve-Path $Config).Path
$manifestPath = Join-Path $repoRoot "artifacts\index_manifest.json"
if (-not (Test-Path $manifestPath)) {
    Write-Host "Dense index artifact not found; building it with Ollama..."
    & $python (Join-Path $repoRoot "scripts\build_v2_index.py") --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dense index build failed. Start Ollama and run: ollama pull nomic-embed-text"
    }
}

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    throw "Ollama is not reachable at http://127.0.0.1:11434. Start Ollama before launching the demo."
}

& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit is not installed in the selected Python environment."
}

if (-not $env:DEEPSEEK_API_KEY) {
    Write-Warning "DEEPSEEK_API_KEY is not set; the demo will show retrieval and abstain on generation."
}

$env:V2_CONFIG_PATH = $configPath
& $python -m streamlit run (Join-Path $repoRoot "demo\app.py")
exit $LASTEXITCODE
