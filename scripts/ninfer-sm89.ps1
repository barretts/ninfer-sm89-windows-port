param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Start'
)

$ErrorActionPreference = 'Stop'
$containerName = 'ninfer-sm89-64k'
$imageName = 'ninfer-sm89-toolchain'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$binaryPath = Join-Path $repoPath 'build-sm89/apps/ninfer-serve'
$modelRoot = 'E:\modelmadness\ninfer-5080'
$modelPath = Join-Path $modelRoot 'qwen3_8_27b.ninfer'
$api = 'http://127.0.0.1:18088/v1'

function Get-ContainerState {
    $state = & docker container inspect --format '{{.State.Status}}' $containerName 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($state | Select-Object -First 1)
}

switch ($Action) {
    'Status' {
        $state = Get-ContainerState
        if (-not $state) { $state = 'absent' }
        Write-Host "$containerName : $state"
        if ($state -eq 'running') { Write-Host "API: $api" }
        return
    }
    'Stop' {
        if ((Get-ContainerState) -eq 'running') {
            & docker stop $containerName
            if ($LASTEXITCODE -ne 0) { throw "Could not stop $containerName" }
        } else {
            Write-Host "$containerName is not running"
        }
        return
    }
}

if (-not (Test-Path -LiteralPath $binaryPath)) {
    throw "Build the SM89 server first: $binaryPath"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Model artifact is missing: $modelPath"
}

$state = Get-ContainerState
if ($state -and $state -ne 'running') {
    & docker rm $containerName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not remove stopped $containerName" }
    $state = $null
}
if ($state -ne 'running') {
    $args = @(
        'run', '-d', '--gpus', 'all', '--name', $containerName,
        '--mount', "type=bind,source=$repoPath,target=/src",
        '--mount', "type=bind,source=$modelRoot,target=/model,readonly",
        '--publish', '127.0.0.1:18088:8080',
        $imageName,
        '/src/build-sm89/apps/ninfer-serve', '/model/qwen3_8_27b.ninfer',
        '--host', '0.0.0.0', '--port', '8080',
        '--model-id', 'qwen3.8-27b-sm89',
        '--max-context', '65536', '--kv-capacity', '65536',
        '--prefill-chunk', '896', '--kv-dtype', 'q4',
        '--spec', 'mtp', '--draft-tokens', '3', '--no-cuda-graph',
        '--max-concurrency', '1', '--default-thinking-budget', '2048',
        '--prefix-checkpoint-policy', 'rolling-tool',
        '--vision', '--vision-max-tokens', '1792'
    )
    & docker @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Docker failed to start the NInfer container' }
}

$deadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $deadline) {
    if ((Get-ContainerState) -ne 'running') {
        throw "Server exited during startup. Inspect: docker logs $containerName"
    }
    try {
        $models = Invoke-RestMethod -Uri "$api/models" -TimeoutSec 3
        if ($models.data.id -contains 'qwen3.8-27b-sm89') {
            Write-Host "Ready: $api  (model qwen3.8-27b-sm89; 64K context/KV)"
            return
        }
    } catch { }
    Start-Sleep -Seconds 3
}
throw "Server did not become ready in four minutes. Inspect: docker logs $containerName"
