param(
    [ValidateSet('Start','Setup','Stop','Status','Backup','Restore','ListBackups','Export','Import')][string]$Action='Start',
    [switch]$InstallDocker,
    [switch]$Gpu,
    [switch]$Rebuild,
    [ValidatePattern('^[a-z0-9][a-z0-9_-]*$')][string]$Project='ai07-portable',
    [int]$Port=0,[int]$ApiPort=0,[int]$LabelPort=0,
    [string]$Backup='', [string]$Path=''
)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
Set-Location -LiteralPath $projectRoot
if(-not(Get-Command docker -ErrorAction SilentlyContinue)){
    if(-not $InstallDocker){throw 'Docker is required. Run: powershell -ExecutionPolicy Bypass -File .\start.ps1 -InstallDocker'}
    & winget install --id Docker.DockerDesktop --exact --accept-source-agreements --accept-package-agreements
    if($LASTEXITCODE -ne 0){throw 'Docker Desktop installation failed'}
    $env:Path+=";$env:ProgramFiles\Docker\Docker\resources\bin"
}
function Test-DockerEngine {
    $ErrorActionPreference='Continue'
    & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}
if(-not(Test-DockerEngine)){
    $desktop="$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if(Test-Path -LiteralPath $desktop){Start-Process -FilePath $desktop -WindowStyle Hidden}
    $ready=$false
    for($attempt=0;$attempt -lt 60;$attempt++){
        Start-Sleep -Seconds 2
        if(Test-DockerEngine){$ready=$true;break}
    }
    if(-not $ready){throw 'Start Docker Desktop Linux engine. A first-time WSL2 installation may require a Windows restart.'}
}
function Invoke-Docker {
    & docker @args
    if($LASTEXITCODE -ne 0){throw "Docker command failed (exit $LASTEXITCODE)"}
}
function Test-LocalImage([string]$Image) {
    $ErrorActionPreference='Continue'
    & docker image inspect $Image --format '{{.Id}}' 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}
function Invoke-DockerDownload([string[]]$CommandArguments) {
    $ErrorActionPreference='Continue'
    for($attempt=1;$attempt -le 4;$attempt++){
        $networkFailure=$false
        & docker @CommandArguments 2>&1 | ForEach-Object {
            $line=$_.ToString()
            if($line -match '(?i)\bEOF\b|TLS handshake|i/o timeout|connection reset|connection refused|no such host|network is unreachable|temporary failure|502|503|504|unexpected end|timed out'){$networkFailure=$true}
            Write-Host $line
        }
        $commandExit=$LASTEXITCODE
        if($commandExit -eq 0){return}
        if(-not $networkFailure -or $attempt -eq 4){
            throw "Docker download/build failed (exit $commandExit). For Docker Hub EOF/TLS errors, check Docker Desktop network/proxy connectivity and rerun. Existing data volumes are preserved."
        }
        Write-Host "Registry/network connection interrupted; retrying ($attempt/4)..."
        Start-Sleep -Seconds (2*$attempt)
    }
}
function Ensure-Image([string]$Image) {
    if(-not(Test-LocalImage $Image)){Invoke-DockerDownload -CommandArguments @('pull',$Image)}
}
$env:COMPOSE_PROJECT_NAME=$Project
if($Port){$env:AI07_WEB_PORT=[string]$Port}
if($ApiPort){$env:AI07_API_PORT=[string]$ApiPort}
if($LabelPort){$env:AI07_LS_PORT=[string]$LabelPort}
if($Gpu){$env:AI07_GPU='1'}
$initArgs=@('run','--rm','-i','--mount',"type=volume,source=${Project}_credentials,target=/state",'--mount',"type=volume,source=${Project}_platform-state,target=/platform",'--mount',"type=volume,source=${Project}_backups,target=/backups")
foreach($key in @('AI07_WEB_PORT','AI07_API_PORT','AI07_LS_PORT','AI07_GPU','AI07_BIND_ADDRESS','AI07_ALLOWED_ORIGINS','AI07_LABEL_STUDIO_PUBLIC_URL','AI07_SECURE_COOKIE','AI07_REQUIRE_LOGIN')){
    $initArgs+=@('-e',$key)
}
$pythonImage='python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1'
Ensure-Image $pythonImage
function Read-Settings {
    $settings=Get-Content -LiteralPath "$projectRoot/scripts/portable-init.py" -Raw | & docker @initArgs $pythonImage python -
    if($LASTEXITCODE -ne 0){throw 'Credential initialization failed'}
    foreach($line in $settings){if($line -match '^([A-Z0-9_]+)=(.*)$'){[Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process')}}
}
Read-Settings
$composeArgs=@('compose','--project-name',$Project,'-f','compose.portable.yaml')
if($env:AI07_GPU -eq '1'){
    $env:AI07_WORKER_BASE_IMAGE="${Project}-gpu:local"
    $composeArgs+=@('-f','compose.portable.gpu.yaml')
}else{$env:AI07_WORKER_BASE_IMAGE="${Project}-api:local"}
function Invoke-Compose {Invoke-Docker @composeArgs @args}
function Build-Compose {Invoke-DockerDownload -CommandArguments ($composeArgs+$args)}
function Test-PreparedRuntime {
    foreach($image in @("${Project}-api:local","${Project}-web:local","${Project}-worker:local")){
        if(-not(Test-LocalImage $image)){return $false}
    }
    $probe="import os,importlib.util;from pathlib import Path;gpu=os.getenv('AI07_CHECK_GPU')=='1';cuda=bool(__import__('torch').version.cuda) if importlib.util.find_spec('torch') else False;paths=['/app/artifacts/scene-assets/library/wooden_crate_01.glb'];paths+=['/app/models/sd15/model_index.json','/app/models/sd15/unet/diffusion_pytorch_model.fp16.safetensors','/app/models/yolo11n.pt','/app/artifacts/torch-cache/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'] if gpu else [];print('ready' if cuda==gpu and all(Path(p).is_file() for p in paths) else 'missing')"
    $result=& docker run --rm --pull never --entrypoint python -e "AI07_CHECK_GPU=$env:AI07_GPU" --mount "type=volume,source=${Project}_models,target=/app/models,readonly" --mount "type=volume,source=${Project}_artifacts,target=/app/artifacts,readonly" "${Project}-worker:local" -c $probe
    if($LASTEXITCODE -ne 0){throw 'Cannot check the prepared runtime'}
    return $result -eq 'ready'
}
$services=@('platform-web','platform-api','worker','annotation-platform')
switch($Action){
    {$_ -in 'Start','Setup'} {
        Ensure-Image 'postgres:16-alpine'
        Ensure-Image 'heartexlabs/label-studio:1.22.0'
        $prepare=$Rebuild -or $Action -eq 'Setup'
        if(-not $prepare){$prepare=-not(Test-PreparedRuntime)}
        if($prepare){
            Build-Compose --profile setup build assets platform-api
            if(-not(Test-LocalImage "${Project}-engines:fds6.11.1-blender4.5.3")){Build-Compose --profile setup build engines}
            if($env:AI07_GPU -eq '1'){
                Build-Compose --profile gpu build gpu
                foreach($asset in @('sd15','yolo','fid')){Invoke-Compose --profile setup run --rm --no-deps --pull never assets --only $asset}
            }
            Build-Compose build worker platform-web
            Invoke-Compose run --rm --no-deps --pull never --entrypoint blender worker --background --python /app/tools/create-scene-models.py
        }else{
            Write-Output 'Using prepared local images and data. Skipping registry access and image builds; use -Rebuild after changing source.'
        }
        if($Action -eq 'Start'){
            Invoke-Compose up -d --wait --no-build --pull never @services
            Write-Output "Workbench: http://localhost:$env:AI07_WEB_PORT  API: http://localhost:$env:AI07_API_PORT/docs"
            Write-Output 'New installation: choose your administrator username and password. Existing installations keep their accounts. Data lives in Docker named volumes.'
        }
    }
    Stop {Invoke-Compose stop}
    Status {Invoke-Compose ps}
    ListBackups {Invoke-Compose --profile tools run --rm --no-deps backup list}
    {$_ -in 'Backup','Restore'} {
        if($Action -eq 'Restore' -and -not $Backup){throw 'Restore requires -Backup name.tar; it replaces target stack data.'}
        Invoke-Compose stop @services
        $completed=$false
        try{
            Invoke-Compose up -d --wait platform-db
            $backupArgs=@($Action.ToLower());if($Backup){$backupArgs+=@('--name',$Backup)}
            Invoke-Compose --profile tools run --rm --no-deps backup @backupArgs
            $completed=$true
        }finally{
            if($completed -or $Action -eq 'Backup'){
                Read-Settings
                Invoke-Compose up -d --wait @services
            }else{Write-Warning 'Restore failed. Services remain stopped; inspect the error and retry restore.'}
        }
    }
    {$_ -in 'Export','Import'} {
        if($Backup -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*\.tar$' -or -not $Path){throw 'Specify -Backup name.tar and -Path directory (Export) or archive file (Import).'}
        $transferId=(& docker create --mount "type=volume,source=${Project}_backups,target=/backups" $pythonImage python -c pass)
        if($LASTEXITCODE -ne 0){throw 'Cannot create transfer container'}
        try{
            if($Action -eq 'Export'){
                $destination=[IO.Path]::GetFullPath($Path);[IO.Directory]::CreateDirectory($destination)|Out-Null
                Invoke-Docker cp "${transferId}:/backups/$Backup" (Join-Path $destination $Backup)
                Invoke-Docker cp "${transferId}:/backups/$Backup.sha256" (Join-Path $destination "$Backup.sha256")
            }else{
                $source=(Resolve-Path -LiteralPath $Path).Path
                Invoke-Docker cp $source "${transferId}:/backups/$Backup"
                Invoke-Docker cp "$source.sha256" "${transferId}:/backups/$Backup.sha256"
            }
        }finally{Invoke-Docker rm $transferId | Out-Null}
    }
}
