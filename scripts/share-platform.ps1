param(
    [ValidateSet('Start','Status','Stop')][string]$Operation='Start',
    [ValidatePattern('^[a-z0-9][a-z0-9_-]*$')][string]$Project='ai07-portable',
    [switch]$Prepare
)
$ErrorActionPreference='Stop'
$shareRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $shareRoot
# Keep authentication enabled after sharing ends. Never publish the anonymous research view.
if($Operation -eq 'Start'){$env:AI07_REQUIRE_LOGIN='1';$env:AI07_SECURE_COOKIE='1'}
. "$shareRoot/start.ps1" -Action Status -Project $Project
$shareService="${Project}-public-tunnel-1"
function Read-LocalJson([string]$Route) {
    $request=[Net.HttpWebRequest]::Create("http://127.0.0.1:$env:AI07_API_PORT$Route")
    $request.Proxy=$null;$request.Timeout=10000
    $response=$request.GetResponse()
    try{
        $reader=[IO.StreamReader]::new($response.GetResponseStream())
        try{return ($reader.ReadToEnd()|ConvertFrom-Json)}finally{$reader.Dispose()}
    }finally{$response.Dispose()}
}
function Read-ShareUrl {
    $ErrorActionPreference='Continue'
    $lines=(& docker logs --tail 250 $shareService 2>&1|Out-String)
    $urls=[regex]::Matches($lines,'https://[a-z0-9-]+\.trycloudflare\.com\b')
    if($urls.Count){return $urls[$urls.Count-1].Value}
    return $null
}
function Test-TunnelReady {
    $probe="import urllib.request;print('ready' if urllib.request.urlopen('http://public-tunnel:2000/ready',timeout=3).status==200 else 'pending')"
    $ErrorActionPreference='Continue'
    $output=& docker @composeArgs exec -T platform-api python -c $probe 2>$null
    return $LASTEXITCODE -eq 0 -and $output -eq 'ready'
}
if($Operation -eq 'Stop'){
    Invoke-Compose -f compose.share.yaml --profile sharing stop public-tunnel
    Write-Output 'Remote sharing stopped. Platform accounts and data are preserved; login remains required.'
    return
}
if($Operation -eq 'Status'){
    $running=& docker ps -q --filter "name=^/$shareService$"
    if($running -and (Test-TunnelReady)){Write-Output (Read-ShareUrl)}else{Write-Output 'No ready remote sharing tunnel. Start sharing again or inspect tunnel connectivity.'}
    return
}
$initialAuth=Read-LocalJson '/api/auth'
if($initialAuth.setup_required){throw 'Create your administrator account locally before enabling remote sharing.'}
if($Prepare -or -not($initialAuth.PSObject.Properties.Name -contains 'require_login')){
    Build-Compose build platform-api platform-web
}
$active=Invoke-Compose exec -T platform-api python -c "from platform_api.db import Session,Job;from sqlalchemy import select,func;s=Session();print(s.scalar(select(func.count()).select_from(Job).where(Job.status.in_(['queued','running']))));s.close()"
if([int](($active|Out-String).Trim()) -gt 0){throw 'Finish or cancel active generation jobs, then rerun sharing. No tunnel has been opened.'}
Invoke-Compose up -d --wait --no-build --pull never platform-api platform-web
$guarded=Read-LocalJson '/api/auth'
if(-not $guarded.require_login -or $guarded.setup_required){throw 'Private login mode is not ready. Remote sharing refused.'}
# The official tool runs on the same Docker network and only forwards the web service.
$env:AI07_TUNNEL_IMAGE='cloudflare/cloudflared:latest'
Ensure-Image $env:AI07_TUNNEL_IMAGE
Invoke-Compose -f compose.share.yaml --profile sharing up -d --no-build --pull never public-tunnel
$shareUrl=$null
for($attempt=0;$attempt -lt 40;$attempt++){
    $shareUrl=Read-ShareUrl
    if($shareUrl -and (Test-TunnelReady)){break}
    $shareUrl=$null
    Start-Sleep -Seconds 2
}
if(-not $shareUrl){
    Invoke-Compose -f compose.share.yaml --profile sharing stop public-tunnel
    throw 'No connected HTTPS tunnel is ready. Check Cloudflare connectivity (outbound UDP or TCP 7844) and rerun; the tunnel was stopped.'
}
$origins=@($env:AI07_ALLOWED_ORIGINS.Split(',')|Where-Object {$_ -and $_ -notmatch '^https://[a-z0-9-]+\.trycloudflare\.com$'})
$env:AI07_ALLOWED_ORIGINS=(@($origins)+@($shareUrl)|Select-Object -Unique)-join ','
Read-Settings
Invoke-Compose up -d --wait --no-build --pull never platform-api platform-web
Invoke-Compose exec -T platform-web nginx -s reload
Write-Output "Remote platform: $shareUrl"
Write-Output 'Share this HTTPS URL with team members. They sign in with platform accounts; no Docker installation or API URL is needed.'
Write-Output 'Keep this computer awake and Docker running. This temporary URL changes when the tunnel is recreated.'
Write-Output 'Stop sharing: powershell -ExecutionPolicy Bypass -File scripts/share-platform.ps1 -Operation Stop'
