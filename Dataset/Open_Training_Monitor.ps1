$ErrorActionPreference = 'Stop'
$monitorUrl = 'http://127.0.0.1:55318/'
$monitorReady = $false
try {
    $monitorCheck = Invoke-RestMethod -Uri ($monitorUrl + 'api/state') -TimeoutSec 4
    $monitorReady = $null -ne $monitorCheck.runs
} catch {}
if (-not $monitorReady) {
    $monitorTunnel = Start-Process -FilePath 'C:\Windows\System32\OpenSSH\ssh.exe' -ArgumentList '-N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:55318:127.0.0.1:55318 trinity-2-23' -WindowStyle Hidden -PassThru
    for ($monitorAttempt = 0; $monitorAttempt -lt 12; $monitorAttempt++) {
        Start-Sleep -Seconds 2
        try {
            $monitorCheck = Invoke-RestMethod -Uri ($monitorUrl + 'api/state') -TimeoutSec 3
            if ($null -ne $monitorCheck.runs) { $monitorReady = $true; break }
        } catch {}
        if ($monitorTunnel.HasExited) { throw 'SSH tunnel exited. Check access to trinity-2-23 and local port 55318.' }
    }
    if (-not $monitorReady) { throw 'Dashboard connection timed out. Remote training continues independently.' }
}
Start-Process $monitorUrl
