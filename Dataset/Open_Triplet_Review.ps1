$ErrorActionPreference = 'Stop'
$tripletUrl = 'http://127.0.0.1:55317/'
$tripletReady = $false
try {
    $tripletCheck = Invoke-RestMethod -Uri ($tripletUrl + 'api/pairs?split=pilot&index=0') -TimeoutSec 5
    $tripletReady = $tripletCheck.all_total -eq 60000
} catch {}
if (-not $tripletReady) {
    & ssh -o BatchMode=yes -o ConnectTimeout=20 trinity-2-23 'tmux has-session -t photo-triplet-view 2>/dev/null || tmux new-session -d -s photo-triplet-view "/tmp/shihangz/photo-tone-transfer/Runtime/venv/bin/python -u /tmp/shihangz/photo-tone-transfer/Research/data_collection/triplet_view_server.py --root /tmp/shihangz/photo-tone-transfer --port 55317 > /tmp/shihangz/photo-tone-transfer/Logs/triplet_view.log 2>&1"'
    if ($LASTEXITCODE -ne 0) { throw 'Unable to start the compute-node viewer.' }
    $tripletTunnel = Start-Process -FilePath 'C:\Windows\System32\OpenSSH\ssh.exe' -ArgumentList '-N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:55317:127.0.0.1:55317 trinity-2-23' -WindowStyle Hidden -PassThru
    for ($tripletAttempt=0; $tripletAttempt -lt 15; $tripletAttempt++) {
        Start-Sleep -Seconds 2
        try {
            $tripletCheck = Invoke-RestMethod -Uri ($tripletUrl + 'api/pairs?split=pilot&index=0') -TimeoutSec 3
            if ($tripletCheck.all_total -eq 60000) { $tripletReady=$true; break }
        } catch {}
        if ($tripletTunnel.HasExited) { throw 'SSH tunnel exited. Check whether port 55317 is already occupied.' }
    }
    if (-not $tripletReady) { throw 'Viewer connection timed out.' }
}
Start-Process $tripletUrl
