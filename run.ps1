# OrbitX PowerShell Quick Launcher
param (
    [switch]$Dashboard,
    [switch]$Terminal,
    [switch]$Bench,
    [switch]$Test,
    [switch]$Report,
    [int]$Port = 8000,
    [string]$Agent = "ppo"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runPy = Join-Path $scriptDir "run.py"

$argsList = @()
if ($Dashboard) { $argsList += "--dashboard"; $argsList += "--port"; $argsList += $Port }
elseif ($Terminal) { $argsList += "--terminal"; $argsList += "--agent"; $argsList += $Agent }
elseif ($Bench) { $argsList += "--bench" }
elseif ($Test) { $argsList += "--test" }
elseif ($Report) { $argsList += "--report" }

python $runPy $argsList
