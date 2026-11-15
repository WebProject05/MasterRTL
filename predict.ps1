param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$VerilogPath,
    [switch]$Plot,
    [string]$Transfer
)

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$argsList = @("scripts/predict.py", $VerilogPath)
if ($Plot) { $argsList += "--plot" }
if ($Transfer) { $argsList += @("--transfer", $Transfer) }

& $py @argsList
