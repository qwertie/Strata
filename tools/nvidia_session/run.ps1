<#
.SYNOPSIS
The code-sharing refactor's phase B test session on an NVIDIA PC: builds Strata's engines as they were before the
refactor and as they are now, runs both the same way, and says where they differ. Read README.md beside this file.
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -Only multi -Results .\results\20261101-0900   # one phase again, same folder
#>
param(
    [string]$DataDir,                # a copy of the owner's Strata-data folder with the files README.md lists
    [string]$DeepSeekDir,            # the folder of the DeepSeek V4.1 GGUF files
    [ValidateSet('preflight', 'build', 'unit', 'smoke', 'multi', 'deepseek', 'speed', 'emulate')]
    [string[]]$Only = @('preflight', 'build', 'unit', 'smoke', 'multi', 'deepseek', 'speed', 'emulate'),
    [string]$Before = '87d20a5c',    # the last commit before any C++ moved
    [string]$Work = "$env:SystemDrive\sn",   # source copies and builds; short, because CUDA builds hit MAX_PATH
    [string]$Results,                # a results folder to continue in (finished runs are kept)
    [int]$Jobs = [Environment]::ProcessorCount,
    [switch]$Quick,                  # fewer runs and fewer variants: about half the time
    [switch]$DryRun                  # print every command, run none
)

#: The B3 checks on two cards (docs/REFACTOR_PLAN.md, the B test list): name | card orders (S = the smaller card,
#: B = the bigger, the first one is CUDA0) | engine arguments | settings. -Quick keeps the rows marked '*', in their
#: first order, one run each.
$MultiChecks = @(
    '*split-auto     | SB BS | --layer-split auto |',
    '*split-12       | SB BS | --layer-split 12 |',
    'split-auto-own0 | BS | --layer-split auto | STRATA_SPLIT_OWN_BUFFERS=0',
    'split-12-own0   | BS | --layer-split 12 | STRATA_SPLIT_OWN_BUFFERS=0',
    'split-auto-pw1  | BS | --layer-split auto --pipeline-windows 1 |',
    'split-12-pw1    | BS | --layer-split 12 --pipeline-windows 1 |',
    'split-auto-res  | BS | --layer-split auto --resident-experts |',
    'split-12-res    | BS | --layer-split 12 --resident-experts |',
    '*peer           | BS SB | --peer-device 1 |',
    'peer-direct0    | BS | --peer-device 1 | STRATA_PEER_DIRECT=0',
    'peer-nop2p      | BS | --peer-device 1 | STRATA_PEER_NO_P2P=1',
    '*helper         | BS SB | --expert-cache-device1 auto |',
    'helper-ropt     | BS | --expert-cache-device1 auto --remote-expert-opt |',
    'helper-early    | BS | --expert-cache-device1 auto | STRATA_EARLY_REMOTE_CONTEXTS=1',
    'helper-source   | BS | --expert-cache-device1 auto | STRATA_PREFILL_HELPER_SOURCE=1',
    '*elastic        | B | --vram-elastic |',
    '*kv-grow        | B | --kv-grow |',
    '*pin            | B | --resident-experts --resident-budget-gib 30 | STRATA_PARTIAL_PIN=1')

function Main {
    $Root = (Resolve-Path "$PSScriptRoot\..\..").Path
    $Py = "$Root\.venv\Scripts\python.exe"
    $After = (git -C $Root rev-parse --short HEAD)
    if (-not $Results) { $Results = "$PSScriptRoot\results\$(Get-Date -Format yyyyMMdd-HHmm)" }
    $Steps = "$Results\steps"
    New-Item -ItemType Directory -Force $Steps | Out-Null
    $Runs = if ($Quick) { 1 } else { 2 }
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # cards numbered as nvidia-smi numbers them
    $env:CUDA_CACHE_MAXSIZE = '4294967296'     # room for the code the emulated builds compile when they start
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH" }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VsDir = if (Test-Path $vswhere) {      # CUDA 13.0 takes Visual Studio 2022 (17.x) only
        & $vswhere -latest -products * -version '[17.0,18.0)' -property installationPath `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
    }
    $Cards = Get-Cards
    $Big, $Small = $Cards[0], $Cards[1]
    $phases = [ordered]@{
        preflight = { Test-Preflight }
        build     = { Invoke-Builds '86-real' '86' }
        unit      = { Invoke-Unit '86' }
        smoke     = { Invoke-Smoke '86' }
        multi     = { Invoke-Multi }
        deepseek  = { Invoke-DeepSeek }
        speed     = {
            if (Test-Need speed-q38-server @("$DataDir\models\IQ3_S")) {
                Invoke-Pairs speed-q38-server server strata.exe 86 "--cards $Big" ($Runs + 1) '--tolerance' 5
            }
        }
        emulate   = { Invoke-Emulate }
    }
    foreach ($p in $phases.Keys) {
        if ($Only -contains $p) { & $phases[$p] }
    }
    Invoke-Report summary $Results
    if ($DryRun) { return }
    "`nCommits that changed engine code, to bisect a DIFF ($Before..$After):" | Add-Content "$Results\summary.txt"
    git -C $Root log --format='%h %s' "$Before..$After" -- src include CMakeLists.txt cmake |
        Add-Content "$Results\summary.txt"
    Get-ChildItem $Results -Recurse -Include *.ses, dspack | Remove-Item -Recurse -Force   # big, and of no use later
    Compress-Archive -Force -Path $Results -DestinationPath "$Results.zip"
    Write-Host "`nSend this file back: $Results.zip"
}

# Records the PC (cards, driver, tools, RAM, disk) and says what is missing.
function Test-Preflight {
    $tools = [ordered]@{
        'nvidia-smi' = 'nvidia-smi'; 'nvcc (CUDA 13.0)' = "$env:CUDA_PATH\bin\nvcc.exe"; git = 'git'
        'Python (.venv)' = $Py; cmake = "$Root\.venv\Scripts\cmake.exe"; ninja = "$Root\.venv\Scripts\ninja.exe"
        'Visual Studio 2022 C++' = "$VsDir\VC\Auxiliary\Build\vcvars64.bat"
    }
    $notes = @($tools.Keys | Where-Object { -not (Get-Command $tools[$_] -ErrorAction SilentlyContinue) } |
        ForEach-Object { "missing: $_" })
    $null = Invoke-Step preflight @(
        'nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv',
        'nvidia-smi topo -p2p r',
        "`"$($tools['nvcc (CUDA 13.0)'])`" --version", 'git --version', "`"$Py`" --version",
        "`"$($tools.cmake)`" --version", "echo Visual Studio: $VsDir", "echo Before $Before, after $After")
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $free = [math]::Round((Get-PSDrive $Work.Substring(0, 1)).Free / 1GB)
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    "RAM $ram GB; $cpu; $free GB free for $Work" | Add-Content "$Steps\preflight.log"
    if ($free -lt 40) { $notes += "only $free GB free for $Work (six builds need about 20-30)" }
    if ($ram -lt 46) { $notes += "$ram GB of RAM: Qwen3.8 IQ3_S and DeepSeek were planned for 48" }
    if ($Cards.Count -lt 2) { $notes += 'one card: the multi-GPU checks are skipped' }
    Add-Row preflight $(if ($notes -match '^missing') { 'FAIL' } else { 'PASS' }) (
        ($notes + "cards $($Cards -join ','), the bigger first; RAM $ram GB") -join '; ')
}

# Builds the engines and tests before and after the refactor for one CUDA architecture list, each from its own git
# worktree into $Work\<before|after>-<tag>.
function Invoke-Builds([string]$Arch, [string]$Tag) {
    $cmake = "`"$Root\.venv\Scripts\cmake.exe`""
    foreach ($side in 'before', 'after') {
        $src, $bd = "$Work\$side", "$Work\$side-$Tag"
        $rev = @{ before = $Before; after = $After }[$side]
        $start = Get-Date
        $rc = Invoke-Step "build-$side-$Tag" @(
            "if not exist `"$src\.git`" git -C `"$Root`" worktree add --detach `"$src`" $rev || exit /b 1",
            "call `"$VsDir\VC\Auxiliary\Build\vcvars64.bat`" >nul || exit /b 1",
            ("if not exist `"$bd\CMakeCache.txt`" $cmake -G Ninja -S `"$src`" -B `"$bd`" -DCMAKE_BUILD_TYPE=Release " +
             "`"-DCMAKE_MAKE_PROGRAM=$Root\.venv\Scripts\ninja.exe`" " +
             "`"-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH\bin\nvcc.exe`" " +
             "-DCMAKE_CUDA_ARCHITECTURES=$Arch -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON " +
             "-DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_PORTABLE=ON || exit /b 1"),
            # CUDA 13.0's ptxas now and then fails on a file it just wrote; the same command then gets past it
            "$cmake --build `"$bd`" -j $Jobs || $cmake --build `"$bd`" -j $Jobs")
        $ok = $rc -eq 0 -and (Test-Path "$bd\strata.exe")
        Add-Row "build-$side-$Tag" $(if ($ok) { 'PASS' } else { 'FAIL' }) (
            "$rev, $([int]((Get-Date) - $start).TotalMinutes) min")
    }
}

# Runs ctest in both builds of a tag (with STRATA_EMULATE_CC=$Emulate when given) and compares the per-test results.
function Invoke-Unit([string]$Tag, [string]$Filter = '', [string]$Emulate = '') {
    foreach ($side in 'before', 'after') {
        $null = Invoke-Step "unit-$side-$Tag" @("set CUDA_VISIBLE_DEVICES=$($Cards -join ',')",
            "set STRATA_EMULATE_CC=$Emulate",
            "`"$Root\.venv\Scripts\ctest.exe`" --test-dir `"$Work\$side-$Tag`" --timeout 900 $Filter")
    }
    Invoke-Report ctest $Results "unit-$Tag" "$Steps\unit-before-$Tag.log" "$Steps\unit-after-$Tag.log"
}

# The protocol checks of Qwen3.8 and Qwen3.6 on the bigger card.
function Invoke-Smoke([string]$Tag, [string]$Drive = '') {
    $one = "--cards $Big $Drive"
    if (Test-Need "smoke-q38-$Tag" @("$DataDir\models\IQ3_S", "$DataDir\packs\iq3_s")) {
        Invoke-Pairs "smoke-q38-$Tag" q38 strata.exe $Tag $one
        Invoke-Pairs "smoke-q38state-$Tag" q38state strata.exe $Tag (
            "$one --other `"$Results\smoke-q38state-$Tag\before-1.out.ses`"")
    }
    if (Test-Need "smoke-q36-$Tag" @("$DataDir\models\qwen36-UD-Q4_K_XL")) {
        Invoke-Pairs "smoke-q36-$Tag" q36 strata-q36.exe $Tag $one
    }
}

function Invoke-Multi {
    if ($null -eq $Small) {
        return Add-Row multi SKIP 'one card'
    }
    if (-not (Test-Need multi @("$DataDir\models\IQ3_S", "$DataDir\packs\iq3_s"))) { return }
    foreach ($row in $MultiChecks) {
        $name, $orders, $engineArgs, $settings = $row.TrimStart('*') -split '\|' | ForEach-Object { $_.Trim() }
        $core = $row.StartsWith('*')
        if ($Quick -and -not $core) { continue }
        foreach ($order in @(-split $orders)[0..$(if ($Quick) { 0 } else { 9 })]) {
            $cards = ($order.ToCharArray() | ForEach-Object { if ($_ -eq 'B') { $Big } else { $Small } }) -join ','
            $set = if ($settings) { " --env $settings" } else { '' }
            Invoke-Pairs "multi-$name-$order" multi strata.exe 86 "--cards $cards --args=`"$engineArgs`"$set" (
                $(if ($core) { $Runs } else { 1 }))
        }
    }
}

# The B2 checks: DeepSeek's answers without the expert cache (where they repeat), with and without its drafter, and
# its decode speed with the shipped hot-expert profile.
function Invoke-DeepSeek {
    $ds = "$DataDir\packs\deepseek-v4.1-flash-ist-das-2.00bpw"
    if (-not (Test-Need deepseek @($ds, $DeepSeekDir))) { return }
    $base = "--cards $Big --deepseek `"$DeepSeekDir`""
    $plain = '--expert-profile none --no-expert-cache'
    Invoke-Pairs ds-dspark-auto ds strata-dsv41.exe 86 "$base --args=`"$plain --dspark auto`""
    if (-not $Quick) {
        Invoke-Pairs ds-dspark-none ds strata-dsv41.exe 86 "$base --args=`"$plain --dspark none`""
        Invoke-Pairs ds-dspark-sample ds strata-dsv41.exe 86 (
            "$base --args=`"$plain --dspark auto`" --env DS4_DSPARK_SAMPLE=1")
    }
    Invoke-Pairs ds-speed ds strata-dsv41.exe 86 "$base --args=`"--dspark auto`"" ($Runs + 1) '--tolerance' 5
}

# Older cards' code paths on these cards: builds of PTX for sm_75 and sm_80 (compiled by the driver when they start)
# with STRATA_EMULATE_CC, so that the host picks the paths that card would.
function Invoke-Emulate {
    foreach ($cc in @(75, 80)[0..$(if ($Quick) { 0 } else { 1 })]) {
        Invoke-Builds "$cc-virtual" "${cc}v"
        Invoke-Unit "${cc}v" '-R "parity|prefill_|qsa|sampler|verify"' $cc
        Invoke-Smoke "${cc}v" "--env STRATA_EMULATE_CC=$cc"
    }
}

# Runs one drive.py scenario with the engines of both builds, alternating, $Count times each, then judges the check.
# $Drive holds drive.py's arguments after --exe, --out and --data. Runs that finished earlier are kept.
function Invoke-Pairs([string]$Check, [string]$Scenario, [string]$Exe, [string]$Tag, [string]$Drive,
                      [int]$Count = $Runs) {
    $dir = "$Results\$Check"
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($r in 1..$Count) {
        foreach ($side in 'before', 'after') {
            $out = "$dir\$side-$r.out"
            if ((Test-Path $out) -and (Get-Content $out -Tail 1) -eq 'exit 0') { continue }
            $null = Invoke-Step "$Check-$side-$r" ("`"$Py`" `"$PSScriptRoot\drive.py`" $Scenario " +
                "--exe `"$Work\$side-$Tag\$Exe`" --out `"$out`" --data `"$DataDir`" $Drive")
        }
    }
    Invoke-Report runs $Results $Check @args
}

# Says whether a check's files are there; if not, records the check as skipped (-DryRun goes on anyway).
function Test-Need([string]$Check, [string[]]$Paths) {
    $missing = @($Paths | Where-Object { -not $_ -or -not (Test-Path $_) })
    if ($missing) { Add-Row $Check SKIP "missing: $($missing -join ', ')" }
    return $DryRun -or -not $missing
}

# Gets the cards' nvidia-smi numbers, the one with the most memory first; with no nvidia-smi (a dry run elsewhere),
# the plan's PC: an RTX 3060 as card 0 and the bigger 3080 as card 1.
function Get-Cards {
    $rows = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits
    }
    if (-not $rows) { return @(1, 0) }
    return @($rows | Sort-Object { [int]($_ -split ',')[1] } -Descending | ForEach-Object { [int]($_ -split ',')[0] })
}

# Writes a step's command lines to steps\<name>.cmd and runs them, their output in steps\<name>.log; with -DryRun
# only prints them. Returns the exit code.
function Invoke-Step([string]$Name, [string[]]$Lines) {
    $file = "$Steps\$Name.cmd"
    $text = @('@echo off', 'chcp 65001 >nul', "call :main >> `"$Steps\$Name.log`" 2>&1", 'exit /b %errorlevel%',
              ':main')
    [IO.File]::WriteAllLines($file, $text + $Lines)   # UTF-8 without the byte order mark cmd would take for a command
    Write-Host "[$(Get-Date -Format HH:mm)] $Name"
    if ($DryRun) {
        $Lines | ForEach-Object { Write-Host "    $_" }
        return 0
    }
    & cmd /c $file
    return $LASTEXITCODE
}

function Add-Row([string]$Check, [string]$Status, [string]$Note) {
    Invoke-Report row $Results $Check $Status $Note
}

# Runs report.py with these arguments (prints them with -DryRun).
function Invoke-Report {
    if ($DryRun) {
        return Write-Host "    report.py $args"
    }
    & $Py "$PSScriptRoot\report.py" @args | Write-Host
}

Main
