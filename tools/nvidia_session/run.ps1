<#
.SYNOPSIS
The code-sharing refactor's phase B test session on an NVIDIA PC: builds Strata's engines as they were before the
refactor and as they are now, runs both the same way, and says where they differ. Read README.md beside this file.
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -Followup   # only the checks the first session left open, 4 runs a build
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -Only multi -Results .\results\20261101-0900   # one phase again, same folder
.EXAMPLE
.\run.ps1 -DataDir E:\Strata-data -Only preflight, build, memory -Before 4fa1d1cf   # the memory manager's checks
#>
param(
    [string]$DataDir,                # a copy of the owner's Strata-data folder with the files README.md lists
    [string]$DeepSeekDir,            # the folder of the DeepSeek V4.1 GGUF files
    [ValidateSet('preflight', 'build', 'unit', 'smoke', 'multi', 'deepseek', 'speed', 'emulate', 'followup',
                 'memory')]
    [string[]]$Only = @('preflight', 'build', 'unit', 'smoke', 'multi', 'deepseek', 'speed', 'emulate'),
    [string]$Before = '87d20a5c',    # the last commit before any C++ moved
    [string]$Work = "$env:SystemDrive\sn",   # source copies and builds; short, because CUDA builds hit MAX_PATH
    [string]$Results,                # a results folder to continue in (finished runs are kept)
    [int]$Jobs = [Environment]::ProcessorCount,
    [switch]$Quick,                  # fewer runs and fewer variants: about half the time
    [switch]$Followup,               # only the checks the first session left open (the followup phase)
    [int]$Runs,                      # runs a build of each check (default: 2, 1 with -Quick, 4 with -Followup)
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
#: drive.py's settings that make the engine say where a decode's time went, in a follow-up session's multi-GPU and
#: speed runs (both builds): free VRAM after each startup step, each stage's host time per window, and each request's
#: time per window. Each prints a few lines a request, none a token. report.py summarizes them per build.
$Telemetry = ' --env STRATA_TRACE=1 --env STRATA_SPLIT_TIMING=1 --env STRATA_DECODE_TIMING=1'
#: strata-q38's memory settings in the memory phase: explicit, so that its pools leave room under Windows' commit
#: limit (its RAM pool is page-locked, and by default takes what RAM has free)
$Q38Memory = '--ram-mib 20480'

function Main {
    $Root = (Resolve-Path "$PSScriptRoot\..\..").Path
    # a trailing backslash would escape the closing quote of "...\" in the steps' command lines
    $DataDir, $DeepSeekDir = @($DataDir, $DeepSeekDir) | ForEach-Object { $_ -replace '[\\/]+$', '\.' }
    $Py = "$Root\.venv\Scripts\python.exe"
    $After = (git -c "safe.directory=$Root" -C $Root rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or -not $After) {
        throw "Cannot identify the kit checkout at $Root"
    }
    if (-not $Results) { $Results = "$PSScriptRoot\results\$(Get-Date -Format yyyyMMdd-HHmm)" }
    $Steps = "$Results\steps"
    New-Item -ItemType Directory -Force $Steps | Out-Null
    if (-not $Runs) { $Runs = if ($Followup) { 4 } elseif ($Quick) { 1 } else { 2 } }
    if ($Followup -and -not $PSBoundParameters.ContainsKey('Only')) { $Only = 'preflight', 'build', 'unit', 'followup' }
    $Blocked = @{}
    $Prepared = @{}                            # the worktrees Initialize-Worktree has seen in this session
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # cards numbered as nvidia-smi numbers them
    $env:CUDA_CACHE_MAXSIZE = '4294967296'     # room for the code the emulated builds compile when they start
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH" }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VsDir = if (Test-Path $vswhere) {      # CUDA 13 takes Visual Studio 2022 (17.x) only
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
                Invoke-Pairs speed-q38-server server strata.exe 86 "--cards $Big$(if ($Followup) { $Telemetry })" (
                    $Runs + 1)
            }
        }
        emulate   = { Invoke-Emulate }
        # The checks the first session (20261030-1419) left open: answers that changed in one run of the build
        # after (helper), slower decodes (split-*-res measured once, peer, speed), and a save/restore check whose build
        # before also varied (q38state on the sm_75 build)
        followup  = {
            Invoke-Multi 'helper', 'peer', 'split-auto-res', 'split-12-res'
            & $phases.speed
            Invoke-Builds '75-virtual' '75v'
            Invoke-Smoke '75v' '--env STRATA_EMULATE_CC=75' 'q38state'
        }
        memory    = { Invoke-Memory }
    }
    foreach ($p in $phases.Keys) {
        if ($Only -contains $p) { & $phases[$p] }
    }
    Invoke-Report summary $Results
    if ($DryRun) { return }
    "`nCommits that changed engine code, to bisect a DIFF ($Before..$After):" | Add-Content "$Results\summary.txt"
    git -c "safe.directory=$Root" -C $Root log --format='%h %s' "$Before..$After" -- src include CMakeLists.txt cmake |
        Add-Content "$Results\summary.txt"
    Get-ChildItem $Results -Recurse -Include *.ses, dspack | Remove-Item -Recurse -Force   # big, and of no use later
    Compress-Archive -Force -Path $Results -DestinationPath "$Results.zip"
    Write-Host "`nSend this file back: $Results.zip"
}

# Records the PC (Windows, cards, driver, PCIe links, tools, RAM, page file, disks) and says what is missing.
function Test-Preflight {
    $tools = [ordered]@{
        'nvidia-smi' = 'nvidia-smi'; 'nvcc (CUDA 13)' = "$env:CUDA_PATH\bin\nvcc.exe"; git = 'git'
        'Python (.venv)' = $Py; cmake = "$Root\.venv\Scripts\cmake.exe"; ninja = "$Root\.venv\Scripts\ninja.exe"
        'Visual Studio 2022 C++' = "$VsDir\VC\Auxiliary\Build\vcvars64.bat"
    }
    $notes = @($tools.Keys | Where-Object { -not (Get-Command $tools[$_] -ErrorAction SilentlyContinue) } |
        ForEach-Object { "missing: $_" })
    $null = Invoke-Step preflight @(
        'nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv',
        ('nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,' +
         'pcie.link.width.current --format=csv'),
        'nvidia-smi topo -p2p r',
        "`"$($tools['nvcc (CUDA 13)'])`" --version", "git -c `"safe.directory=$Root`" --version", "`"$Py`" --version",
        "`"$($tools.cmake)`" --version", "echo Visual Studio: $VsDir", "echo Before $Before, after $After",
        "echo Kit commit: $After")
    if ($DryRun) { return }
    $os = Get-CimInstance Win32_OperatingSystem
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    $pageFiles = @(Get-CimInstance Win32_PageFileUsage | ForEach-Object { "$($_.Name) $($_.AllocatedBaseSize) MB" })
    $disks = @($Work, $DataDir, $DeepSeekDir | Where-Object { $_ } | ForEach-Object { $_.Substring(0, 1) } |
        Sort-Object -Unique | ForEach-Object { "$_`: $([math]::Round((Get-PSDrive $_).Free / 1GB)) GB free" })
    $free = [math]::Round((Get-PSDrive $Work.Substring(0, 1)).Free / 1GB)
    @("$($os.Caption) $($os.Version); RAM $ram GB; $cpu",
      "page file: $(if ($pageFiles) { $pageFiles -join ', ' } else { 'none' })", "disks: $($disks -join ', ')") |
        Add-Content "$Steps\preflight.log"
    if ($free -lt 40) { $notes += "only $free GB free for $Work (six builds need about 20-30)" }
    if ($ram -lt 46) { $notes += "$ram GB of RAM: Qwen3.8 IQ3_S and DeepSeek were planned for 48" }
    if ($Cards.Count -lt 2) { $notes += 'one card: the multi-GPU checks are skipped' }
    Add-Row preflight $(if ($notes -match '^missing') { 'FAIL' } else { 'PASS' }) (
        ($notes + "kit commit $After" + "cards $($Cards -join ','), the bigger first; RAM $ram GB") -join '; ')
}

# Builds the engines and tests before and after the refactor for one CUDA architecture list, each from its own git
# worktree into $Work\<before|after>-<tag>. A target that fails does not stop the others (ninja -k 0).
function Invoke-Builds([string]$Arch, [string]$Tag) {
    $cmake = "`"$Root\.venv\Scripts\cmake.exe`""
    foreach ($side in 'before', 'after') {
        $src, $bd = "$Work\$side", "$Work\$side-$Tag"
        $rev = @{ before = $Before; after = $After }[$side]
        $start = Get-Date
        $Blocked["$side-$Tag"] = $true
        if (-not (Initialize-Worktree $side $src $rev)) {
            Add-Row "build-$side-$Tag" FAIL "worktree preparation failed: see steps\worktree-$side.log"
            continue
        }
        $actual = git -c "safe.directory=$src" -C $src rev-parse HEAD 2>$null
        $actualCode = $LASTEXITCODE
        $wanted = git -c "safe.directory=$Root" -C $Root rev-parse "$rev^{commit}" 2>$null
        if (-not $DryRun -and ($actualCode -ne 0 -or $LASTEXITCODE -ne 0 -or -not $actual -or
                              -not $wanted -or $actual -ne $wanted)) {
            Add-Row "build-$side-$Tag" FAIL "cannot verify $src at $rev; use a fresh -Work folder"
            continue
        }
        if (-not $DryRun) {
            Remove-Item "$bd\kit-build.json" -ErrorAction SilentlyContinue
        }
        $build ="$cmake --build `"$bd`" -j $Jobs -- -k 0"
        $buildCode = Invoke-Step "build-$side-$Tag" @(
            "call `"$VsDir\VC\Auxiliary\Build\vcvars64.bat`" >nul || exit /b 1",
            ("if not exist `"$bd\CMakeCache.txt`" $cmake -G Ninja -S `"$src`" -B `"$bd`" -DCMAKE_BUILD_TYPE=Release " +
             "`"-DCMAKE_MAKE_PROGRAM=$Root\.venv\Scripts\ninja.exe`" " +
             "`"-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH\bin\nvcc.exe`" " +
             "-DCMAKE_CUDA_ARCHITECTURES=$Arch -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON " +
             "-DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_PORTABLE=ON || exit /b 1"),
            # CUDA 13.0's ptxas now and then fails on a file it just wrote; the same command then gets past it.
            # report.py judges the last attempt, after the line echoed here.
            "$build || (echo retrying the build & $build)")
        if ($buildCode -ne 0) {
            Add-Row "build-$side-$Tag" FAIL "build failed at source HEAD $actual; see steps\build-$side-$Tag.log"
            continue
        }
        if (-not $DryRun) {
            $hashes = @{}
            foreach ($exe in Get-ChildItem "$bd\strata*.exe" -ErrorAction SilentlyContinue) {
                $hashes[$exe.Name] = (Get-FileHash $exe.FullName -Algorithm SHA256).Hash
            }
            @{ revision = $actual; hashes = $hashes } | ConvertTo-Json | Set-Content "$bd\kit-build.json"
            Copy-Item "$bd\kit-build.json" "$Results\build-$side-$Tag.json"
        }
        $Blocked["$side-$Tag"] = $false
        Invoke-Report build $Results "build-$side-$Tag" "$Steps\build-$side-$Tag.log" $bd (
            "source HEAD $actual; binary SHA256: build-$side-$Tag.json; " +
            "$([int]((Get-Date) - $start).TotalMinutes) min")
    }
}

# Checks out a build's worktree once a session (at $Rev, dropping the edits of an earlier session), applies each
# file's section of portability.patch where it applies (the before tree has the files at their old paths), and gives
# it llama.cpp's gguf-py for the iq_parity fixtures: the main checkout's copy, else setup's download.
function Initialize-Worktree([string]$Side, [string]$Src, [string]$Rev) {
    if ($Prepared.ContainsKey($Side)) {
        return $Prepared[$Side]
    }
    $log = "$Steps\worktree-$Side.log"
    $code = Invoke-Step "worktree-$Side" @(
        ("if not exist `"$Src\.git`" git -c `"safe.directory=$Root`" -C `"$Root`" " +
         "worktree add --detach `"$Src`" $Rev || exit /b 1"),
        "git -c `"safe.directory=$Src`" -C `"$Src`" checkout --force --detach $Rev || exit /b 1")
    if ($code -ne 0) {
        $Prepared[$Side] = $false
        return $false
    }
    $Prepared[$Side] = $true
    if ($DryRun) {
        Write-Host "    portability.patch and gguf-py into $Src"
        return $true
    }
    # The calls Visual Studio rejects; a file without them was made portable another way (strata::platform::set_env).
    $Rejected = '\b(setenv|unsetenv|__builtin_popcount)\('
    $patchText = Get-Content -Raw "$PSScriptRoot\portability.patch"
    foreach ($section in ($patchText -split '(?m)^(?=diff --git )' | Select-Object -Skip 1)) {
        $file = [regex]::Match($section, '^diff --git a/(\S+)').Groups[1].Value
        $patch = New-TemporaryFile
        [IO.File]::WriteAllText($patch, $section)
        $result = if (-not (Test-Path "$Src\$file")) { 'not at this path' } else {
            git -c "safe.directory=$Src" -C $Src apply --ignore-whitespace $patch 2>$null
            if ($LASTEXITCODE -eq 0) { 'patched' } else {
                git -c "safe.directory=$Src" -C $Src apply --reverse --check --ignore-whitespace $patch 2>$null
                if ($LASTEXITCODE -eq 0) { 'fixed already' }
                elseif (-not (Select-String -Quiet -Pattern $Rejected -Path "$Src\$file")) { 'not needed' }
                else { 'PATCH FAILED' }
            }
        }
        Remove-Item $patch
        "portability.patch, $file`: $result" | Add-Content $log
        if ($result -eq 'PATCH FAILED') {
            $Prepared[$Side] = $false
            return $false
        }
    }
    $gguf = "$Src\third_party\llama.cpp\gguf-py"
    if (Test-Path "$gguf\gguf") { return $true }
    if (Test-Path "$Root\third_party\llama.cpp\gguf-py\gguf") {
        New-Item -ItemType Directory -Force (Split-Path $gguf) | Out-Null
        Copy-Item -Recurse "$Root\third_party\llama.cpp\gguf-py" $gguf
    } else {
        & $Py -c "import sys; sys.path.insert(0, r'$Src'); import setup; setup.get_llama_cpp()" *>> $log
    }
    "gguf-py: $(if (Test-Path "$gguf\gguf") { 'in place' } else { 'MISSING (iq_parity will not run)' })" |
        Add-Content $log
    return $true
}

# Runs ctest in both builds of a tag (with STRATA_EMULATE_CC=$Emulate when given) and compares the per-test results.
function Invoke-Unit([string]$Tag, [string]$Filter = '', [string]$Emulate = '') {
    foreach ($side in 'before', 'after') {
        if (-not (Test-BuildReady $side $Tag)) {
            return Add-Row "unit-$Tag" FAIL "$side-$Tag build is unverified or failed; run build first"
        }
    }
    foreach ($side in 'before', 'after') {
        $null = Invoke-Step "unit-$side-$Tag" @("set CUDA_VISIBLE_DEVICES=$($Cards -join ',')",
            "set STRATA_EMULATE_CC=$Emulate",
            "`"$Root\.venv\Scripts\ctest.exe`" --test-dir `"$Work\$side-$Tag`" --timeout 900 $Filter")
    }
    Invoke-Report ctest $Results "unit-$Tag" "$Steps\unit-before-$Tag.log" "$Steps\unit-after-$Tag.log"
}

# The protocol checks of Qwen3.8 and Qwen3.6 on the bigger card (the $Scenarios of drive.py named).
function Invoke-Smoke([string]$Tag, [string]$Drive = '', [string[]]$Scenarios = @('q38', 'q38state', 'q36')) {
    $one = "--cards $Big $Drive"
    if (Test-Need "smoke-q38-$Tag" @("$DataDir\models\IQ3_S", "$DataDir\packs\iq3_s")) {
        if ($Scenarios -contains 'q38') {
            Invoke-Pairs "smoke-q38-$Tag" q38 strata.exe $Tag $one
        }
        if ($Scenarios -contains 'q38state') {
            Invoke-Pairs "smoke-q38state-$Tag" q38state strata.exe $Tag (
                "$one --other `"$Results\smoke-q38state-$Tag\before-1.out.ses`"")
        }
    }
    if ($Scenarios -contains 'q36' -and (Test-Need "smoke-q36-$Tag" @("$DataDir\models\qwen36-UD-Q4_K_XL"))) {
        Invoke-Pairs "smoke-q36-$Tag" q36 strata-q36.exe $Tag $one
    }
}

# Runs the rows of $MultiChecks (those named in $Names, else all), each card order of a row: $Runs times a build for
# the rows marked '*' and in a follow-up session, once for the others.
function Invoke-Multi([string[]]$Names) {
    if ($null -eq $Small) {
        return Add-Row multi SKIP 'one card'
    }
    if (-not (Test-Need multi @("$DataDir\models\IQ3_S", "$DataDir\packs\iq3_s"))) { return }
    foreach ($row in $MultiChecks) {
        $name, $orders, $engineArgs, $settings = $row.TrimStart('*') -split '\|' | ForEach-Object { $_.Trim() }
        $core = $row.StartsWith('*')
        if (($Names -and $Names -notcontains $name) -or ($Quick -and -not $core)) { continue }
        foreach ($order in @(-split $orders)[0..$(if ($Quick) { 0 } else { 9 })]) {
            $cards = Get-CardList $order
            $set = $(if ($settings) { " --env $settings" }) + $(if ($Followup) { $Telemetry })
            Invoke-Pairs "multi-$name-$order" multi strata.exe 86 "--cards $cards `"--args=$engineArgs`"$set" (
                $(if ($core -or $Followup) { $Runs } else { 1 }))
        }
    }
}

# The B2 checks: DeepSeek's answers without the expert cache (where they repeat), with and without its drafter, and
# its decode speed with the shipped hot-expert profile.
function Invoke-DeepSeek {
    if (-not $DeepSeekDir -and -not $DryRun) {
        return Add-Row deepseek SKIP 'needs -DeepSeekDir: folder containing the DeepSeek GGUF shards and Engram'
    }
    $ds = "$DataDir\packs\deepseek-v4.1-flash-ist-das-2.00bpw"
    if (-not (Test-Need deepseek @($ds, $DeepSeekDir))) { return }
    $base = "--cards $Big --deepseek `"$DeepSeekDir`""
    $plain = '--expert-profile none --no-expert-cache'
    Invoke-Pairs ds-dspark-auto ds strata-dsv41.exe 86 "$base `"--args=$plain --dspark auto`""
    if (-not $Quick) {
        Invoke-Pairs ds-dspark-none ds strata-dsv41.exe 86 "$base `"--args=$plain --dspark none`""
        Invoke-Pairs ds-dspark-sample ds strata-dsv41.exe 86 (
            "$base `"--args=$plain --dspark auto`" --env DS4_DSPARK_SAMPLE=1")
    }
    Invoke-Pairs ds-speed ds strata-dsv41.exe 86 "$base `"--args=--dspark auto`"" ($Runs + 1)
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

# The memory manager's checks (docs/MEMORY_MANAGER_PLAN.md, steps M7 to M9), with -Before at the M7 commit: strata-q38
# on one card against the build before; the old engine's layer split, whose planner moved to shared code, against the
# build before; that split's speed against strata-q38's; strata-q38 on both cards (each order), with the second card
# computing experts of the first card's layers in one hop, and with its cut moved while it decodes, against one card;
# tools\measure_rows.py on the bigger card.
function Invoke-Memory {
    Invoke-DeepSeekMemory
    if (-not (Test-Need memory @("$DataDir\models\IQ3_S", "$DataDir\packs\iq3_s"))) { return }
    $q38 = "`"--args=$Q38Memory`""
    Invoke-Pairs memory-q38-1card multi strata-q38.exe 86 "--cards $Big $q38"
    Invoke-Multi 'split-auto'
    if ($null -eq $Small) {
        Add-Row memory-2cards SKIP 'one card'
    } else {
        $one = "strata-q38.exe --cards $Big $q38"
        # one card against two: the same experts on the GPUs (the profile's first 400, each on the card of its
        # layer: the 3060 as the second card has room for a pool of under 1 GiB), none of a layer's experts read over
        # the bus and no CPU share of a prompt, so that both compute each expert in the same place; and no drafting,
        # which on a split changes tokens at near-ties (docs/MEMORY_MANAGER_PLAN.md, M8 part 4)
        $same = "--expert-cache 400 --pcie-frac 0 --lookahead 0"
        $sameEnv = '--env STRATA_PREFILL_CPU_SHARE=0'
        $oneSame = "strata-q38.exe --cards $Big `"--args=$Q38Memory $same`" $sameEnv"
        $split = "`"--args=$Q38Memory $same --layer-split auto`" $sameEnv"
        Invoke-Versus memory-speed-split-q38 multi "strata.exe --cards $Small,$Big `"--args=--layer-split auto`"" $one
        foreach ($order in 'SB', 'BS') {
            $cards = Get-CardList $order
            Invoke-Versus "memory-q38-2cards-$order" multi $oneSame "strata-q38.exe --cards $cards $split"
            $groupCommon = "--layer-split 24 --batch 4 --expert-cache 400 --pcie-frac 0 --adapt-swaps 0"
            $groupEnv = "$sameEnv --env DS4_COMMIT_CAP_GB=30"
            $pagedGroups = ("--ram-mib 8192 --gpu-mib 4096,4096 $groupCommon " +
                '--lookahead 0 --rows 8 --remote-experts off')
            Invoke-Versus "memory-groups-q38-$order" groups (
                "strata-q38.exe --cards $cards `"--args=$pagedGroups --batch-groups 1`" $groupEnv") (
                "strata-q38.exe --cards $cards `"--args=$pagedGroups --batch-groups auto`" $groupEnv")
            $oldGroups = "$groupCommon --resident-budget-gib 8 --suffix-draft 0"
            Invoke-Versus "memory-groups-old-$order" groups (
                "strata.exe --cards $cards `"--args=$oldGroups --batch-groups 1`" $groupEnv") (
                "strata.exe --cards $cards `"--args=$oldGroups --batch-groups auto`" $groupEnv")
            # the first card's pool too small for its share of those experts: the second card holds the rest and
            # computes them in one hop, so that the GPUs still hold what one card does
            $hop = "`"--args=$Q38Memory $same --layer-split 24 --gpu-mib 512,8192 --remote-experts on`" $sameEnv"
            Invoke-Versus "memory-hop-$order" multi $oneSame "strata-q38.exe --cards $cards $hop"
            $defaultQueues = '--env GPU_MAX_HW_QUEUES=4 --env CUDA_DEVICE_MAX_CONNECTIONS=8'
            Invoke-Versus "memory-hop-default-queues-$order" multi (
                "strata-q38.exe --cards $cards $hop --env GPU_MAX_HW_QUEUES=16 " +
                '--env CUDA_DEVICE_MAX_CONNECTIONS=32') ("strata-q38.exe --cards $cards $hop $defaultQueues")
            # pools with room to spare (12 GiB on the bigger card, 5 GiB on the smaller), so that with the first
            # card's halved the cut moves and each of the profile's first 3000 experts still fits on a card
            $mib = ($order.ToCharArray() | ForEach-Object { if ($_ -eq 'B') { 12288 } else { 5120 } }) -join ','
            $move = "`"--args=$Q38Memory --expert-cache 3000 --pcie-frac 0 --gpu-mib $mib --layer-split auto`""
            Invoke-Versus "memory-cut-move-$order" cutmove "strata-q38.exe --cards $cards $move" (
                "strata-q38.exe --cards $cards $move --move-cut")
        }
    }
    # the kit's own run config (the model's arguments from -DataDir's files, the drafter included), not a
    # strata-iq3_s.json in the checkout: one copied from another PC names paths this PC may not have
    $exe = "$Work\after-86\strata-q38.exe"
    $config = "$Results\rows-config.json"
    $configCode = Invoke-Step memory-rows-config ("`"$Py`" `"$PSScriptRoot\drive.py`" config --out `"$config`" " +
        "--data `"$DataDir`" --exe `"$exe`"")
    foreach ($case in @(@('card', ''), @('ram', '--gpu-mib 8192'), @('ssd', '--gpu-mib 8192 --ram-mib 4096'))) {
        $name = "memory-rows-$($case[0])"
        if ($configCode -ne 0 -or -not (Test-BuildReady after 86 strata-q38.exe)) {
            Add-Row $name FAIL 'row config failed or after-86 build is unverified; see steps'
            continue
        }
        $code = Invoke-Step $name ("`"$Py`" `"$Root\tools\measure_rows.py`" --config `"$config`" --exe `"$exe`" " +
            "--case $($case[0]) --out `"$Results\rows-$($case[0])`" -- --gpu $Big $Q38Memory $($case[1])")
        if ($code -eq 0) {
            Add-Row $name PASS "measured: rows-$($case[0]).json"
        } else {
            Add-Row $name FAIL "measure_rows.py stopped (exit $code): see steps\$name.log"
        }
    }
    Invoke-Report rows $Results memory-rows-report "$Results\rows-card.json" "$Results\rows-ram.json" (
        "$Results\rows-ssd.json")
}

# D9's checks use only the build after: DeepSeek's paged engine did not exist at M7. Small, fixed expert sets leave
# room for both cards to keep their layers' experts as the cut moves, without drafting or adaptive placement.
function Invoke-DeepSeekMemory {
    if (-not $DeepSeekDir -and -not $DryRun) {
        return Add-Row memory-ds41 SKIP 'D9 needs -DeepSeekDir: folder containing the DeepSeek GGUF shards and Engram'
    }
    if ($null -eq $Small) {
        return Add-Row memory-ds41 SKIP 'two NVIDIA cards needed'
    }
    $pack = "$DataDir\packs\deepseek-v4.1-flash-ist-das-2.00bpw"
    if (-not (Test-Need memory-ds41 @($pack, $DeepSeekDir))) { return }
    $base = "--deepseek `"$DeepSeekDir`" --env DS4_COMMIT_CAP_GB=30"
    $fixed = '--dspark none --no-expert-cache --expert-cache 240 --ram-mib 8192 --remote-experts off'
    $one = "strata-ds41.exe --cards $Big $base `"--args=$fixed --gpus 0 --gpu-mib 4096`""
    foreach ($order in 'SB', 'BS') {
        $cards = Get-CardList $order
        $split = "strata-ds41.exe --cards $cards $base " +
                 "`"--args=$fixed --gpus 0,1 --gpu-mib 2048,2048 --layer-split 20`""
        Invoke-Versus "memory-ds41-2cards-$order" dsmemory $one $split
        foreach ($draft in 'on', 'off') {
            foreach ($count in 1..4) {
                $groupExperts = if ($draft -eq 'on') { 2000 } else { 240 }
                $groupArgs = "--gpus 0,1 --layer-split 20 --gpu-mib 4096,7680 --ram-mib 8192 " +
                             "--expert-cache $groupExperts --no-expert-cache --remote-experts off --batch 4 " +
                             "--concurrent $count --dspark $(if ($draft -eq 'off') { 'none' } else { 'on' })"
                $groupRun = "strata-ds41.exe --cards $cards $base --conversations $count"
                Invoke-Versus "memory-groups-ds41-$($count)c-dspark-$draft-$order" dsgroups (
                    "$groupRun `"--args=$groupArgs --batch-groups 1`"") (
                    "$groupRun `"--args=$groupArgs --batch-groups auto`"")
            }
        }
        $cutRun = $split.Replace('--layer-split 20', '--layer-split 10')
        Invoke-Versus "memory-ds41-cut-move-$order" dscutmove $cutRun "$cutRun --move-cut"
        foreach ($mode in 'off', 'auto', 'on') {
            $modeArgs = $fixed.Replace('--remote-experts off', "--remote-experts $mode")
            $modeRun = "strata-ds41.exe --cards $cards $base " +
                       "`"--args=$modeArgs --gpus 0,1 --gpu-mib 256,4096 --layer-split 20`""
            Invoke-Versus "memory-ds41-remote-$mode-$order" dsmemory $one $modeRun
        }
        $hopArgs = $fixed.Replace('--remote-experts off', '--remote-experts on')
        $hop = "strata-ds41.exe --cards $cards $base " +
               "`"--args=$hopArgs --gpus 0,1 --gpu-mib 256,4096 --layer-split 20`""
        Invoke-Versus "memory-ds41-hop-default-queues-$order" dsmemory (
            "$hop --env GPU_MAX_HW_QUEUES=16 --env CUDA_DEVICE_MAX_CONNECTIONS=32") (
            "$hop --env GPU_MAX_HW_QUEUES=4 --env CUDA_DEVICE_MAX_CONNECTIONS=8")
    }
}

# Runs one drive.py scenario with two engines of the build after, alternating, $Count times each, and judges the
# check as Invoke-Pairs does: $Run0 as "before" and $Run1 as "after", each the engine's file name and its drive.py
# arguments after --data.
function Invoke-Versus([string]$Check, [string]$Scenario, [string]$Run0, [string]$Run1, [int]$Count = $Runs) {
    $args0, $args1 = $Run0, $Run1 | ForEach-Object {
        $exe, $drive = $_ -split ' ', 2
        "--exe `"$Work\after-86\$exe`" $drive"
    }
    Invoke-Sides $Check $Scenario $args0 $args1 $Count
}

# Runs one drive.py scenario with the engines of both builds, alternating, $Count times each, then judges the check.
# $Drive holds drive.py's arguments after --exe, --out and --data.
function Invoke-Pairs([string]$Check, [string]$Scenario, [string]$Exe, [string]$Tag, [string]$Drive,
                      [int]$Count = $Runs) {
    Invoke-Sides $Check $Scenario "--exe `"$Work\before-$Tag\$Exe`" $Drive" "--exe `"$Work\after-$Tag\$Exe`" $Drive" (
        $Count)
}

# Runs drive.py's $Scenario with $BeforeArgs and with $AfterArgs (its arguments besides --out and --data), alternating,
# $Count times each, then judges the check. Runs that finished earlier are kept. Skipped when strata-q38 refuses what
# the check asks for as not built yet.
function Invoke-Sides([string]$Check, [string]$Scenario, [string]$BeforeArgs, [string]$AfterArgs, [int]$Count) {
    foreach ($drive in $BeforeArgs, $AfterArgs) {
        $exePath = [regex]::Match($drive, '--exe "([^"]+)"').Groups[1].Value
        $folder = Split-Path (Split-Path $exePath) -Leaf
        $side, $tag = $folder -split '-', 2
        if (-not (Test-BuildReady $side $tag (Split-Path $exePath -Leaf))) {
            return Add-Row $Check FAIL "$folder build is unverified or failed; run build first"
        }
    }
    $dir = "$Results\$Check"
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($r in 1..$Count) {
        foreach ($side in 'before', 'after') {
            $out = "$dir\$side-$r.out"
            if ((Test-Path $out) -and (Get-Content $out -Tail 1) -eq 'exit 0') { continue }
            $drive = $(if ($side -eq 'before') { $BeforeArgs } else { $AfterArgs })
            $null = Invoke-Step "$Check-$side-$r" ("`"$Py`" `"$PSScriptRoot\drive.py`" $Scenario " +
                "--out `"$out`" --data `"$DataDir`" $drive")
            if (-not $DryRun -and (Select-String -Quiet 'is not there yet' "$out.stderr" -ErrorAction Ignore)) {
                return Add-Row $Check SKIP 'strata-q38 refuses it as not built yet (see its stderr)'
            }
        }
    }
    Invoke-Report runs $Results $Check
}

# Verifies that a dependent check uses an unchanged executable from an identified source revision.
function Test-BuildReady([string]$Side, [string]$Tag, [string]$Exe = 'strata.exe') {
    if ($DryRun) {
        return $true
    }
    if ($Blocked["$Side-$Tag"]) {
        return $false
    }
    $bd = "$Work\$Side-$Tag"
    try {
        $stamp = Get-Content -Raw "$bd\kit-build.json" -ErrorAction Stop | ConvertFrom-Json
        $rev = @{ before = $Before; after = $After }[$Side]
        $wanted = git -c "safe.directory=$Root" -C $Root rev-parse "$rev^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $wanted -or $stamp.revision -ne $wanted) {
            return $false
        }
        return $stamp.hashes.$Exe -eq (Get-FileHash "$bd\$Exe" -Algorithm SHA256 -ErrorAction Stop).Hash
    } catch {
        return $false
    }
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
    if ($DryRun) {
        return @(1, 0)
    }
    $rows = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits
    }
    if (-not $rows) { return @(1, 0) }
    return @($rows | Sort-Object { [int]($_ -split ',')[1] } -Descending | ForEach-Object { [int]($_ -split ',')[0] })
}

# Gets drive.py's --cards for a card order such as 'SB' (B: the bigger card, S: the smaller), for example "0,1".
function Get-CardList([string]$Order) {
    return ($Order.ToCharArray() | ForEach-Object { if ($_ -eq 'B') { $Big } else { $Small } }) -join ','
}

# Writes a step's command lines to steps\<name>.cmd and runs them, their output in steps\<name>.log (a new one each
# time); with -DryRun only prints them. Returns the exit code.
function Invoke-Step([string]$Name, [string[]]$Lines) {
    $file = "$Steps\$Name.cmd"
    $text = @('@echo off', 'chcp 65001 >nul', "call :main > `"$Steps\$Name.log`" 2>&1", 'exit /b %errorlevel%',
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
