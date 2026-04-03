<#
.SYNOPSIS
    Lattice Planner Batch Data Collection (PowerShell Version)
#>

# Parameters
$WORKERS = 3
$RENDER = $false
$MAP_NAME = "Simple"
$RACELINES = @("raceline0", "raceline1", "raceline2")
$OPP_SPEED_SCALES = @(0.4, 0.5, 0.6)
$INTERVAL_IDX = 10
$SIM_DURATION = 10.0
$NUM_STARTPOINTS = 50

# Calculate total jobs
$total_jobs = $RACELINES.Length * $OPP_SPEED_SCALES.Length * $NUM_STARTPOINTS

Write-Host "Lattice Planner Batch Data Collection"
Write-Host "====================================="
Write-Host "Map: $MAP_NAME"
Write-Host "Racelines (Ego & Opp): $($RACELINES -join ' ')"
Write-Host "Speed scales: $($OPP_SPEED_SCALES -join ' ')"
Write-Host "Interval: $INTERVAL_IDX"
Write-Host "Time per run: ${SIM_DURATION}s"
Write-Host "Starting points: $NUM_STARTPOINTS"
Write-Host "Total jobs: $total_jobs"

# Job ScriptBlock
$jobScript = {
    param($wd, $map, $race, $opp, $oppSpeed, $ego, $interval, $simDur, $rndr)
    Set-Location $wd
    
    $pyArgs = @(
        "demonstration.py",
        "--map_name", $map,
        "--raceline", $race,
        "--opp_raceline", $opp,
        "--opp_speed_scale", [string]$oppSpeed,
        "--ego_idx", [string]$ego,
        "--interval_idx", [string]$interval,
        "--sim_duration", [string]$simDur
    )
    if ($rndr) {
        $pyArgs += "--render"
    }

    # Execute the python script and redirect stdout/stderr to null
    & python $pyArgs *>$null
}

$workingDir = Get-Location

# Generate parameter combinations and run simulations
foreach ($raceline in $RACELINES) {
    # Generate ego_idx_range for this specific raceline length
    $raceline_path = Join-Path "f1tenth_racetracks" (Join-Path $MAP_NAME "$raceline.csv")
    
    if (-not (Test-Path $raceline_path)) {
        Write-Host "Warning: Raceline file not found: $raceline_path" -ForegroundColor Yellow
        continue
    }

    # Count lines skipping the first two (equivalent to tail -n +3)
    $content = Get-Content $raceline_path
    $max_waypoints = ($content | Select-Object -Skip 2).Count
    
    $ego_idx_range = @()
    for ($i = 0; $i -lt $NUM_STARTPOINTS; $i++) {
        $idx = [math]::Truncate($i * ($max_waypoints - 1) / ($NUM_STARTPOINTS - 1))
        $ego_idx_range += $idx
    }

    foreach ($opp_speed in $OPP_SPEED_SCALES) {
        foreach ($ego_idx in $ego_idx_range) {
            
            # Wait for available worker slots
            while ((Get-Job -State Running).Count -ge $WORKERS) {
                Start-Sleep -Milliseconds 100
            }
            
            $jobArgs = @(
                $workingDir.Path,
                $MAP_NAME,
                $raceline,
                $raceline, # opp_raceline is same as raceline in python call
                $opp_speed,
                $ego_idx,
                $INTERVAL_IDX,
                $SIM_DURATION,
                $RENDER
            )

            Start-Job -ScriptBlock $jobScript -ArgumentList $jobArgs | Out-Null
        }
    }
}

# Wait for all jobs to finish
$runningJobs = Get-Job -State Running
if ($runningJobs) {
    Write-Host "`nWaiting for remaining jobs to finish..."
    $runningJobs | Wait-Job | Out-Null
}

# Cleanup jobs
Get-Job | Remove-Job

Write-Host "`nAll simulations completed"

# Find output directories
$output_dirs = @(Get-ChildItem -Directory -Filter "Dataset_${MAP_NAME}_*" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)

Write-Host "Output directories created: $($output_dirs.Count)"

# Print basic statistics for each output directory
foreach ($output_dir in $output_dirs) {
    Write-Host "`nValidating: $($output_dir | Split-Path -Leaf)"
    
    $success_dir = Join-Path $output_dir "success"
    $collision_dir = Join-Path $output_dir "collision"
    
    $success_count = 0
    $collision_count = 0
    $follow_count = 0
    $overtake_count = 0
    
    if (Test-Path $success_dir -PathType Container) {
        $csv_files = @(Get-ChildItem -Path $success_dir -Filter "*_ol*_e*_o*_s*.csv" -File)
        foreach ($csv_file in $csv_files) {
            $success_count++
            $filename = $csv_file.Name
            
            if ($filename.StartsWith("f_")) {
                $follow_count++
            } else {
                $overtake_count++
            }
        }
    }
    
    if (Test-Path $collision_dir -PathType Container) {
        $collision_files = @(Get-ChildItem -Path $collision_dir -Filter "*.json" -File)
        $collision_count = $collision_files.Count
    }
    
    $total_simulations = $success_count + $collision_count
    
    Write-Host "  Total simulations: $total_simulations"
    Write-Host "  Successful: $success_count (Follow: $follow_count, Overtake: $overtake_count)"
    Write-Host "  Collisions: $collision_count"
}
