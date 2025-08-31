#!/usr/bin/env bash

# Enhanced wrapper for BoloGAN monitoring - Thesis Federico Santi
# Based on original wrapper.sh with added monitoring capabilities

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directories
BOLOGAN_HOME="/home/saint/Documents/UNIBO/tesi"
FASTCALO_DIR="${BOLOGAN_HOME}/FastCaloChallenge"
RESULTS_DIR="${BOLOGAN_HOME}/results"
MONITORING_DIR="${RESULTS_DIR}/monitoring"

# Create monitoring directories
mkdir -p "${MONITORING_DIR}"

# Experiment metadata
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="${MONITORING_DIR}/${EXPERIMENT_ID}"
mkdir -p "${EXPERIMENT_DIR}"/{metadata,system_monitoring,process_monitoring,gpu_monitoring,logs,analysis}

# Sampling rates
ENERGY_SAMPLE_SECS=${ENERGY_SAMPLE_SECS:-0.5}
SYSTEM_SAMPLE_SECS=${SYSTEM_SAMPLE_SECS:-1.0}
THERMAL_SAMPLE_SECS=${THERMAL_SAMPLE_SECS:-0.5}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to add PID to monitoring list
add_monitoring_pid() {
    local pid=$1
    local name=$2
    
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        echo "$pid" >> "${EXPERIMENT_DIR}/monitoring_pids.txt"
        echo "Started $name monitoring (PID: $pid)"
    fi
}

# Function to test if turbostat actually works
test_turbostat() {
    if ! command -v turbostat &>/dev/null; then
        return 1  # Command not found
    fi
    
    echo "Testing turbostat capabilities..." >&2
    # Test if turbostat can actually read energy values
    local test_output
    test_output=$(timeout 3 turbostat --show PkgWatt --interval 0.1 --num_iterations 1 2>&1)
    local exit_code=$?
    
    # Check for common permission errors
    if [[ $exit_code -ne 0 ]] || [[ "$test_output" == *"Permission denied"* ]] || [[ "$test_output" == *"Operation not permitted"* ]] || [[ "$test_output" == *"No such file or directory"* ]]; then
        echo "turbostat test failed: $test_output" >&2
        return 1
    fi
    
    return 0
}

# Function to test if perf can read power events
test_perf_power() {
    if ! command -v perf &>/dev/null; then
        return 1  # Command not found
    fi
    
    echo "Testing perf power capabilities..." >&2
    
    # First check if power events are listed
    local power_events
    power_events=$(perf list 2>/dev/null | grep -E "power/energy-(pkg|ram)/" | head -1)
    
    if [[ -z "$power_events" ]]; then
        echo "perf: no power events listed in 'perf list'" >&2
        return 1
    fi
    
    echo "perf: found power events in list, testing actual access..." >&2
    
    # Test actual access to power events with a quick measurement
    local test_output
    local exit_code
    
    # Run perf with explicit timeout and capture all output
    test_output=$(timeout 3 perf stat -a -e power/energy-pkg/ sleep 0.5 2>&1)
    exit_code=$?
    
    echo "perf test exit code: $exit_code" >&2
    if [[ ${#test_output} -lt 200 ]]; then
        echo "perf test output: $test_output" >&2
    else
        echo "perf test output (truncated): ${test_output:0:200}..." >&2
    fi
    
    # Check for various failure conditions
    if [[ $exit_code -ne 0 ]]; then
        echo "perf power test failed with exit code: $exit_code" >&2
        return 1
    fi
    
    # Parse output for failure indicators
    if [[ "$test_output" == *"Permission denied"* ]]; then
        echo "perf: permission denied accessing power events" >&2
        return 1
    fi
    
    # Check for the specific "<not supported>"
    if [[ "$test_output" == *"<not supported>"* ]]; then
        echo "perf: power events explicitly marked as <not supported>" >&2
        return 1
    fi
    
    # Additional check: look for actual numeric values (Joules)
    if [[ "$test_output" =~ [0-9]+(\.[0-9]+)?[[:space:]]+Joules ]]; then
        echo "perf: successfully measured energy values" >&2
        return 0
    fi
    
    # If we can't find actual measurements, it's probably not working
    echo "perf: no valid energy measurements found in output" >&2
    echo "perf: this usually means the events are listed but not accessible" >&2
    return 1
}

# Function to test RAPL access
test_rapl_access() {
    if [[ ! -d /sys/class/powercap ]]; then
        return 1  # Directory doesn't exist
    fi
    
    echo "Testing RAPL accessibility..." >&2
    # Check if any intel-rapl domains exist and are readable
    local found_readable=0
    for rapl_path in /sys/class/powercap/intel-rapl:*; do
        if [[ -d "$rapl_path" && -f "$rapl_path/name" && -f "$rapl_path/energy_uj" ]]; then
            # Test if we can actually read the energy value
            if cat "$rapl_path/energy_uj" >/dev/null 2>&1; then
                found_readable=1
                break
            fi
        fi
    done
    
    if [[ $found_readable -eq 0 ]]; then
        echo "RAPL: no readable energy domains found" >&2
        return 1
    fi
    
    return 0
}

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

# Function to start system monitoring
start_monitoring() {
    local task=$1
    local sys_prefix="${EXPERIMENT_DIR}/system_monitoring/${task}_system"
    local gpu_prefix="${EXPERIMENT_DIR}/gpu_monitoring/${task}_gpu"
    
    echo "Starting monitoring for task: ${task}"
    echo "Monitoring data will be saved to: ${EXPERIMENT_DIR}"
    
    # CPU and Memory monitoring (check if commands exist)
    if command -v vmstat &> /dev/null; then
        vmstat 1 > "${sys_prefix}_vmstat.log" 2>&1 &
        add_monitoring_pid $! "vmstat"
    else
        echo "vmstat not available - CPU monitoring limited"
    fi
    
    if command -v iostat &> /dev/null; then
        iostat -x $SYSTEM_SAMPLE_SECS > "${sys_prefix}_iostat.log" 2>&1 &
        add_monitoring_pid $! "iostat"
    else
        echo "iostat not available - I/O monitoring disabled"
    fi
    
    # Memory usage detailed
    free -h -s $SYSTEM_SAMPLE_SECS > "${sys_prefix}_memory.log" 2>&1 &
    add_monitoring_pid $! "memory"
    
    # Top monitoring
    if command -v top &> /dev/null; then
        {
            while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
                echo "### $(date '+%F %T')"
                top -b -n 1 -w 512 | head -n 40
                sleep $SYSTEM_SAMPLE_SECS
            done
        } > "${sys_prefix}_top.log" 2>&1 &
        add_monitoring_pid $! "top"
    else
        echo "Top not available"
    fi

    # CPU Frequency monitoring
    if [[ -f /proc/cpuinfo ]]; then
        {
            cpu_count=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "0")
            
            if [[ $cpu_count -gt 0 ]]; then
                echo "# Discovered $cpu_count CPU cores for frequency monitoring" >&2
                
                header="timestamp_ms"
                for ((i=0; i<cpu_count; i++)); do
                    header="${header},cpu${i}_mhz"
                done
                echo "$header"
                
                # Monitoring loop
                while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
                    timestamp_ms=$(date +%s%3N)
                    freq_data=$(grep "cpu MHz" /proc/cpuinfo 2>/dev/null | awk '{print $4}' | tr '\n' ',')
                    
                    if [[ -n "$freq_data" ]]; then
                        freq_data=${freq_data%,}
                        echo "${timestamp_ms},${freq_data}"
                    else
                        # Fallback
                        row="$timestamp_ms"
                        for ((i=0; i<cpu_count; i++)); do
                            row="${row},-1"
                        done
                        echo "$row"
                    fi
                    
                    sleep $THERMAL_SAMPLE_SECS  # 2Hz sampling
                done
            else
                echo "Unable to determine CPU count from /proc/cpuinfo" >&2
                echo "timestamp_ms,error"
                echo "$(date +%s%3N),no_cpu_data"
            fi

        } > "${sys_prefix}_cpu_freq.csv" 2> "${sys_prefix}_cpu_freq.log" &
        
        add_monitoring_pid $! "cpu_freq"
        echo "CPU frequency monitoring started for $cpu_count cores (2Hz sampling)" >&2
    else
        echo "CPU frequency monitoring not available"
    fi

    # Thermal monitoring
    if [[ -d /sys/class/thermal ]]; then
        {
            # Discovery delle thermal zones disponibili
            thermal_zones=()
            for thermal_path in /sys/class/thermal/thermal_zone*; do
                if [[ -d "$thermal_path" && -f "$thermal_path/temp" ]]; then
                    zone_name=$(basename "$thermal_path")
                    thermal_zones+=("$zone_name")
                    echo "# Discovered thermal zone: $zone_name ($thermal_path)" >&2
                fi
            done
            
            if [[ ${#thermal_zones[@]} -gt 0 ]]; then
                echo "# Found ${#thermal_zones[@]} thermal zones for monitoring" >&2

                header="timestamp_ms"
                for zone in "${thermal_zones[@]}"; do
                    header="${header},${zone}_milliC"
                done
                echo "$header"
                
                # Monitoring loop
                while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
                    timestamp_ms=$(date +%s%3N)
                    row="$timestamp_ms"
                    
                    # Leggi temperatura da ogni thermal zone
                    for zone in "${thermal_zones[@]}"; do
                        thermal_path="/sys/class/thermal/$zone"
                        if [[ -f "$thermal_path/temp" ]]; then
                            temp_millic=$(cat "$thermal_path/temp" 2>/dev/null || echo "-1")
                        else
                            temp_millic="-1"
                        fi
                        row="${row},${temp_millic}"
                    done
                    
                    echo "$row"
                    sleep $THERMAL_SAMPLE_SECS  # 2Hz sampling
                done
            else
                echo "No thermal zones found in /sys/class/thermal/" >&2
                echo "timestamp_ms,error"
                echo "$(date +%s%3N),no_thermal_data"
            fi
            
        } > "${sys_prefix}_thermal.csv" 2> "${sys_prefix}_thermal.log" &
        
        add_monitoring_pid $! "thermal"
        echo "Thermal monitoring started for ${#thermal_zones[@]} zones (2Hz sampling)" >&2
    else
        echo "Thermal monitoring not available"
    fi
    
    # Energy monitoring (if available)
    # Try turbostat with actual capability test
    if test_turbostat; then
        echo "turbostat: working, starting monitoring"
        turbostat --show PkgWatt,CorWatt,GFXWatt,PkgTmp --interval $ENERGY_SAMPLE_SECS > "${sys_prefix}_energy_turbostat.log" 2>&1 &
        add_monitoring_pid $! "turbostat"
    # Try perf with actual capability test
    elif test_perf_power; then
        echo " power events: working, starting monitoring"
        perf stat -a -I $(($ENERGY_SAMPLE_SECS * 1000)) -e power/energy-pkg/ -e power/energy-ram/ > "${sys_prefix}_energy_perf.log" 2>&1 &
        add_monitoring_pid $! "perf_power"
    elif test_rapl_access; then
        echo "RAPL: readable, starting monitoring"
        {
            declare -A rapl_domains
            declare -a rapl_order
            
            # Scansiona tutti i domini intel-rapl
            for rapl_path in /sys/class/powercap/intel-rapl:*; do
                if [[ -d "$rapl_path" && -f "$rapl_path/name" && -f "$rapl_path/energy_uj" ]]; then
                    if cat "$rapl_path/energy_uj" >/dev/null 2>&1; then
                        domain_name=$(cat "$rapl_path/name" 2>/dev/null || echo "unknown")
                        domain_id=$(basename "$rapl_path")
                        rapl_domains["$domain_id"]="$domain_name"
                        rapl_order+=("$domain_id")
                        echo "# Discovered readable RAPL domain: $domain_id -> $domain_name ($rapl_path)" >&2
                    fi
                fi
            done

            if [[ ${#rapl_order[@]} -eq 0 ]]; then
                echo "# No readable RAPL domains found after testing" >&2
                echo "timestamp_ms,error"
                echo "$(date +%s%3N),no_readable_rapl_domains"
            else
                header="timestamp_ms"
                for domain_id in "${rapl_order[@]}"; do
                    domain_name="${rapl_domains[$domain_id]}"
                    clean_name=$(echo "$domain_name" | tr ' -' '_' | tr -cd '[:alnum:]_')
                    header="${header},${domain_id}_${clean_name}_uj"
                done
                echo "$header"
                
                # Monitoring loop
                while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
                    timestamp_ms=$(date +%s%3N)
                    row="$timestamp_ms"
                    
                    # Leggi tutti i domini nell'ordine scoperto
                    for domain_id in "${rapl_order[@]}"; do
                        rapl_path="/sys/class/powercap/$domain_id"
                        if [[ -f "$rapl_path/energy_uj" ]]; then
                            energy_uj=$(cat "$rapl_path/energy_uj" 2>/dev/null || echo "-1")
                        else
                            energy_uj="-1"
                        fi
                        row="${row},${energy_uj}"
                    done
                    
                    echo "$row"
                    sleep $ENERGY_SAMPLE_SECS  # 2Hz per maggiore granularità energetica
                done
            fi
        } > "${sys_prefix}_energy_rapl.csv" 2> "${sys_prefix}_energy_rapl.log" &

        add_monitoring_pid $! "rapl"
        echo "RAPL monitoring started with ${#rapl_order[@]} readable domains"
    else
        echo "No energy monitoring source available (turbostat/perf/rapl)."
    fi
        
    # GPU monitoring (if NVIDIA GPU available)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv -l $ENERGY_SAMPLE_SECS > "${gpu_prefix}.csv" 2>&1 &
        add_monitoring_pid $! "nvidia-smi"
    else
        echo "nvidia-smi not available - GPU monitoring disabled"
    fi
    
}

# Function to monitor specific process
start_process_monitoring() {
    local process_name=$1
    local output_prefix="${EXPERIMENT_DIR}/process_monitoring/${process_name}_process"
    local perf_started=false

    echo "Starting process monitoring for: $process_name"
    
    # Process monitoring with unified flag-based shutdown
    {
        local consecutive_missing=0
        echo "$(date '+%Y-%m-%d %H:%M:%S') Process monitor started for $process_name" >> "${output_prefix}_ps.log"
        
        while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
            # Look for python3 processes running train.py or evaluate.py
            python_pids=$(pgrep -f "python3.*${process_name}" 2>/dev/null || true)
            
            if [[ -n "$python_pids" ]]; then
                consecutive_missing=0  # Reset counter
                
                for pid in $python_pids; do
                    # CPU and memory usage for specific process
                    ps -p $pid -o pid,ppid,cmd,%mem,%cpu,rss,vsz,etime --no-headers >> "${output_prefix}_ps.log" 2>/dev/null || true
                    
                    # Process I/O (if available)
                    if [[ -f "/proc/$pid/io" ]]; then
                        echo "$(date '+%Y-%m-%d %H:%M:%S') $(cat /proc/$pid/io 2>/dev/null | tr '\n' ' ')" >> "${output_prefix}_io.log" 2>/dev/null || true
                    fi
                done

                # Start perf monitoring for this process (only once)
                if [[ "$perf_started" == false ]] && command -v perf &> /dev/null; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Starting perf monitoring for PID $pid" >> "${output_prefix}_perf.log" 2>/dev/null || true
                    
                    # Start perf stat in background - no field separator, readable format
                    perf stat -p $pid -I $(($SYSTEM_SAMPLE_SECS * 1000)) -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,page-faults,context-switches,cpu-migrations \
                        > "${output_prefix}_perf_raw.log" 2>&1 &
                    
                    # Add perf PID to monitoring file
                    add_monitoring_pid $! "perf_process"
                    perf_started=true
                     
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Perf monitoring started for target process $pid" >> "${output_prefix}_perf.log" 2>/dev/null || true
                fi    
            else
                # Process not found, increment counter
                consecutive_missing=$((consecutive_missing + 1))
                
                # If process missing for 10 seconds, assume it's finished
                if [[ $consecutive_missing -ge 10 ]]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Process $process_name no longer running for 10s, stopping monitor" >> "${output_prefix}_ps.log" 2>/dev/null || true
                    break
                fi
            fi
            
            sleep $SYSTEM_SAMPLE_SECS
        done
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') Process monitoring for $process_name completed" >> "${output_prefix}_ps.log" 2>/dev/null || true
    } &
    
    add_monitoring_pid $! "process_monitor"
}

# Unified function to stop all monitoring processes
stop_monitoring() {
    echo "Stopping all monitoring processes..."

    # Remove flag for cooperative shutdown
    local flag="${EXPERIMENT_DIR}/monitoring_active"
    rm -f "$flag" 2>/dev/null || true

    # Brief wait for cooperative shutdown
    sleep 2

    # Collect all PIDs
    local all_pids=()
    
    # All monitoring PIDs
    local pidfile="${EXPERIMENT_DIR}/monitoring_pids.txt"

    if [[ -f "$pidfile" ]]; then
        while read -r pid; do
            [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && all_pids+=("$pid")
        done < "$pidfile"
    fi

    # Termination logic
    if [[ ${#all_pids[@]} -gt 0 ]]; then
        echo "Terminating ${#all_pids[@]} monitoring processes..."
        
        # TERM signal to all processes
        for pid in "${all_pids[@]}"; do
                kill -TERM "$pid" 2>/dev/null || true
        done

        # Wait for cooperative shutdown
        local timeout=5
        local waited=0
        local any_alive=0
        while [[ $waited -lt $timeout ]]; do
            any_alive=0
            for pid in "${all_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    any_alive=1
                    break
                fi
            done
            [[ $any_alive -eq 0 ]] && break
            sleep 1
            waited=$((waited + 1))
        done

        # Force kill remaining processes
        if [[ $any_alive -eq 1 ]]; then
            for pid in "${all_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing stubborn process: $pid"
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            done
        fi
    fi

    # 5) Cleanup PID files
    rm -f "${EXPERIMENT_DIR}/monitoring_pids.txt" 2>/dev/null || true

    echo "All monitoring stopped"
}


# Function to save experiment metadata
save_experiment_metadata() {
    local task=$1
    local input=$2
    local config_string=$3
    local start_time=$4
    
    cat > "${EXPERIMENT_DIR}/metadata/experiment_metadata.json" << EOF
{
    "experiment_id": "${EXPERIMENT_ID}",
    "task": "${task}",
    "input_file": "${input}",
    "config_string": "${config_string}",
    "config_details": {
        "model": "${model}",
        "config_file": "config/config_${config}.json",
        "mask": "${mask}",
        "prep": "${prep}",
        "label_scheme": "${label_scheme}",
        "split_energy": "${split_energy}"
    },
    "start_time": "${start_time}",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "pwd": "$(pwd)",
    "git_commit": "$(cd ${FASTCALO_DIR} && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(cd ${FASTCALO_DIR} && git rev-parse --abbrev-ref HEAD || echo 'unknown')",
    "container_info": {
        "apptainer_version": "$APPTAINER_VERSION",
        "container_path": "$(echo $APPTAINER_CONTAINER 2>/dev/null || echo 'unknown')"
    },
    "system_info": {
        "kernel": "$(uname -r)",
        "os": "$(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"')",
        "cpu_info": "$(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2 | xargs)",
        "memory_total": "$(free -h | grep Mem | awk '{print $2}')",
        "cpu_cores": "$(nproc)"
    }
}
EOF
}

# Function to calculate execution summary
calculate_summary() {
    local task=$1
    local start_time=$2
    local end_time=$3
    
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    cat > "${EXPERIMENT_DIR}/metadata/execution_summary.json" << EOF
{
    "execution_summary": {
        "task": "${task}",
        "start_timestamp": ${start_time},
        "end_timestamp": ${end_time},
        "duration_seconds": ${duration},
        "duration_formatted": "${hours}h ${minutes}m ${seconds}s",
        "experiment_dir": "${EXPERIMENT_DIR}"
    }
}
EOF

    echo ""
    echo "EXECUTION SUMMARY"
    echo "===================="
    echo "Task: ${task}"
    echo "Duration: ${hours}h ${minutes}m ${seconds}s"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Results saved to: ${EXPERIMENT_DIR}"
    echo ""
}

# Unified cleanup function
cleanup() {
    echo "Cleaning up..."
    stop_monitoring
    echo "Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# =============================================================================
# ORIGINAL WRAPPER.SH LOGIC (MODIFIED)
# =============================================================================

# Change to training directory
cd "${FASTCALO_DIR}/training"

echo "Starting BoloGAN Enhanced Wrapper - Monitoring System"
echo "Arguments: $@"

task=$1
input=$2
config_string=$3    # Configuration string like "BNReLU_hpo27-M1"
loading=$4          # Loading strategies (--prefetch, --cache, etc.)
max_iter=${5:-5000} # Maximum iterations (default: 5000)

# Parse configuration string (original logic with better variable names)
model=`echo $config_string | cut -d '_' -f 1`
config_mask=`echo $config_string | cut -d '_' -f 2-1000`
config_mask=`echo $config_mask | cut -d '.' -f 1`
config=`echo $config_mask | cut -d '-' -f 1`
mask=`echo $config_mask | cut -d '-' -f 2 | cut -d 'M' -f 2`
prep=`echo $config_mask | cut -d '-' -f 3 | cut -d 'P' -f 2`
label_scheme=`echo $config_mask | cut -d '-' -f 4 | cut -d 'L' -f 2`
split_energy=`echo $config_mask | cut -d '-' -f 5 | cut -d 'S' -f 2`

echo "   Configuration:"
echo "   input=$input"
echo "   config_string=$config_string"
echo "   ├── model=$model"
echo "   ├── config=$config (→ config/config_${config}.json)"
echo "   ├── mask=$mask"
echo "   ├── prep=$prep"
echo "   ├── label_scheme=$label_scheme"
echo "   └── split_energy=$split_energy"
echo "   loading=$loading"
echo "   max_iter=$max_iter"

# Version detection (original logic)
if [[ $mask == ?(n)+([0-9]) ]]; then
    version='v2'
    addition="--mask=${mask//n/-}"
else
    version='v1'
    train_addition=""
fi

# Build command arguments (original logic)
if [[ ! -z "$prep" ]]; then
    train_addition="$train_addition -p $prep"
    evaluate_addition="$evaluate_addition -p $prep"
fi

if [[ ! -z "$loading" ]]; then
    train_addition="$train_addition $loading"
    evaluate_addition="$evaluate_addition $loading"
fi

if [[ ! -z "$label_scheme" ]]; then
    train_addition="$train_addition --label_scheme $label_scheme"
fi

if [[ ! -z "$split_energy" ]]; then
    train_addition="$train_addition --split_energy_position $split_energy"
    evaluate_addition="$evaluate_addition --split_energy_position $split_energy"
fi

ds=`echo $input | grep -oP '(?<=input/dataset).'`
if [[ "$ds" = "2" ]]; then
    evaluate_addition="$evaluate_addition --normalise"
fi

# Build final command
if [[ ${task} == *'train'* ]]; then
    command="python3 train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${config_string} -c ../config/config_${config}.json ${train_addition} --max_iter ${max_iter}"
else
    command="python3 evaluate.py -i ${input} -t ../output/dataset${ds}/${version}/${config_string} --checkpoint ${evaluate_addition} --debug --save_h5"
fi

echo "Command to execute: $command"

# =============================================================================
# EXECUTION WITH MONITORING
# =============================================================================

# Save experiment metadata
start_timestamp=$(date +%s)
save_experiment_metadata "$task" "$input" "$config_string" "$(date -d @$start_timestamp)"

# Create monitoring active flag
touch "${EXPERIMENT_DIR}/monitoring_active"

# Start monitoring
start_monitoring "$task"

# Start process-specific monitoring
if [[ ${task} == *'train'* ]]; then
    start_process_monitoring "train.py"
else
    start_process_monitoring "evaluate.py"
fi

echo ""
echo "Starting execution..."
echo "Start time: $(date)"
echo ""

# Execute the command and capture output
exec > >(tee "${EXPERIMENT_DIR}/logs/${task}_stdout.log") 2> >(tee "${EXPERIMENT_DIR}/logs/${task}_stderr.log" >&2)

# Execute the actual command
eval "$command"
command_exit_code=$?

echo ""
echo "Execution completed"
echo "End time: $(date)"
echo "Exit code: $command_exit_code"

# Remove monitoring active flag
rm -f "${EXPERIMENT_DIR}/monitoring_active"

# Calculate and save summary
end_timestamp=$(date +%s)
calculate_summary "$task" "$start_timestamp" "$end_timestamp"

# Cleanup will be called automatically by trap

cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds

exit $command_exit_code