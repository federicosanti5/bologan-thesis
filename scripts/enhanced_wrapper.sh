#!/usr/bin/env bash

# Enhanced wrapper for BoloGAN monitoring - Thesis Federico Tosoni
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

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

# Function to start system monitoring
start_monitoring() {
    local task=$1
    local sys_prefix="${EXPERIMENT_DIR}/system_monitoring/${task}_system"
    local gpu_prefix="${EXPERIMENT_DIR}/gpu_monitoring/${task}_gpu"
    
    echo "🔍 Starting monitoring for task: ${task}"
    echo "📊 Monitoring data will be saved to: ${EXPERIMENT_DIR}"

    # Detailed CPU usage
    if command -v vmstat &> /dev/null; then
        vmstat 1 > "${sys_prefix}_vmstat.log" 2>&1 &
        VMSTAT_PID=$!
    else
        echo "⚠️  vmstat not available - CPU monitoring limited"
        VMSTAT_PID=""
    fi
    
    # I/O monitoring  
    if command -v iostat &> /dev/null; then
        iostat -x 1 > "${sys_prefix}_iostat.log" 2>&1 &
        IOSTAT_PID=$!
    else
        echo "⚠️  iostat not available - I/O monitoring disabled"
        IOSTAT_PID=""
    fi

    # Memory usage detailed
    free -h -s 1 > "${sys_prefix}_memory.log" 2>&1 &
    MEMORY_PID=$!
    
    # CPU and Memory monitoring
    if command -v htop &> /dev/null; then
        htop -d 1 > "${sys_prefix}_htop.log" 2>&1 &
        HTOP_PID=$!
    elif command -v top &> /dev/null; then
        top -b -d 1 > "${sys_prefix}_top.log" 2>&1 &
        HTOP_PID=$!
        echo "ℹ️  Using 'top' instead of 'htop'"
    else
        echo "⚠️  Neither htop nor top available"
        HTOP_PID=""
    fi
    
    # Energy monitoring (if available)
    if command -v turbostat &> /dev/null; then
        # RAPL energy counters (requires sudo or proper permissions)
        turbostat --show PkgWatt,CorWatt,GFXWatt,PkgTmp --interval 1 > "${sys_prefix}_energy.log" 2>&1 &
        ENERGY_PID=$!
    else
        echo "⚠️  turbostat not available - energy monitoring disabled"
        ENERGY_PID=""
    fi
    
    # GPU monitoring (if NVIDIA GPU available)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv -l 1 > "${gpu_prefix}.csv" 2>&1 &
        GPU_PID=$!
    else
        echo "⚠️  nvidia-smi not available - GPU monitoring disabled"
        GPU_PID=""
    fi
    
    # Process-specific monitoring
    # We'll get the PID of the Python process later
    
    # Save monitoring PIDs for cleanup
    echo "${HTOP_PID} ${VMSTAT_PID} ${IOSTAT_PID} ${MEMORY_PID} ${ENERGY_PID} ${GPU_PID}" > "${EXPERIMENT_DIR}/monitoring_pids.txt"
}

# Function to stop monitoring
stop_monitoring() {
    echo "🛑 Stopping monitoring..."
    
    if [ -f "${EXPERIMENT_DIR}/monitoring_pids.txt" ]; then
        read -r HTOP_PID VMSTAT_PID IOSTAT_PID MEMORY_PID ENERGY_PID GPU_PID < "${EXPERIMENT_DIR}/monitoring_pids.txt"
        
        # Kill monitoring processes
        for pid in $HTOP_PID $VMSTAT_PID $IOSTAT_PID $MEMORY_PID $ENERGY_PID $GPU_PID; do
            if [ ! -z "$pid" ] && [ "$pid" != "" ]; then
                kill $pid 2>/dev/null || true
            fi
        done
        
        rm "${EXPERIMENT_DIR}/monitoring_pids.txt"
    fi
    
    echo "✅ Monitoring stopped"
}

# Function to monitor specific process

#!/usr/bin/env bash

# Enhanced wrapper for BoloGAN monitoring - Thesis Federico Tosoni
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

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

# Function to start system monitoring
start_monitoring() {
    local task=$1
    local sys_prefix="${EXPERIMENT_DIR}/system_monitoring/${task}_system"
    local gpu_prefix="${EXPERIMENT_DIR}/gpu_monitoring/${task}_gpu"
    
    echo "🔍 Starting monitoring for task: ${task}"
    echo "📊 Monitoring data will be saved to: ${EXPERIMENT_DIR}"
    
    # CPU and Memory monitoring (check if commands exist)
    if command -v vmstat &> /dev/null; then
        vmstat 1 > "${sys_prefix}_vmstat.log" 2>&1 &
        VMSTAT_PID=$!
    else
        echo "⚠️  vmstat not available - CPU monitoring limited"
        VMSTAT_PID=""
    fi
    
    if command -v iostat &> /dev/null; then
        iostat -x 1 > "${sys_prefix}_iostat.log" 2>&1 &
        IOSTAT_PID=$!
    else
        echo "⚠️  iostat not available - I/O monitoring disabled"
        IOSTAT_PID=""
    fi
    
    # Memory usage detailed
    free -h -s 1 > "${sys_prefix}_memory.log" 2>&1 &
    MEMORY_PID=$!
    
    # htop alternative - use top if htop not available
    if command -v htop &> /dev/null; then
        htop -d 1 > "${sys_prefix}_htop.log" 2>&1 &
        HTOP_PID=$!
    elif command -v top &> /dev/null; then
        top -b -d 1 > "${sys_prefix}_top.log" 2>&1 &
        HTOP_PID=$!
        echo "ℹ️  Using 'top' instead of 'htop'"
    else
        echo "⚠️  Neither htop nor top available"
        HTOP_PID=""
    fi
    
    # Energy monitoring (if available)
    if command -v turbostat &> /dev/null; then
        turbostat --show PkgWatt,CorWatt,GFXWatt,PkgTmp --interval 1 > "${sys_prefix}_energy.log" 2>&1 &
        ENERGY_PID=$!
    else
        echo "⚠️  turbostat not available - energy monitoring disabled"
        ENERGY_PID=""
    fi
    
    # GPU monitoring (if NVIDIA GPU available)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv -l 1 > "${gpu_prefix}.csv" 2>&1 &
        GPU_PID=$!
    else
        echo "⚠️  nvidia-smi not available - GPU monitoring disabled"
        GPU_PID=""
    fi
    
    # Save monitoring PIDs for cleanup
    echo "${HTOP_PID} ${VMSTAT_PID} ${IOSTAT_PID} ${MEMORY_PID} ${ENERGY_PID} ${GPU_PID}" > "${EXPERIMENT_DIR}/monitoring_pids.txt"
}

# Function to stop monitoring
stop_monitoring() {
    echo "🛑 Stopping monitoring..."
    
    if [ -f "${EXPERIMENT_DIR}/monitoring_pids.txt" ]; then
        read -r HTOP_PID VMSTAT_PID IOSTAT_PID MEMORY_PID ENERGY_PID GPU_PID < "${EXPERIMENT_DIR}/monitoring_pids.txt"
        
        # Kill monitoring processes
        for pid in $HTOP_PID $VMSTAT_PID $IOSTAT_PID $MEMORY_PID $ENERGY_PID $GPU_PID; do
            if [ ! -z "$pid" ] && [ "$pid" != "" ]; then
                kill $pid 2>/dev/null || true
            fi
        done
        
        rm "${EXPERIMENT_DIR}/monitoring_pids.txt"
    fi
    
    echo "✅ Monitoring stopped"
}

# Function to monitor specific process
monitor_process() {
    local process_name=$1
    local output_prefix="${EXPERIMENT_DIR}/process_monitoring/${process_name}_process"
    local consecutive_missing=0

    echo "Starting process monitoring for: $process_name"
    
    # Find Python processes related to training
    while true; do

        # Check monitoring flag first
        if [ ! -f "${EXPERIMENT_DIR}/monitoring_active" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Monitoring flag removed, stopping $process_name monitor" >> "${output_prefix}_ps.log" 2>/dev/null || true
            break
        fi

        # Look for python3 processes running train.py or evaluate.py
        python_pids=$(pgrep -f "python3.*${process_name}" 2>/dev/null || true)
        
        if [ ! -z "$python_pids" ]; then
            for pid in $python_pids; do
                # CPU and memory usage for specific process
                ps -p $pid -o pid,ppid,cmd,%mem,%cpu,rss,vsz,etime --no-headers >> "${output_prefix}_ps.log" 2>/dev/null || true
                
                # Process I/O (if available)
                if [ -f "/proc/$pid/io" ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') $(cat /proc/$pid/io 2>/dev/null | tr '\n' ' ')" >> "${output_prefix}_io.log" 2>/dev/null || true
                fi
            done
        else
            # Process not found, increment counter
            consecutive_missing=$((consecutive_missing + 1))
            
            # If process missing for 10 seconds, assume it's finished
            if [ $consecutive_missing -ge 10 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') Process $process_name no longer running for 10s, stopping monitor" >> "${output_prefix}_ps.log" 2>/dev/null || true
                break
            fi
        fi
        
        # Safe sleep with error handling
        sleep 1 2>/dev/null || {
            echo "$(date '+%Y-%m-%d %H:%M:%S') Sleep interrupted, stopping $process_name monitor" >> "${output_prefix}_ps.log" 2>/dev/null || true
            break
        }
    done
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') Process monitoring for $process_name completed" >> "${output_prefix}_ps.log" 2>/dev/null || true
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
    echo "🎯 EXECUTION SUMMARY"
    echo "===================="
    echo "Task: ${task}"
    echo "Duration: ${hours}h ${minutes}m ${seconds}s"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Results saved to: ${EXPERIMENT_DIR}"
    echo ""
}

# Cleanup function
# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    
    # Remove monitoring flag FIRST to signal all processes to stop
    rm -f "${EXPERIMENT_DIR}/monitoring_active" 2>/dev/null || true
    
    # Give processes time to see the flag and exit gracefully
    sleep 2
    
    # Now stop system monitoring
    stop_monitoring
    
    # Force stop any remaining process monitors
    for pid_file in "${EXPERIMENT_DIR}"/process_monitor_*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null || true)
            if [ ! -z "$pid" ] && [ "$pid" != "" ]; then
                echo "Stopping process monitor PID: $pid"
                kill $pid 2>/dev/null || true
                sleep 1
                kill -9 $pid 2>/dev/null || true  # Force kill if still running
            fi
            rm "$pid_file" 2>/dev/null || true
        fi
    done
    
    # Kill any remaining monitoring processes by pattern
    pkill -f "vmstat.*${EXPERIMENT_ID}" 2>/dev/null || true
    pkill -f "iostat.*${EXPERIMENT_ID}" 2>/dev/null || true
    pkill -f "free.*${EXPERIMENT_ID}" 2>/dev/null || true
    
    echo "✅ Cleanup completed"
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

# Start monitoring
start_monitoring "$task"

# Create monitoring active flag
touch "${EXPERIMENT_DIR}/monitoring_active"

# Start process-specific monitoring
if [[ ${task} == *'train'* ]]; then
    monitor_process "train.py" &
else
    monitor_process "evaluate.py" &
fi

echo ""
echo "🏃 Starting execution..."
echo "⏱️  Start time: $(date)"
echo ""

# Execute the command and capture output
exec > >(tee "${EXPERIMENT_DIR}/logs/${task}_stdout.log") 2> >(tee "${EXPERIMENT_DIR}/logs/${task}_stderr.log" >&2)

# Execute the actual command
eval $command
command_exit_code=$?

echo ""
echo "✅ Execution completed"
echo "⏱️ End time: $(date)"
echo "🎯 Exit code: $command_exit_code"

# Remove monitoring active flag
rm -f "${EXPERIMENT_DIR}/monitoring_active"

# Calculate and save summary
end_timestamp=$(date +%s)
calculate_summary "$task" "$start_timestamp" "$end_timestamp"

# Cleanup will be called automatically by trap

cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds

exit $command_exit_code