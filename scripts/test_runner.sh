#!/usr/bin/env bash

# BoloGAN Test Runner - Automation script for thesis experiments
# Author: Federico Santi

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load configuration from file
load_config() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local config_file="${script_dir}/config.sh"
    
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Configuration file not found: $config_file"
        echo "Please create $config_file with the following content:"
        echo "BOLOGAN_HOME=\"/path/to/your/project\""
        exit 1
    fi
    
    # Source the configuration file
    source "$config_file"
    
    # Validate BOLOGAN_HOME is set
    if [[ -z "$BOLOGAN_HOME" ]]; then
        echo "ERROR: BOLOGAN_HOME not defined in $config_file"
        echo "Please add: BOLOGAN_HOME=\"/path/to/your/project\""
        exit 1
    fi
    
    # Validate the path exists
    if [[ ! -d "$BOLOGAN_HOME" ]]; then
        echo "ERROR: BOLOGAN_HOME directory does not exist: $BOLOGAN_HOME"
        echo "Please check the path in $config_file"
        exit 1
    fi
}

# Load configuration
load_config

# Derive other paths from BOLOGAN_HOME
FASTCALO_DIR="${BOLOGAN_HOME}/FastCaloChallenge"
CONTAINER_PATH="${BOLOGAN_HOME}/containers/GANtainer_Mntr.sif"
RESULTS_DIR="${BOLOGAN_HOME}/results"
ENHANCED_WRAPPER="${BOLOGAN_HOME}/scripts/enhanced_wrapper.sh"

CLEANUP_HANDLED=false

# Default parameters
DEFAULT_DATASET="${FASTCALO_DIR}/input/dataset1/dataset_1_pions_1.hdf5"
DEFAULT_CONFIG_STRING="BNReLU_hpo27-M1" # BN = Batch Normalize
                                        # ReLu = Funzione di Attivazione ReLu (Rectified Linear Unit)
                                        # hpo27 = Hyper Parameters Optimization Config #27
                                        # M1 = Mask Variant (diversi preprocessing)

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

log_info() {
    echo "[$(date '+%H:%M:%S')] INFO: $*"
}

log_warn() {
    echo "[$(date '+%H:%M:%S')] WARN: $*" >&2
}

log_error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2
}

log_debug() {
    echo "[$(date '+%H:%M:%S')] DEBUG: $*" >&2
}

log_section() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== $* =========="
}

log_subsection() {
    echo "[$(date '+%H:%M:%S')] --- $* ---"
}

# =============================================================================
# FUNCTIONS
# =============================================================================

show_help() {
    cat << EOF
BoloGAN Test Runner - Automated Experiment Execution

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    setup           Setup environment and check prerequisites
    train           Run training experiment
    evaluate        Run evaluation experiment  
    full            Run both training and evaluation
    list            List available experiments
    analyze         Analyze results from experiment
    clean           Clean old results
    
OPTIONS:
    -d, --dataset PATH      Dataset file path (default: dataset_1_pions_1.hdf5)
    -c, --config CONFIG     Configuration string (default: BNReLU_hpo27-M1)
                            Format: MODEL_CONFIG-MASK[OPTIONAL_FLAGS]
                            Examples: BNReLU_hpo27-M1, LeakyReLU_hpo15-M2-P1-L3
    -m, --max-iter NUM      Maximum iterations for training (default: 5000)
    -l, --loading ARGS      Data loading/preprocessing args (e.g., "--mask 5.0 --debug --add_noise")
                            Valid loading args: --mask NUM, --debug, --add_noise, --loading MODEL_PATH
    -b, --batch-size NUM    Batch size (if supported by config)
    -n, --name NAME         Experiment name prefix
    -h, --help              Show this help message
    --no-container          Run without container (not recommended)
    --cpu-only              Force CPU-only execution
    --gpu                   Try to use GPU if available

CONFIGURATION STRING FORMAT:
    MODEL_CONFIG-MASK[OPTIONAL]
    
    Where:
    - MODEL: Architecture name (e.g., BNReLU, LeakyReLU)
    - CONFIG: Config file name (e.g., hpo27 → config_hpo27.json)
    - MASK: Mask variant (e.g., M1, M2)
    - OPTIONAL: Additional flags like -P1 (prep), -L2 (label_scheme), -S3 (split_energy)

EXAMPLES:
    $0 setup                                    # Check prerequisites
    $0 train                                    # Basic training
    $0 train -m 10000 -n "extended"             # Extended training
    $0 train -l "--mask 5.0 --debug"            # Training with masking and debug
    $0 train -l "--add_noise" -m 8000           # Training with data augmentation
    $0 evaluate                                 # Evaluate last training
    $0 full -c "BNReLU_hpo27-M1"                # Full train+eval cycle
    $0 analyze exp_20241201_143022              # Analyze specific experiment
    $0 list                                     # List all experiments
    
ENVIRONMENT:
    BOLOGAN_HOME    Base directory (default: $BOLOGAN_HOME)
    OMP_NUM_THREADS Number of CPU threads (default: auto-detect)

EOF
}

# Signal handling function for graceful shutdown
cleanup_runner() {
    if [[ "$CLEANUP_HANDLED" == true ]]; then
        log_warn "Cleanup already handled, ignoring additional interrupt"
        return
    fi

    CLEANUP_HANDLED=true

    log_section "INTERRUPT SIGNAL RECEIVED"
    log_info "Initiating graceful shutdown..."

    local latest_exp_dir=""
    if [[ -d "$RESULTS_DIR/monitoring" ]]; then
        latest_exp_dir=$(find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    

    # Try to signal the enhanced wrapper directly
    if [[ -n "$latest_exp_dir" && -f "$latest_exp_dir/wrapper_pid.txt" ]]; then
        local wrapper_pid=$(cat "$latest_exp_dir/wrapper_pid.txt")
        if [[ -n "$wrapper_pid" && "$wrapper_pid" =~ ^[0-9]+$ ]] && kill -0 "$wrapper_pid" 2>/dev/null; then
            log_info "Sending graceful shutdown signal to wrapper (PID: $wrapper_pid)"
            kill -INT "$wrapper_pid" 2>/dev/null || true
            
            # Wait for wrapper cleanup
            local timeout=15
            local waited=0
            local wrapper_alive=true
            
            while [[ $waited -lt $timeout && "$wrapper_alive" == true ]]; do
                if ! kill -0 "$wrapper_pid" 2>/dev/null; then
                    wrapper_alive=false
                    break
                fi
                sleep 1
                waited=$((waited + 1))
            done
            
            if [[ "$wrapper_alive" == true ]]; then
                log_warn "Wrapper didn't respond in time, forcing shutdown"

                # Sequential cleanup before killing wrapper
                if [[ -n "$latest_exp_dir" ]]; then

                    # Removing monitoring flag
                    rm -f "$latest_exp_dir/monitoring_active" 2>/dev/null || true

                    # Kill Python process first
                    if [[ -f "$latest_exp_dir/python_pgid.txt" ]]; then
                        local python_pgid=$(cat "$latest_exp_dir/python_pgid.txt")
                        if [[ -n "$python_pgid" && "$python_pgid" =~ ^[0-9]+$ ]] && kill -0 "$python_pgid" 2>/dev/null; then
                            log_info "Force killing Python process (PGID: $python_pgid)"
                            kill -KILL "-$python_pgid" 2>/dev/null || true
                        fi
                    fi
                    
                    # Kill monitoring processes
                    if [[ -f "$latest_exp_dir/monitoring_pids.txt" ]]; then
                        log_info "Force killing monitoring processes"
                        while read -r pid; do
                            [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
                        done < "$latest_exp_dir/monitoring_pids.txt" 2>/dev/null || true
                    fi
                    
                    # Clean up files
                    rm -f "$latest_exp_dir/python_pgid.txt" 2>/dev/null || true
                    rm -f "$latest_exp_dir/monitoring_pids.txt" 2>/dev/null || true
                    rm -f "$latest_exp_dir/wrapper_pid.txt" 2>/dev/null || true
                fi

                # Force kill wrapper
                kill -KILL "$wrapper_pid" 2>/dev/null || true
            else
                log_info "Wrapper shutdown completed gracefully"
            fi
        fi
    else
        log_warn "No wrapper PID found"
    fi

    # Cleanup container if it exists
    if [ ! -z "$APPTAINER_PID" ]; then
        if kill -0 "$APPTAINER_PID" 2>/dev/null; then
            log_info "Stopping container process (PID: $APPTAINER_PID)"
            kill -TERM "$APPTAINER_PID" 2>/dev/null || true
            sleep 3
            if kill -0 "$APPTAINER_PID" 2>/dev/null; then
                kill -KILL "$APPTAINER_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    log_info "Test runner cleanup completed"
    exit 130
}

# Validate experiment arguments
validate_arguments() {
    local task=$1
    local issues=0
    
    log_subsection "Argument Validation"
    
    # Validate dataset path for experiment tasks
    if [[ "$task" == "train" || "$task" == "evaluate" || "$task" == "full" ]]; then
        if [ ! -f "$DATASET" ]; then
            log_error "Dataset file not found: $DATASET"
            issues=$((issues + 1))
        else
            log_info "Dataset validated: $DATASET"
        fi
    fi
    
    # Validate max_iter is numeric
    if [[ ! "$MAX_ITER" =~ ^[0-9]+$ ]]; then
        log_error "max-iter must be a positive integer: $MAX_ITER"
        issues=$((issues + 1))
    else
        log_info "Max iterations: $MAX_ITER"
    fi
    
    # Validate config string format
    if [[ ! "$CONFIG_STRING" =~ ^[A-Za-z]+_[A-Za-z0-9]+-M[0-9]+.*$ ]]; then
        log_warn "Config string may not be valid format: $CONFIG_STRING"
        log_warn "Expected format: MODEL_CONFIG-MASK (e.g., BNReLU_hpo27-M1)"
    else
        log_info "Configuration validated: $CONFIG_STRING"
    fi
    
    # Validate loading args format if provided
    if [[ -n "$LOADING_ARGS" ]]; then
        log_info "Loading arguments: $LOADING_ARGS"
    fi
    
    return $issues
}

# Check common prerequisites (always needed)
check_common_prerequisites() {
    log_subsection "Common Prerequisites"
    local issues=0
    
    # Check main directory
    if [ ! -d "$FASTCALO_DIR" ]; then
        log_error "FastCaloChallenge directory not found: $FASTCALO_DIR"
        issues=$((issues + 1))
    else
        log_info "FastCaloChallenge directory found"
    fi
    
    # Check enhanced wrapper
    if [ ! -f "$ENHANCED_WRAPPER" ]; then
        log_error "Enhanced wrapper not found: $ENHANCED_WRAPPER"
        log_error "Please save the enhanced wrapper script as: $ENHANCED_WRAPPER"
        issues=$((issues + 1))
    else
        log_info "Enhanced wrapper found"
    fi
    
    # Check basic commands
    if ! command -v python3 &> /dev/null; then
        log_error "python3 command not found"
        issues=$((issues + 1))
    else
        log_info "Python3 available"
    fi
    
    return $issues
}

# Check experiment prerequisites (for train/evaluate tasks)
check_experiment_prerequisites() {
    log_section "PREREQUISITES CHECK"
    
    # First check common prerequisites
    check_common_prerequisites
    local issues=$?
    
    log_subsection "Experiment Prerequisites"
    
    # Check container
    if [ ! -f "$CONTAINER_PATH" ]; then
        log_error "Container not found: $CONTAINER_PATH"
        issues=$((issues + 1))
    else
        log_info "Container found: $CONTAINER_PATH"
    fi
    
    # Check apptainer command
    if ! command -v apptainer &> /dev/null; then
        log_error "apptainer command not found"
        issues=$((issues + 1))
    else
        log_info "Apptainer available: $(apptainer --version)"
    fi
    
    if [ $issues -eq 0 ]; then
        log_info "All prerequisites satisfied"
        return 0
    else
        log_error "Found $issues issues. Please resolve them before proceeding"
        return 1
    fi
}

show_optimization_tips() {
    log_section "OPTIMIZATION TIPS"
    log_info "Optional environment variables for better performance:"
    log_info "  For CPU optimization:"
    log_info "    export OMP_NUM_THREADS=$(nproc)"
    log_info "  For TensorFlow logging:"
    log_info "    export TF_CPP_MIN_LOG_LEVEL=2"
    log_info "  For oneDNN optimizations:"
    log_info "    export TF_ENABLE_ONEDNN_OPTS=1"
    log_info "  Set all at once:"
    log_info "    export OMP_NUM_THREADS=$(nproc) TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1"
}

setup_environment() {
    log_section "ENVIRONMENT SETUP"
    
    # Create directories
    mkdir -p "$RESULTS_DIR"/{monitoring,analysis}
    log_info "Created results directories"
    
    # Check environment variables without modifying them
    log_subsection "Environment Variables"
    log_info "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-'not set (will auto-detect)'}"
    log_info "Available CPU cores: $(nproc)"
    
    if [ -z "$OMP_NUM_THREADS" ]; then
        log_info "Tip: You can set OMP_NUM_THREADS=$(nproc) for optimal CPU usage"
    fi
    
    log_info "TF_CPP_MIN_LOG_LEVEL: ${TF_CPP_MIN_LOG_LEVEL:-'not set (default: 0)'}"
    log_info "TF_ENABLE_ONEDNN_OPTS: ${TF_ENABLE_ONEDNN_OPTS:-'not set (default: auto)'}"
    
    log_info "Environment setup complete"
}

# Show experiment summary after completion
show_experiment_summary() {
    local exp_dir="$1"
    local exit_code="$2"
    
    log_section "EXPERIMENT SUMMARY"
    
    if [ "$exit_code" -eq 0 ]; then
        log_info "Experiment completed successfully"
    else
        log_error "Experiment failed with exit code: $exit_code"
        case $exit_code in
            130) log_info "Cause: Interrupted by user" ;;
            135) log_info "Cause: Bus error - likely container/memory issue" ;;
            *) log_info "Cause: Check logs in $exp_dir/logs/ for details" ;;
        esac
    fi
    
    if [ -d "$exp_dir" ]; then
        log_subsection "Generated Files"
        
        # Count different types of files
        local csv_count=$(find "$exp_dir" -name "*.csv" 2>/dev/null | wc -l)
        local log_count=$(find "$exp_dir" -name "*.log" 2>/dev/null | wc -l)
        local json_count=$(find "$exp_dir" -name "*.json" 2>/dev/null | wc -l)
        
        log_info "Data files generated:"
        log_info "  CSV files: $csv_count"
        log_info "  Log files: $log_count"
        log_info "  JSON metadata: $json_count"
        
        # Check monitoring status
        if [ -d "$exp_dir/system_monitoring" ]; then
            log_info "System monitoring: Data collected"
        fi
        
        if [ -d "$exp_dir/process_monitoring" ]; then
            log_info "Process monitoring: Data collected"
        fi
        
        if [ -d "$exp_dir/gpu_monitoring" ]; then
            log_info "GPU monitoring: Data collected"
        fi
        
        log_info "Full results directory: $exp_dir"
        
        # Show disk usage
        local size=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
        log_info "Total size: ${size:-'unknown'}"
    else
        log_warn "Results directory not found: $exp_dir"
    fi
}

run_experiment() {
    local task=$1
    local dataset=${2:-$DEFAULT_DATASET}
    local config_string=${3:-$DEFAULT_CONFIG_STRING}
    local loading_args=${4:-""}
    local max_iter=${5:-5000}

    log_section "EXPERIMENT EXECUTION"

    # Setting variable about version
    export APPTAINERENV_APPTAINER_VERSION=$(apptainer --version | cut -d' ' -f3)
    
    log_info "Task: $task"
    log_info "Dataset: $dataset"
    log_info "Configuration: $config_string"
    log_info "Loading args: ${loading_args:-'(none)'}"
    log_info "Max iterations: $max_iter"
    
    # Check if we're in a SLURM environment
    if [ ! -z "$SLURM_JOB_ID" ]; then
        log_info "SLURM environment detected"
        log_info "Node: $(hostname)"
        log_info "Job ID: $SLURM_JOB_ID"
    fi
    
    # Enter container and run enhanced wrapper
    if [ "$USE_CONTAINER" = "true" ]; then
        log_info "Using container: $CONTAINER_PATH"
        setsid apptainer exec "$CONTAINER_PATH" bash "$ENHANCED_WRAPPER" "$task" "$dataset" "$config_string" "$loading_args" "$max_iter" &
        APPTAINER_PID=$!

        log_info "Container process started (PID: $APPTAINER_PID)"

        # Wait for completion
        wait $APPTAINER_PID
        local exit_code=$?

        # Clear PID
        unset APPTAINER_PID

        local exp_dir=$(find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

        show_experiment_summary "$exp_dir" "$exit_code"

        return $exit_code
    else
        log_warn "Running without container (not recommended)"
        bash "$ENHANCED_WRAPPER" "$task" "$dataset" "$config_string" "$loading_args" "$max_iter"
        local exit_code=$?

        local exp_dir=$(find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        show_experiment_summary "$exp_dir" "$exit_code"
        return $exit_code
    fi
}

list_experiments() {
    log_section "AVAILABLE EXPERIMENTS"
    
    if [ ! -d "$RESULTS_DIR/monitoring" ]; then
        log_info "No experiments directory found"
        return 0
    fi
    
    local count=0
    local experiments=()
    
    # Collect and sort experiments
    for exp_dir in "$RESULTS_DIR/monitoring"/exp_*; do
        if [ -d "$exp_dir" ]; then
            experiments+=("$(basename "$exp_dir")")
        fi
    done
    
    if [ ${#experiments[@]} -eq 0 ]; then
        log_info "No experiments found"
        return 0
    fi
    
    # Sort experiments (bash built-in sort)
    IFS=$'\n' experiments=($(sort <<<"${experiments[*]}"))
    unset IFS
    
    # Display each experiment
    for exp_id in "${experiments[@]}"; do
        local exp_path="$RESULTS_DIR/monitoring/$exp_id"
        log_info "Experiment: $exp_id"
        
        # Parse metadata with bash (avoiding Python)
        local metadata_file="$exp_path/metadata/experiment_metadata.json"
        local summary_file="$exp_path/metadata/execution_summary.json"
        
        if [ -f "$metadata_file" ]; then
            # Extract task and config with grep/sed
            local task=$(grep '"task"' "$metadata_file" 2>/dev/null | sed 's/.*"task": *"\([^"]*\)".*/\1/' || echo "unknown")
            local config=$(grep '"config_string"' "$metadata_file" 2>/dev/null | sed 's/.*"config_string": *"\([^"]*\)".*/\1/' || echo "unknown")
            local start_time=$(grep '"start_time"' "$metadata_file" 2>/dev/null | sed 's/.*"start_time": *"\([^"]*\)".*/\1/' || echo "unknown")
            
            log_info "  Task: $task"
            log_info "  Config: $config"
            log_info "  Started: $start_time"
        fi
        
        if [ -f "$summary_file" ]; then
            local duration=$(grep '"duration_formatted"' "$summary_file" 2>/dev/null | sed 's/.*"duration_formatted": *"\([^"]*\)".*/\1/' || echo "unknown")
            log_info "  Duration: $duration"
        fi
        
        log_info "  Path: $exp_path"
        echo ""
        count=$((count + 1))
    done
    
    log_info "Total experiments: $count"
}

analyze_experiment() {
    local exp_id=$1
    
    log_section "EXPERIMENT ANALYSIS"
    
    if [ -z "$exp_id" ]; then
        log_error "No experiment ID specified"
        log_info "Usage: $0 analyze <experiment_id>"
        log_info "Use '$0 list' to see available experiments"
        return 1
    fi
    
    local exp_dir="$RESULTS_DIR/monitoring/$exp_id"
    
    if [ ! -d "$exp_dir" ]; then
        log_error "Experiment not found: $exp_dir"
        return 1
    fi
    
    log_info "Analyzing experiment: $exp_id"
    
    # Check if analysis script exists
    local analysis_script="$BOLOGAN_HOME/scripts/analyze_monitoring.py"
    
    if [ ! -f "$analysis_script" ]; then
        log_error "Analysis script not found: $analysis_script"
        log_info "Please save the analysis script as: $analysis_script"
        return 1
    fi
    
    # Run analysis
    log_info "Running analysis script..."
    python3 "$analysis_script" "$exp_dir" --plots --report
    
    log_info "Analysis complete for $exp_id"
}

clean_old_results() {
    local days=${1:-7}
    
    log_section "RESULTS CLEANUP"
    log_info "Cleaning results older than $days days..."
    
    if [ -d "$RESULTS_DIR/monitoring" ]; then
        local count=$(find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -mtime +$days | wc -l)
        if [ "$count" -gt 0 ]; then
            log_info "Found $count experiments to remove"
            find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -mtime +$days -exec rm -rf {} \; 2>/dev/null || true
            log_info "Removed $count old experiments"
        else
            log_info "No old experiments found to remove"
        fi
    else
        log_info "No monitoring directory found"
    fi
    
    log_info "Cleanup complete"
}

# Set up signal trap
trap cleanup_runner INT TERM

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Default values
DATASET="$DEFAULT_DATASET"
CONFIG_STRING="$DEFAULT_CONFIG_STRING"
LOADING_ARGS=""      # Data loading optimizations
MAX_ITER=5000        # Training iterations
BATCH_SIZE=""
EXP_NAME=""
USE_CONTAINER=true
FORCE_CPU=false
USE_GPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_STRING="$2"
            shift 2
            ;;
        -m|--max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        -l|--loading)
            LOADING_ARGS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -n|--name)
            EXP_NAME="$2"
            shift 2
            ;;
        --no-container)
            USE_CONTAINER=false
            shift
            ;;
        --cpu-only)
            FORCE_CPU=true
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        setup|train|evaluate|full|list|analyze|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            if [ -z "$COMMAND" ] && [[ "$1" =~ ^exp_ ]]; then
                # This is likely an experiment ID for analyze command
                EXP_ID="$1"
                shift
            else
                echo "❌ Unknown option: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Check if command was provided
if [ -z "$COMMAND" ]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Execute command
case "$COMMAND" in
    setup)
        check_common_prerequisites || exit 1
        setup_environment
        show_optimization_tips
        ;;
    train)
        check_experiment_prerequisites || exit 1
        validate_arguments "$COMMAND" || exit 1        
        run_experiment "train" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS" "$MAX_ITER"
        ;;
    evaluate)
        check_experiment_prerequisites || exit 1
        validate_arguments "$COMMAND" || exit 1
        run_experiment "evaluate" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS"
        ;;
    full)
        check_experiment_prerequisites || exit 1
        validate_arguments "$COMMAND" || exit 1
        
        log_section "FULL TRAINING + EVALUATION CYCLE"
        
        run_experiment "train" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS" "$MAX_ITER"
        local train_exit=$?

        if [ $train_exit -eq 0 ]; then
            log_info "Training completed successfully, starting evaluation"
            sleep 2
            run_experiment "evaluate" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS"
        else
            log_error "Training failed with exit code $train_exit, skipping evaluation"
            exit $train_exit
        fi
        ;;
    list)
        list_experiments
        ;;
    analyze)
        check_common_prerequisites || exit 1
        if [ -z "$EXP_ID" ]; then
            log_error "No experiment ID specified for analysis"
            log_info "Usage: $0 analyze <experiment_id>"
            list_experiments
            exit 1
        fi
        analyze_experiment "$EXP_ID"
        ;;
    clean)
        clean_old_results 7
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_info "Operation completed successfully!"