#!/usr/bin/env bash

# BoloGAN Test Runner - Automation script for thesis experiments
# Author: Federico Santi

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

BOLOGAN_HOME="/home/saint/Documents/UNIBO/tesi"
FASTCALO_DIR="${BOLOGAN_HOME}/FastCaloChallenge"
CONTAINER_PATH="${BOLOGAN_HOME}/containers/FastCaloGANtainer_Plus.sif"
RESULTS_DIR="${BOLOGAN_HOME}/results"
ENHANCED_WRAPPER="${BOLOGAN_HOME}/scripts/enhanced_wrapper.sh"

# Default parameters
DEFAULT_DATASET="${FASTCALO_DIR}/input/dataset1/dataset_1_pions_1.hdf5"
DEFAULT_CONFIG_STRING="BNReLU_hpo27-M1" # BN = Batch Normalize
                                        # ReLu = Funzione di Attivazione ReLu (Rectified Linear Unit)
                                        # hpo27 = Hyper Parameters Optimization Config #27
                                        # M1 = Mask Variant (diversi preprocessing)

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

check_prerequisites() {
    echo "--> Checking prerequisites..."
    
    local issues=0
    
    # Check directories
    if [ ! -d "$FASTCALO_DIR" ]; then
        echo "!) FastCaloChallenge directory not found: $FASTCALO_DIR"
        issues=$((issues + 1))
    fi
    
    if [ ! -f "$CONTAINER_PATH" ]; then
        echo "!) Container not found: $CONTAINER_PATH"
        issues=$((issues + 1))
    fi
    
    if [ ! -f "$DEFAULT_DATASET" ]; then
        echo "!) Default dataset not found: $DEFAULT_DATASET"
        issues=$((issues + 1))
    fi
    
    if [ ! -f "$ENHANCED_WRAPPER" ]; then
        echo "!) Enhanced wrapper not found: $ENHANCED_WRAPPER"
        echo "   Please save the enhanced wrapper script as: $ENHANCED_WRAPPER"
        issues=$((issues + 1))
    fi
    
    # Check commands
    if ! command -v apptainer &> /dev/null; then
        echo "!) apptainer command not found"
        issues=$((issues + 1))
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "!) python3 command not found"
        issues=$((issues + 1))
    fi
    
    # Check system resources
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$mem_gb" -lt 16 ]; then
        echo "i) Available RAM: ${mem_gb}GB (recommended: 16GB+)"
    else
        echo "v) Available RAM: ${mem_gb}GB"
    fi
    
    local cpu_cores=$(nproc)
    echo "i) CPU cores: $cpu_cores"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        echo "v) NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        echo "i) No NVIDIA GPU detected (CPU-only mode)"
    fi
    
    if [ $issues -eq 0 ]; then
        echo -e "V) All prerequisites satisfied!\n"
        return 0
    else
        echo -e "!!) Found $issues issues. Please resolve them before proceeding.\n"
        return 1
    fi
}

show_optimization_tips() {
    echo ""
    echo "i) OPTIMIZATION TIPS (optional - set manually if desired):"
    echo "   For better CPU performance:"
    echo "     export OMP_NUM_THREADS=$(nproc)"
    echo ""
    echo "   For less TensorFlow logging:"
    echo "     export TF_CPP_MIN_LOG_LEVEL=2"
    echo ""
    echo "   For oneDNN optimizations:"
    echo "     export TF_ENABLE_ONEDNN_OPTS=1"
    echo ""
    echo "   To set all at once:"
    echo "     export OMP_NUM_THREADS=$(nproc) TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1"
    echo ""
}

setup_environment() {
    echo "--> Setting up environment..."
    
    # Create directories
    mkdir -p "$RESULTS_DIR"/{monitoring,analysis}
    
    # Check environment variables without modifying them
    echo "   Environment check:"
    echo "    - OMP_NUM_THREADS: ${OMP_NUM_THREADS:-'not set (will auto-detect)'}"
    echo "    - Available CPU cores: $(nproc)"
    
    if [ -z "$OMP_NUM_THREADS" ]; then
        echo "i) Tip: You can set OMP_NUM_THREADS=$(nproc) for optimal CPU usage"
    fi
    
    # Show TensorFlow environment (without modifying)
    echo "    - TF_CPP_MIN_LOG_LEVEL: ${TF_CPP_MIN_LOG_LEVEL:-'not set (default: 0)'}"
    echo "    - TF_ENABLE_ONEDNN_OPTS: ${TF_ENABLE_ONEDNN_OPTS:-'not set (default: auto)'}"
    
    echo "V) Environment setup complete (no modifications made)"
}

run_experiment() {
    local task=$1
    local dataset=${2:-$DEFAULT_DATASET}
    local config_string=${3:-$DEFAULT_CONFIG_STRING}
    local loading_args=${4:-""}
    local max_iter=${5:-5000}

    # Setting variable about version
    export APPTAINERENV_APPTAINER_VERSION=$(apptainer --version | cut -d' ' -f3)
    
    echo "--> Starting $task experiment"
    echo "      - Dataset: $dataset"
    echo "      - Config String: $config_string"
    echo "      - Loading Args: ${loading_args:-'(none)'}"
    echo "      - Max iterations: $max_iter"
    
    # Check if we're in a SLURM environment
    if [ ! -z "$SLURM_JOB_ID" ]; then
        echo "i) Running on SLURM node: $(hostname)"
        echo "      SLURM Job ID: $SLURM_JOB_ID"
    fi
    
    # Enter container and run enhanced wrapper
    if [ "$USE_CONTAINER" = "true" ]; then
        echo "📦 Using container: $CONTAINER_PATH"
        apptainer exec "$CONTAINER_PATH" bash "$ENHANCED_WRAPPER" "$task" "$dataset" "$config_string" "$loading_args" "$max_iter"
    else
        echo "🔧 Running without container (not recommended)"
        bash "$ENHANCED_WRAPPER" "$task" "$dataset" "$config_string" "$loading_args" "$max_iter"
    fi
}

list_experiments() {
    echo "📋 Available experiments:"
    echo ""
    
    if [ ! -d "$RESULTS_DIR/monitoring" ]; then
        echo "No experiments found in $RESULTS_DIR/monitoring"
        return 0
    fi
    
    local count=0
    for exp_dir in "$RESULTS_DIR/monitoring"/exp_*; do
        if [ -d "$exp_dir" ]; then
            local exp_id=$(basename "$exp_dir")
            local metadata_file="$exp_dir/metadata/experiment_metadata.json"
            local summary_file="$exp_dir/metadata/execution_summary.json"
            
            echo "🔬 $exp_id"
            
            if [ -f "$metadata_file" ]; then
                local task=$(python3 -c "import json; print(json.load(open('$metadata_file')).get('task', 'unknown'))" 2>/dev/null || echo "unknown")
                local start_time=$(python3 -c "import json; print(json.load(open('$metadata_file')).get('start_time', 'unknown'))" 2>/dev/null || echo "unknown")
                echo "   Task: $task"
                echo "   Started: $start_time"
            fi
            
            if [ -f "$summary_file" ]; then
                local duration=$(python3 -c "import json; print(json.load(open('$summary_file')).get('execution_summary', {}).get('duration_formatted', 'unknown'))" 2>/dev/null || echo "unknown")
                echo "   Duration: $duration"
            fi
            
            echo "   Path: $exp_dir"
            echo ""
            count=$((count + 1))
        fi
    done
    
    echo "Total experiments: $count"
}

analyze_experiment() {
    local exp_id=$1
    
    if [ -z "$exp_id" ]; then
        echo "❌ Please specify experiment ID"
        echo "Use '$0 list' to see available experiments"
        return 1
    fi
    
    local exp_dir="$RESULTS_DIR/monitoring/$exp_id"
    
    if [ ! -d "$exp_dir" ]; then
        echo "❌ Experiment not found: $exp_dir"
        return 1
    fi
    
    echo "📊 Analyzing experiment: $exp_id"
    
    # Check if analysis script exists
    local analysis_script="$BOLOGAN_HOME/scripts/analyze_monitoring.py"
    
    if [ ! -f "$analysis_script" ]; then
        echo "❌ Analysis script not found: $analysis_script"
        echo "   Please save the analysis script as: $analysis_script"
        return 1
    fi
    
    # Run analysis
    python3 "$analysis_script" "$exp_dir" --plots --report
    
    echo "✅ Analysis complete for $exp_id"
}

clean_old_results() {
    local days=${1:-7}
    
    echo "🧹 Cleaning results older than $days days..."
    
    if [ -d "$RESULTS_DIR/monitoring" ]; then
        find "$RESULTS_DIR/monitoring" -name "exp_*" -type d -mtime +$days -exec rm -rf {} \; 2>/dev/null || true
    fi
    
    echo "✅ Cleanup complete"
}

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
    echo "❌ No command specified"
    show_help
    exit 1
fi

# Execute command
case "$COMMAND" in
    setup)
        check_prerequisites
        setup_environment
        show_optimization_tips
        ;;
    train)
        check_prerequisites || exit 1
        
        run_experiment "train" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS" "$MAX_ITER"
        ;;
    evaluate)
        check_prerequisites || exit 1
        run_experiment "evaluate" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS"
        ;;
    full)
        check_prerequisites || exit 1
        
        echo "🔄 Running full training + evaluation cycle"
        
        run_experiment "train" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS" "$MAX_ITER"
        
        echo ""
        echo "🔄 Starting evaluation phase..."
        sleep 2
        
        run_experiment "evaluate" "$DATASET" "$CONFIG_STRING" "$LOADING_ARGS"
        ;;
    list)
        list_experiments
        ;;
    analyze)
        if [ -z "$EXP_ID" ]; then
            echo "❌ Please specify experiment ID to analyze"
            echo "Usage: $0 analyze <experiment_id>"
            echo ""
            list_experiments
            exit 1
        fi
        analyze_experiment "$EXP_ID"
        ;;
    clean)
        clean_old_results 7
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "--> Operation completed successfully!"