#!/usr/bin/env bash

# Enhanced wrapper for BoloGAN monitoring - Thesis Federico Santi
# Based on original wrapper.sh with added monitoring capabilities

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directories
source ./config.sh
FASTCALO_DIR="${BOLOGAN_HOME}/FastCaloChallenge"
RESULTS_DIR="${BOLOGAN_HOME}/results"
MONITORING_DIR="${RESULTS_DIR}/monitoring"

# Create monitoring directories
mkdir -p "${MONITORING_DIR}"

# Experiment metadata
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="${MONITORING_DIR}/${EXPERIMENT_ID}"
mkdir -p "${EXPERIMENT_DIR}"/{metadata,system_monitoring,system_monitoring/logs,process_monitoring,process_monitoring/logs,gpu_monitoring,logs,analysis}

# Sampling rates
ENERGY_SAMPLE_SECS=${ENERGY_SAMPLE_SECS:-0.5}
SYSTEM_SAMPLE_SECS=${SYSTEM_SAMPLE_SECS:-1}
CPU_SAMPLE_SECS=${CPU_SAMPLE_SECS:-0.5}

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
# UTILITY FUNCTIONS
# =============================================================================

# Function to add PID to monitoring list
add_monitoring_pid() {
    local pid=$1
    local name=$2
    
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        echo "$pid" >> "${EXPERIMENT_DIR}/monitoring_pids.txt"
        log_debug "Started $name monitoring (PID: $pid)"
    fi
}

# Function to test if turbostat actually works
test_turbostat() {
    if ! command -v turbostat &>/dev/null; then
        return 1  # Command not found
    fi
    
    log_debug "Testing turbostat capabilities..."
    # Test if turbostat can actually read energy values
    local test_output
    test_output=$(timeout 3 turbostat --show PkgWatt --interval 0.1 --num_iterations 1 2>&1)
    local exit_code=$?
    
    # Check for common permission errors
    if [[ $exit_code -ne 0 ]] || [[ "$test_output" == *"Permission denied"* ]] || [[ "$test_output" == *"Operation not permitted"* ]] || [[ "$test_output" == *"No such file or directory"* ]]; then
        log_debug "turbostat test failed: $test_output"
        return 1
    fi
    
    return 0
}

# Function to test if perf can read power events
test_perf_power() {
    if ! command -v perf &>/dev/null; then
        return 1  # Command not found
    fi
    
    log_debug "Testing perf power capabilities..."
    
    # First check if power events are listed
    local power_events
    power_events=$(perf list 2>/dev/null | grep -E "power/energy-(pkg|ram)/" | head -1)
    
    if [[ -z "$power_events" ]]; then
        log_debug "perf: no power events listed in 'perf list'"
        return 1
    fi
    
    log_debug "perf: found power events in list, testing actual access..."
    
    # Test actual access to power events with a quick measurement
    local test_output
    local exit_code
    
    # Run perf with explicit timeout and capture all output
    test_output=$(timeout 3 perf stat -a -e power/energy-pkg/ sleep 0.5 2>&1)
    exit_code=$?
    
    log_debug "perf test exit code: $exit_code"
    if [[ ${#test_output} -lt 200 ]]; then
        log_debug "perf test output: $test_output"
    else
        log_debug "perf test output (truncated): ${test_output:0:200}..."
    fi
    
    # Check for various failure conditions
    if [[ $exit_code -ne 0 ]]; then
        log_debug "perf power test failed with exit code: $exit_code"
        return 1
    fi
    
    # Parse output for failure indicators
    if [[ "$test_output" == *"Permission denied"* ]]; then
        log_debug "perf: permission denied accessing power events"
        return 1
    fi
    
    # Check for the specific "<not supported>"
    if [[ "$test_output" == *"<not supported>"* ]]; then
        log_debug "perf: power events explicitly marked as <not supported>"
        return 1
    fi
    
    # Additional check: look for actual numeric values (Joules)
    if [[ "$test_output" =~ [0-9]+(\.[0-9]+)?[[:space:]]+Joules ]]; then
        log_debug "perf: successfully measured energy values"
        return 0
    fi
    
    # If we can't find actual measurements, it's probably not working
    log_debug "perf: no valid energy measurements found in output"
    log_debug "perf: this usually means the events are listed but not accessible"
    return 1
}

# Function to test RAPL access
test_rapl_access() {
    if [[ ! -d /sys/class/powercap ]]; then
        return 1  # Directory doesn't exist
    fi
    
    log_debug "Testing RAPL accessibility..."
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
        log_debug "RAPL: no readable energy domains found"
        return 1
    fi
    
    return 0
}

debug_system_info() {
    log_debug "System capabilities:"
    log_debug "  - CPU cores: $(nproc)"
    log_debug "  - Memory: $(free -h | grep Mem | awk '{print $2}')"
    log_debug "  - Container: ${APPTAINER_CONTAINER:-'not in container'}"
    log_debug "Monitoring tool availability:"
    command -v vmstat >/dev/null && log_debug "    ✓ vmstat" || log_debug "    ✗ vmstat"
    command -v iostat >/dev/null && log_debug "    ✓ iostat" || log_debug "    ✗ iostat"
    command -v nvidia-smi >/dev/null && log_debug "    ✓ nvidia-smi" || log_debug "    ✗ nvidia-smi"
    
    log_debug "Energy monitoring capabilities:"
    if command -v turbostat >/dev/null; then
        if test_turbostat 2>/dev/null; then
            log_debug "    ✓ turbostat (functional)"
        else
            log_debug "    ⚠ turbostat (installed but no permissions/support)"
        fi
    else
        log_debug "    ✗ turbostat"
    fi
    
    if command -v perf >/dev/null; then
        if test_perf_power 2>/dev/null; then
            log_debug "    ✓ perf power events (functional)"
        else
            log_debug "    ⚠ perf (installed but no power events access)"
        fi
    else
        log_debug "    ✗ perf"
    fi
    
    if test_rapl_access 2>/dev/null; then
        log_debug "    ✓ RAPL (readable domains found)"
    else
        log_debug "    ✗ RAPL (no readable domains)"
    fi
    
    log_debug "Sampling configuration: energy=${ENERGY_SAMPLE_SECS}s, system=${SYSTEM_SAMPLE_SECS}s, cpu=${CPU_SAMPLE_SECS}s"
}

pidstat_to_csv() {
    local print_header="${1:-0}"    # 1 = print header, 0 = suppress header
    awk -v PRINT_HDR="$print_header" '
        BEGIN{
        OFS=","; printed_hdr = (PRINT_HDR ? 0 : 1); off=0;
        c_time=c_uid=c_pid=c_usr=c_sys=c_guest=c_pcpu=c_core=c_minf=c_majf=c_vsz=c_rss=c_mem=c_rds=c_wrs=c_ccwr=c_csw=c_ncsw=c_cmd=0;
        }
        # helper: reads field from mapped column applying offset for data rows
        function F(col, j){ if (col<=0) return ""; j=col-off; return (j>=1 && j<=NF)? $(j) : "" }
        # map columns when we find header containing "UID" and "PID" (with or without leading #)
        $0 ~ /UID[ \t]+PID/ {
        off = ($1=="#") ? 1 : 0;
        c_time=c_uid=c_pid=c_usr=c_sys=c_guest=c_pcpu=c_core=c_minf=c_majf=c_vsz=c_rss=c_mem=c_rds=c_wrs=c_ccwr=c_csw=c_ncsw=c_cmd=0;
        for (i=1;i<=NF;i++){
            tok=$i;
            if (off && i==1) continue;  # skip the "#" token
            if      (tok=="Time")        c_time=i;
            else if (tok=="UID")         c_uid=i;
            else if (tok=="PID")         c_pid=i;
            else if (tok=="%usr")        c_usr=i;
            else if (tok=="%system")     c_sys=i;
            else if (tok=="%guest")      c_guest=i;
            else if (tok=="%CPU")        c_pcpu=i;
            else if (tok=="CPU")         c_core=i;
            else if (tok=="minflt/s")    c_minf=i;
            else if (tok=="majflt/s")    c_majf=i;
            else if (tok=="VSZ")         c_vsz=i;
            else if (tok=="RSS")         c_rss=i;
            else if (tok=="%MEM")        c_mem=i;
            else if (tok ~ /^[kKmMgGtT]?B_rd\/s$/)    c_rds=i;
            else if (tok ~ /^[kKmMgGtT]?B_wr\/s$/)    c_wrs=i;
            else if (tok ~ /^[kKmMgGtT]?B_ccwr\/s$/)  c_ccwr=i;
            else if (tok=="cswch/s")     c_csw=i;
            else if (tok=="nvcswch/s")   c_ncsw=i;
            else if (tok=="Command")     c_cmd=i;
        }
        if (!printed_hdr) {
            print "timestamp,pid,uid,cpu_usr_percent,cpu_sys_percent,cpu_guest_percent,cpu_percent,cpu_core,minorflt_s,majorflt_s,vsz,rss,mem_percent,io_rd_per_s,io_wr_per_s,io_ccwr_per_s,cswch_s,nvcswch_s,command";
            printed_hdr=1;
        }
        next
        }
        # data row: numeric PID in mapped column
        (c_pid>0 && F(c_pid) ~ /^[0-9]+$/) {
        ts = (c_time>0 && F(c_time)!="") ? F(c_time) : "NA";
        # reconstruct complete command
        cstr="unknown";
        if (c_cmd>0){
            start=c_cmd-off; if (start<1) start=1;
            cstr=$start; for (i=start+1;i<=NF;i++) cstr=cstr" "$i;
        }
        gsub(/"/,"\"\"",cstr); cstr="\"" cstr "\"";
        print ts, F(c_pid), F(c_uid), F(c_usr), F(c_sys), F(c_guest), F(c_pcpu), F(c_core),
                F(c_minf), F(c_majf), F(c_vsz), F(c_rss), F(c_mem), F(c_rds), F(c_wrs),
                F(c_ccwr), F(c_csw), F(c_ncsw), cstr
        }
    '
}


# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

# Function to start system monitoring
start_monitoring() {
    local task=$1
    local sys_prefix="${EXPERIMENT_DIR}/system_monitoring/${task}_system"
    local sys_prefix_log="${EXPERIMENT_DIR}/system_monitoring/logs/${task}_system"
    local gpu_prefix="${EXPERIMENT_DIR}/gpu_monitoring/${task}_gpu"
    
    log_section "MONITORING SETUP"
    log_info "Task: ${task}"
    log_info "Data directory: ${EXPERIMENT_DIR}"
    
    log_subsection "System Monitoring"
    
    # CPU and Memory monitoring (check if commands exist)
    if command -v vmstat &> /dev/null; then
        setsid env LC_ALL=C stdbuf -oL -eL vmstat -n $SYSTEM_SAMPLE_SECS \
        | awk -v TS_FMT="+%Y-%m-%dT%H:%M:%S%z" '
            BEGIN { have_header=0; skip_first=0 }
            # skip decorative row "procs ---memory--- ... ---cpu---"
            /^procs/ { next }
            # capture column headers (r b swpd free ... st)
            /^[[:space:]]*r[[:space:]]+b[[:space:]]+swpd/ {
                if (!have_header) {
                    line=$0
                    sub(/^[[:space:]]+/, "", line)
                    gsub(/[[:space:]]+/, ",", line)
                    print "timestamp," line
                    have_header=1
                }
                next
            }
            # data rows
            /^[[:space:]]*[0-9]/ {
                if (!skip_first) { skip_first=1; next }  # skip first sample (averages since boot)
                cmd="date " TS_FMT
                cmd | getline ts
                close(cmd)
                line=$0
                sub(/^[[:space:]]+/, "", line)
                gsub(/[[:space:]]+/, ",", line)
                print ts "," line
                fflush()
            }
        ' > "${sys_prefix}_vmstat.csv" 2>&1 &
        
        local awk_pid = $!
        local pgid=$(ps -o pgid= -p "$awk_pid" | tr -d '[:space:]')

        add_monitoring_pid $pgid "vmstat"
        log_info "vmstat: enabled (1s interval, CSV with timestamp from vmstat header)"
    else
        log_warn "vmstat: not available - CPU monitoring limited"
    fi
    
    # I/O monitoring (iostat -> CSV con timestamp: CPU e Device)
    if command -v iostat &> /dev/null; then

        # --- CPU CSV (iostat -c -y 1) ---
        setsid env LC_ALL=C stdbuf -oL -eL iostat -c -y "$SYSTEM_SAMPLE_SECS" \
        | awk '
            BEGIN { printed_hdr=0; state=0 }
            /^Linux / || /^$/ { next }                 # banner and empty lines
            /^avg-cpu:/ { state=1; next }              # start CPU report

            state==1 {
                gsub(/^[ \t]+|[ \t]+$/,"")
                if ($0 ~ /^%user[ \t]+%nice[ \t]+%system[ \t]+%iowait[ \t]+%steal[ \t]+%idle$/) {
                    if (!printed_hdr) { gsub(/[ \t]+/, ","); print "timestamp," $0; printed_hdr=1 }
                    state=2; next
                } else {
                    if (!printed_hdr) { print "timestamp,%user,%nice,%system,%iowait,%steal,%idle"; printed_hdr=1 }
                    ts = strftime("%Y-%m-%dT%H:%M:%S%z", systime())
                    gsub(/[ \t]+/, ",")
                    print ts "," $0
                    fflush()
                    state=0; next
                }
            }

            state==2 {
                ts = strftime("%Y-%m-%dT%H:%M:%S%z", systime())
                gsub(/^[ \t]+|[ \t]+$/,"")
                gsub(/[ \t]+/, ",")
                print ts "," $0
                fflush()
                state=0; next
            }
        ' >> "${sys_prefix}_iostat_cpu.csv" &

        local awk_pid=$!
        local pgid=$(ps -o pgid= -p "$awk_pid" | tr -d '[:space:]')
        add_monitoring_pid $pgid "iostat_cpu"

        # --- DEVICE CSV (iostat -xd -y 1 -k) ---
        setsid env LC_ALL=C stdbuf -oL -eL iostat -xd -y -k "$SYSTEM_SAMPLE_SECS" \
        | awk '
            BEGIN { have_h=0; in_tbl=0; ts="" }
            /^Linux / { next }               # banner
            /^$/ { in_tbl=0; next }          # end of table for this report

            # Device table header
            /^Device:/ {
                hdr=$0
                sub(/^Device:[ \t]*/,"device,", hdr)
                gsub(/[ \t]+/, ",", hdr)
                if (!have_h) { print "timestamp," hdr; have_h=1 }
                ts = strftime("%Y-%m-%dT%H:%M:%S%z", systime())  # same TS for all rows in this block
                in_tbl=1; next
            }

            # Device rows (sda, dm-0, dm-1, ...)
            in_tbl {
                line=$0
                gsub(/^[ \t]+|[ \t]+$/,"", line)
                gsub(/[ \t]+/, ",", line)
                print ts "," line
                fflush()
            }
        ' >> "${sys_prefix}_iostat_dev.csv" &

        local awk_pid=$!
        local pgid=$(ps -o pgid= -p "$awk_pid" | tr -d '[:space:]')
        add_monitoring_pid $pgid "iostat_dev"

        log_info "iostat: enabled (${SYSTEM_SAMPLE_SECS}s; CPU -> ${sys_prefix}_iostat_cpu.csv, DEV -> ${sys_prefix}_iostat_dev.csv)"
    else
        log_warn "iostat: not available - I/O monitoring disabled"
    fi
    
    # Free monitoring
    setsid env LC_ALL=C stdbuf -oL -eL free -m -s "$SYSTEM_SAMPLE_SECS" \
    | awk -v MEM="${sys_prefix}_free_mem.csv" -v SWP="${sys_prefix}_free_swap.csv" '
        BEGIN { mem_hdr=0; swp_hdr=0; ts="" }

        # Header row (columns)
        /^[[:space:]]+total[[:space:]]+used[[:space:]]+free[[:space:]]+shared[[:space:]]+buff\/cache[[:space:]]+available/ {
            line=$0
            gsub(/^[[:space:]]+/, "", line)
            gsub(/[[:space:]]+/, ",", line)              # "total,used,free,shared,buff/cache,available"
            if (!mem_hdr) { print "timestamp," line > MEM; mem_hdr=1; fflush(MEM) }
            if (!swp_hdr) { print "timestamp,total,used,free" > SWP; swp_hdr=1; fflush(SWP) }
            ts = strftime("%Y-%m-%dT%H:%M:%S%z", systime())  # same TS for Mem and Swap in this block
            next
        }

        # Mem: row
        /^Mem:/ {
            line=$0
            sub(/^Mem:[[:space:]]+/, "", line)
            gsub(/[[:space:]]+/, ",", line)              # values in MiB
            print ts "," line >> MEM
            fflush(MEM)
            next
        }

        # Swap: row
        /^Swap:/ {
            line=$0
            sub(/^Swap:[[:space:]]+/, "", line)
            gsub(/[[:space:]]+/, ",", line)
            # Keep only first 3 fields (total,used,free)
            n=split(line, a, ",")
            out=a[1] "," a[2] "," a[3]
            print ts "," out >> SWP
            fflush(SWP)
            next
        }
    ' &

    local awk_pid=$!
    local pgid=$(ps -o pgid= -p "$awk_pid" | tr -d '[:space:]')
    add_monitoring_pid $pgid "free"

    log_info "free: enabled (${SYSTEM_SAMPLE_SECS}s interval, CSV -> ${sys_prefix}_free_mem.csv & ${sys_prefix}_free_swap.csv)"

    # System-wide pidstat monitoring
    if command -v pidstat &> /dev/null; then
        {
            echo "$(date '+%Y-%m-%d %H:%M:%S') System-wide pidstat monitor started" >&2

            LC_ALL=C LANG=C pidstat -h -u -r -d -w -l | head -5 | pidstat_to_csv 1 | head -1

            while [[ -f "${EXPERIMENT_DIR}/monitoring_active" ]]; do
                # Execute pidstat for all system processes
                LC_ALL=C LANG=C pidstat -h -u -r -d -w -l "${SYSTEM_SAMPLE_SECS}" 1 \
                | pidstat_to_csv \
                | sort -t, -k7,7nr \
                | head -10

                sleep "${SYSTEM_SAMPLE_SECS}"
            done

            echo "$(date '+%Y-%m-%d %H:%M:%S') System-wide pidstat monitoring completed" >&2
        } > "${sys_prefix}_pidstat.csv" 2> "${sys_prefix_log}_pidstat.log" &

        add_monitoring_pid $! "pidstat"
        log_info "pidstat system: enabled (${SYSTEM_SAMPLE_SECS}s interval)"
    else
        log_warn "pidstat system: not available (sysstat package required)"
    fi

    log_subsection "CPU Frequency Monitoring"    

    # CPU Frequency monitoring
    if [[ -f /proc/cpuinfo ]]; then
        cpu_count=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "0")
            
        if [[ $cpu_count -gt 0 ]]; then
            {
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
                    
                    sleep $CPU_SAMPLE_SECS  # 2Hz sampling
                done

            } > "${sys_prefix}_cpu_freq.csv" 2> "${sys_prefix_log}_cpu_freq.log" &
        
            add_monitoring_pid $! "cpu_freq"
            log_info "CPU frequency: ${cpu_count} cores at ${CPU_SAMPLE_SECS}s interval"
        else
            log_warn "CPU frequency: unable to determine CPU count"
        fi        
    else
        log_warn "CPU frequency: /proc/cpuinfo not available"
    fi

    log_subsection "Thermal Monitoring"

    # Thermal monitoring
    if [[ -d /sys/class/thermal ]]; then
        # Count thermal zones first
        local thermal_count=0
        for thermal_path in /sys/class/thermal/thermal_zone*; do
            if [[ -d "$thermal_path" && -f "$thermal_path/temp" ]]; then
                ((thermal_count++))
            fi
        done
    
        if [[ $thermal_count -gt 0 ]]; then
            {
                # Discovery inside subshell
                thermal_zones=()
                for thermal_path in /sys/class/thermal/thermal_zone*; do
                    if [[ -d "$thermal_path" && -f "$thermal_path/temp" ]]; then
                        zone_name=$(basename "$thermal_path")
                        thermal_zones+=("$zone_name")
                        echo "# Discovered thermal zone: $zone_name ($thermal_path)" >&2
                    fi
                done
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
                        
                        # Read temperature from each thermal zone
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
                        sleep $CPU_SAMPLE_SECS  # 2Hz sampling
                    done
            } > "${sys_prefix}_thermal.csv" 2> "${sys_prefix_log}_thermal.log" &
        
            add_monitoring_pid $! "thermal"
            log_info "thermal: ${thermal_count} zones at ${CPU_SAMPLE_SECS}s interval"
        else
            log_warn "thermal: no readable thermal zones found"
        fi
    else
        log_warn "thermal: /sys/class/thermal not available"
    fi

    log_subsection "Energy Monitoring"

    # Energy monitoring (if available)
    # Try turbostat with actual capability test
    if test_turbostat; then
        turbostat --show PkgWatt,CorWatt,GFXWatt,PkgTmp --interval $ENERGY_SAMPLE_SECS > "${sys_prefix}_energy_turbostat.log" 2>&1 &
        add_monitoring_pid $! "turbostat"
        log_info "energy: turbostat at ${ENERGY_SAMPLE_SECS}s interval"
    # Try perf with actual capability test
    elif test_perf_power; then
        local interval=$(echo "($ENERGY_SAMPLE_SECS * 1000) / 1" | bc | cut -d'.' -f1)
        perf stat -a -I $interval -e power/energy-pkg/ -e power/energy-ram/ > "${sys_prefix}_energy_perf.log" 2>&1 &
        add_monitoring_pid $! "perf_power"
        log_info "energy: perf power events at ${interval}ms interval"
    elif test_rapl_access; then

        declare -A rapl_domains
        declare -a rapl_order
        
        # Scan all intel-rapl domains
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
            elog_warn "energy: no readable RAPL domains found"
        else
            {
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
                    
                    # Read all domains in discovered order
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
                    sleep $ENERGY_SAMPLE_SECS  # higher granularity for energy monitoring
                done
            } > "${sys_prefix}_energy_rapl.csv" 2> "${sys_prefix_log}_energy_rapl.log" &

            add_monitoring_pid $! "rapl"
            log_info "energy: RAPL ${#rapl_order[@]} domains at ${ENERGY_SAMPLE_SECS}s interval"

        fi
    else
        log_warn "energy: no monitoring source available (turbostat/perf/rapl)"
    fi

    log_subsection "GPU Monitoring"

    # GPU monitoring (if NVIDIA GPU available)
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv -l $ENERGY_SAMPLE_SECS > "${gpu_prefix}.csv" 2>&1 &
        add_monitoring_pid $! "nvidia-smi"
        log_info "GPU: nvidia-smi at ${ENERGY_SAMPLE_SECS}s interval"        
    else
        log_warn "GPU: nvidia-smi not available"
    fi

    log_info "System monitoring setup completed"
}


# Function to monitor specific process group
start_process_monitoring() {
    local python_pgid=$1
    local process_name=$2
    local output_prefix="${EXPERIMENT_DIR}/process_monitoring/${process_name}_process"
    local output_prefix_log="${EXPERIMENT_DIR}/process_monitoring/logs/${process_name}_process"

    log_subsection "Process Group Monitoring"
    log_info "Monitoring process group: $process_name (PGID: $python_pgid)"
    
    # 1. Process I/O monitoring per process group
    {
        echo "$(date '+%Y-%m-%d %H:%M:%S') Process group I/O monitor started for PGID: $python_pgid" >&2
        echo "timestamp,pgid,pid,read_bytes,write_bytes,read_syscalls,write_syscalls,read_chars,write_chars,cancelled_write_bytes"
        
        while kill -0 "-$python_pgid" 2>/dev/null; do
            current_time=$(date '+%Y-%m-%d %H:%M:%S')
            
            # Find all PIDs belonging to process group
            group_pids=$(pgrep -g "$python_pgid" 2>/dev/null || true)
            
            if [[ -n "$group_pids" ]]; then
                for pid in $group_pids; do
                    io_file="/proc/$pid/io"
                    if [[ -r "$io_file" ]]; then
                        awk -v ts="$current_time" -v pg="$python_pgid" -v pid="$pid" '
                            BEGIN { r=0; w=0; rs=0; ws=0; rc=0; wc=0; cw=0 }
                            /^read_bytes:/             { r=$2 }
                            /^write_bytes:/            { w=$2 }
                            /^syscr:/                  { rs=$2 }
                            /^syscw:/                  { ws=$2 }
                            /^rchar:/                  { rc=$2 }
                            /^wchar:/                  { wc=$2 }
                            /^cancelled_write_bytes:/  { cw=$2 }
                            END {
                                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                                       ts, pg, pid, r, w, rs, ws, rc, wc, cw
                            }
                        ' "$io_file"
                    fi
                done
            fi
            
            sleep $SYSTEM_SAMPLE_SECS
        done
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') Process group I/O monitoring completed for PGID: $python_pgid" >&2
    } > "${output_prefix}_io.csv" 2> "${output_prefix_log}_io.log" &

    add_monitoring_pid $! "process_io"

    # 2. Perf monitoring per process group
    if command -v perf &> /dev/null; then
        {
            echo "$(date '+%Y-%m-%d %H:%M:%S') Starting perf monitoring for PGID $python_pgid" >&2
            
            # Calculate interval in milliseconds
            local interval=$(echo "($SYSTEM_SAMPLE_SECS * 1000) / 1" | bc | cut -d'.' -f1)
            
            # Get all current PIDs of process group (including existing children)
            group_pids=$(pgrep -g "$python_pgid" 2>/dev/null || true)
            
            if [[ -n "$group_pids" ]]; then
                # Build comma-separated PID list for perf
                pid_list=$(echo "$group_pids" | tr '\n' ',' | sed 's/,$//')
                
                echo "$(date '+%Y-%m-%d %H:%M:%S') Monitoring PIDs: $pid_list" >&2
                
                # Monitor all PIDs in group with parseable output
                perf stat -p "$pid_list" -I $interval -x ';' --no-big-num \
                    -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,page-faults,context-switches,cpu-migrations \
                    -o "${output_prefix}_perf_raw.csv" --append
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') No PIDs found in process group $python_pgid" >&2
            fi
                
            echo "$(date '+%Y-%m-%d %H:%M:%S') Perf monitoring completed for PGID $python_pgid" >&2
                
        } > "${output_prefix_log}_perf.log" 2>&1 &
        
        add_monitoring_pid $! "process_perf"
        log_info "Process perf: started (${interval}ms interval)"
    else
        log_warn "Process perf: not available"
    fi

    # 3. pidstat monitoring per process group
    if command -v pidstat &> /dev/null; then
        {
            echo "$(date '+%Y-%m-%d %H:%M:%S') Process group pidstat monitor started for PGID: $python_pgid" >&2

            LC_ALL=C LANG=C pidstat -h -u -r -d -w -l | head -5 | pidstat_to_csv 1 | head -1

            while kill -0 "-$python_pgid" 2>/dev/null; do
                # Find all current PIDs of process group
                group_pids=$(pgrep -g "$python_pgid" 2>/dev/null || true)

                if [[ -n "$group_pids" ]]; then
                    # Build CSV list for pidstat
                    pid_csv=$(echo "$group_pids" | tr '\n' ',' | sed 's/,$//')

                    # Execute pidstat
                    LC_ALL=C LANG=C pidstat -h -u -r -d -w -l -p "$pid_csv" "${SYSTEM_SAMPLE_SECS}" 1 \
                    | pidstat_to_csv
                fi

                sleep "${SYSTEM_SAMPLE_SECS}"
            done

            echo "$(date '+%Y-%m-%d %H:%M:%S') Process group pidstat monitoring completed for PGID: $python_pgid" >&2
        } > "${output_prefix}_pidstat.csv" 2> "${output_prefix_log}_pidstat.log" &

        add_monitoring_pid $! "process_pidstat"
        log_info "Process pidstat: started (${SYSTEM_SAMPLE_SECS}s interval)"
    else
        log_warn "Process pidstat: not available (sysstat package required)"
    fi

    log_info "Process group monitoring started for $process_name (PGID: $python_pgid)"
}

# Unified function to stop all monitoring processes
stop_monitoring() {
    log_section "MONITORING CLEANUP"

    # Remove flag for cooperative shutdown
    local flag="${EXPERIMENT_DIR}/monitoring_active"
    rm -f "$flag" 2>/dev/null || true
    log_info "Stopping process monitoring (flag removed)"

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
        log_info "Terminating ${#all_pids[@]} monitoring processes"
        
        # TERM signal to all processes
        for pid in "${all_pids[@]}"; do
                kill -TERM "-$pid" 2>/dev/null || true
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
            log_warn "Force killing ${#all_pids[@]} stubborn processes"
            for pid in "${all_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            done
        fi
    fi

    # 5) Cleanup PID files
    rm -f "${EXPERIMENT_DIR}/monitoring_pids.txt" 2>/dev/null || true

    log_info "Monitoring cleanup completed"
}

handle_interrupt() {
    log_info "Interrupt or Termination received - stopping Python process"
    
    # Kill Python process
    if [[ -f "${EXPERIMENT_DIR}/python_pgid.txt" ]]; then
        local python_pgid=$(cat "${EXPERIMENT_DIR}/python_pgid.txt")
        if [[ -n "$python_pgid" && "$python_pgid" =~ ^[0-9]+$ ]] && kill -0 "$python_pgid" 2>/dev/null; then
            log_info "Terminating Python process (PID: $python_pgid)"
            kill -TERM "-$python_pgid" 2>/dev/null || true
        fi
    fi
    
    # Don't exit here - let normal flow continue
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

    log_section "EXECUTION SUMMARY"
    log_info "Task: ${task}"
    log_info "Duration: ${hours}h ${minutes}m ${seconds}s"
    log_info "Experiment ID: ${EXPERIMENT_ID}"
    log_info "Results saved to: ${EXPERIMENT_DIR}"
}

# Saving wrapper pid for graceful shutdown
echo "$$" > "${EXPERIMENT_DIR}/wrapper_pid.txt"
log_info "Enhanced wrapper PID saved: $$"

# Set trap for cleanup on exit
trap 'handle_interrupt' INT TERM
trap 'stop_monitoring' EXIT

# =============================================================================
# ORIGINAL WRAPPER.SH LOGIC (MODIFIED)
# =============================================================================

# Change to training directory
cd "${FASTCALO_DIR}/training"

log_section "ENHANCED WRAPPER STARTING"
log_info "Arguments: $*"
log_info "Experiment ID: ${EXPERIMENT_ID}"
log_info "Results directory: ${EXPERIMENT_DIR}"

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

log_subsection "Configuration Parsing"
log_info "input: $input"
log_info "config_string: $config_string"
log_info "  model: $model"
log_info "  config: $config (→ config/config_${config}.json)"
log_info "  mask: $mask"
log_info "  prep: $prep"
log_info "  label_scheme: $label_scheme"
log_info "  split_energy: $split_energy"
log_info "loading: $loading"
log_info "max_iter: $max_iter"

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
    command=(python3 train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${config_string} -c ../config/config_${config}.json ${train_addition} --max_iter ${max_iter})
else
    command=(python3 evaluate.py -i ${input} -t ../output/dataset${ds}/${version}/${config_string} --checkpoint ${evaluate_addition} --debug --save_h5)
fi

log_info "Final command: $command"

# =============================================================================
# EXECUTION WITH MONITORING
# =============================================================================

# Save experiment metadata
start_timestamp=$(date +%s)
save_experiment_metadata "$task" "$input" "$config_string" "$(date -d @$start_timestamp)"

# Create monitoring active flag
touch "${EXPERIMENT_DIR}/monitoring_active"

# Debug system info
debug_system_info

# Start monitoring
start_monitoring "$task"

log_section "COMMAND EXECUTION"
log_info "Starting execution at $(date)"

# Execute the command and capture output
exec > >(tee "${EXPERIMENT_DIR}/logs/${task}_stdout.log") 2> >(tee "${EXPERIMENT_DIR}/logs/${task}_stderr.log" >&2)

# Execute the actual command and saving python process PID
#eval "setsid exec $command" &
setsid "${command[@]}" &
python_pgid=$!

echo "$python_pgid" > "${EXPERIMENT_DIR}/python_pgid.txt"
log_info "Python process started (PID: $python_pgid)"

# Start process-specific monitoring
start_process_monitoring $python_pgid $task

# Waiting python process termination
wait $python_pgid
command_exit_code=$?

rm -f "${EXPERIMENT_DIR}/python_pgid.txt" 2>/dev/null || true

log_section "EXECUTION COMPLETED"
log_info "End time: $(date)"
log_info "Exit code: $command_exit_code"

# Calculate and save summary
end_timestamp=$(date +%s)
calculate_summary "$task" "$start_timestamp" "$end_timestamp"

cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds

rm -f "${EXPERIMENT_DIR}/wrapper_pid.txt" 2>/dev/null || true
exit $command_exit_code

# Cleanup will be called automatically by trap