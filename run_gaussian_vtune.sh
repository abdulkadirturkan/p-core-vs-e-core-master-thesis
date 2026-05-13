#!/bin/bash
# ============================================================
# VTune Profiling + 100-Run Benchmark — Gaussian Filter
# ============================================================

# set -e kapalı (log görmek için)

source /opt/intel/oneapi/vtune/latest/env/vars.sh 2>/dev/null || {
    echo "HATA: VTune bulunamadı!"
    exit 1
}

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
if [[ $paranoid -gt 0 ]]; then
    echo "UYARI: perf_event_paranoid=$paranoid (0 olmalı)"
    read -p "Otomatik düzeltmek için sudo gerekiyor. Devam edilsin mi? [y/N] " res
    if [[ "$res" =~ ^[yY] ]]; then
        sudo sysctl -w kernel.perf_event_paranoid=0
    else
        echo "İşlem iptal edildi. Manuel: sudo sysctl -w kernel.perf_event_paranoid=0"
        exit 1
    fi
fi

ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)
if [[ $ptrace_scope -gt 0 ]]; then
    echo "UYARI: ptrace_scope=$ptrace_scope (0 olmalı)"
    read -p "Otomatik düzeltmek için sudo gerekiyor. Devam edilsin mi? [y/N] " res
    if [[ "$res" =~ ^[yY] ]]; then
        sudo sysctl -w kernel.yama.ptrace_scope=0
    else
        echo "İşlem iptal edildi. Manuel: sudo sysctl -w kernel.yama.ptrace_scope=0"
        exit 1
    fi
fi

# ============================================================
# Yapılandırma
# ============================================================

APP="./bin/gaussian"

sizes=(3 7 11 15 19)
threads=(2 4 6 8 10 12)
runs=100

declare -A CORE_CPUS
CORE_CPUS["Unset"]=""
CORE_CPUS["P-Core"]="0-15"
CORE_CPUS["E-Core"]="16-27"

CORE_ORDER=("Unset" "P-Core" "E-Core")

VTUNE_ANALYSES=("uarch-exploration" "memory-access" "hpc-performance" "threading" "system-overview")

# ============================================================
# Sonuç klasörü
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="./vtune_results/gaussian_${TIMESTAMP}"

mkdir -p "$BASE_DIR"

ln -sfn "gaussian_${TIMESTAMP}" ./vtune_results/latest

# ============================================================
# Benchmark Fonksiyonu
# ============================================================

run_benchmark() {

    local core_name=$1
    local cpu_set=$2
    local size=$3
    local thread=$4
    local log_file=$5

    local total_time=0
    local successful_runs=0

    local min_time=""
    local max_time=""

    local total_wall=0
    local min_wall=""
    local max_wall=""

    local time_list=()
    local wall_list=()

    for ((i = 1; i <= runs; i++)); do

        start=$(date +%s%N)

        if [[ -z "$cpu_set" ]]; then
            output=$($APP "$size" "$thread")
        else
            output=$(taskset -c "$cpu_set" $APP "$size" "$thread")
        fi

        end=$(date +%s%N)

        runtime_line=$(echo "$output" | grep "Filtering completed in")

        if [[ $runtime_line =~ ([0-9.]+) ]]; then

            time_taken="${BASH_REMATCH[1]}"

            time_list+=("$time_taken")

            total_time=$(echo "$total_time + $time_taken" | bc)

            ((successful_runs++))

            if [[ -z $min_time || $(echo "$time_taken < $min_time" | bc) -eq 1 ]]; then
                min_time="$time_taken"
            fi

            if [[ -z $max_time || $(echo "$time_taken > $max_time" | bc) -eq 1 ]]; then
                max_time="$time_taken"
            fi
        fi

        duration_ns=$((end - start))

        duration_s=$(echo "scale=6; $duration_ns / 1000000000" | bc 2>/dev/null || echo "0")

        wall_list+=("$duration_s")

        total_wall=$(echo "$total_wall + $duration_s" | bc 2>/dev/null || echo "$total_wall")

        if [[ -z $min_wall || $(echo "$duration_s < $min_wall" | bc 2>/dev/null || echo 0) -eq 1 ]]; then
            min_wall="$duration_s"
        fi

        if [[ -z $max_wall || $(echo "$duration_s > $max_wall" | bc 2>/dev/null || echo 0) -eq 1 ]]; then
            max_wall="$duration_s"
        fi

        if (( i % 10 == 0 )); then
            echo -n "$i "
        fi
    done

    echo "done"

    if (( successful_runs > 0 )); then

        avg_time=$(echo "scale=6; $total_time / $successful_runs" | bc)

        avg_wall=$(echo "scale=6; $total_wall / $runs" | bc 2>/dev/null || echo "0")

        time_data=$(printf "%s\n" "${time_list[@]}")
        wall_data=$(printf "%s\n" "${wall_list[@]}")

        stddev_time=$(echo "$time_data" | awk -v avg="$avg_time" '
            { sum += ($1 - avg)^2 }
            END { if (NR>0) printf "%.6f", sqrt(sum / NR); else print "0" }
        ')

        stddev_wall=$(echo "$wall_data" | awk -v avg="$avg_wall" '
            { sum += ($1 - avg)^2 }
            END { if (NR>0) printf "%.6f", sqrt(sum / NR); else print "0" }
        ')

        {
            echo "==========================================="
            echo "Core: $core_name | Filter: gaussian | Radius: $size | Threads: $thread"
            echo "Runs: $runs"
            echo "Program-reported time:"
            echo "  Min:  $min_time s"
            echo "  Max:  $max_time s"
            echo "  Avg:  $avg_time s"
            echo "  Std:  $stddev_time s"
            echo "Wall-clock time:"
            echo "  Min:  $min_wall s"
            echo "  Max:  $max_wall s"
            echo "  Avg:  $avg_wall s"
            echo "  Std:  $stddev_wall s"
            echo
        } >> "$log_file"

    else
        echo "HATA: size=$size thread=$thread için runtime alınamadı" >> "$log_file"
    fi
}

# ============================================================
# VTune Profiling
# ============================================================

run_vtune_profile() {

    local core_name=$1
    local cpu_set=$2
    local size=$3
    local thread=$4
    local core_dir=$5

    for analysis in "${VTUNE_ANALYSES[@]}"; do

        result_dir="${core_dir}/vtune/size${size}_t${thread}/${analysis}"

        echo "    [VTUNE] $analysis ..."

        if [[ -z "$cpu_set" ]]; then
            # Construct analysis command with knobs if necessary
            case "$analysis" in
                "system-overview")
                    vtune -collect "$analysis" \
                        -knob analyze-power-usage=true \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
                "hpc-performance")
                    vtune -collect "$analysis" \
                        -knob collect-affinity=true \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
                *)
                    vtune -collect "$analysis" \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
            esac
        else
            case "$analysis" in
                "system-overview")
                    vtune -collect "$analysis" \
                        -knob analyze-power-usage=true \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- taskset -c "$cpu_set" $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
                "hpc-performance")
                    vtune -collect "$analysis" \
                        -knob collect-affinity=true \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- taskset -c "$cpu_set" $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
                *)
                    vtune -collect "$analysis" \
                        -result-dir "$result_dir" \
                        -quiet \
                        -- taskset -c "$cpu_set" $APP "$size" "$thread" 2>&1 | tail -2
                    ;;
            esac
        fi

        vtune -report summary \
            -result-dir "$result_dir" \
            -format csv \
            -report-output "${result_dir}/summary.csv" 2>/dev/null || true

        vtune -report hw-events \
            -result-dir "$result_dir" \
            -format csv \
            -report-output "${result_dir}/hw_events.csv" 2>/dev/null || true

        echo "    [TAMAM] $analysis"

        rm -rf \
            "${result_dir}/sqlite-db" \
            "${result_dir}/archive" \
            "${result_dir}/log" \
            "${result_dir}/config" \
            "${result_dir}/data"* \
            "${result_dir}"/*.vtune \
            2>/dev/null || true
    done
}

# ============================================================
# Başlangıç Bilgisi
# ============================================================

echo "============================================================"
echo "VTune + 100-Run Benchmark — Gaussian Filter"
echo "Tarih    : $(date)"
echo "CPU      : $(lscpu | grep -E 'Model name|Modem ismi' | sed 's/.*:\s*//')"
echo "Sizes    : ${sizes[*]}"
echo "Threads  : ${threads[*]}"
echo "Runs     : $runs"
echo "Çıktı    : $BASE_DIR"
echo "============================================================"
echo ""

# ============================================================
# Ana Döngü
# ============================================================

for core_name in "${CORE_ORDER[@]}"; do

    cpu_set="${CORE_CPUS[$core_name]}"

    safe_name=$(echo "$core_name" | tr ' ' '_' | tr '-' '_')

    core_dir="${BASE_DIR}/${safe_name}"

    log_file="${core_dir}/${safe_name}_benchmark.log"

    mkdir -p "$core_dir"

    > "$log_file"

    echo "============================================================"
    echo "CORE MODU: $core_name"
    if [[ -n "$cpu_set" ]]; then
        echo "CPU Set : $cpu_set"
    else
        echo "CPU Set : (serbest — OS kararı)"
    fi
    echo "Log     : $log_file"
    echo "============================================================"

    for size in "${sizes[@]}"; do
        for thread in "${threads[@]}"; do

            echo ""
            echo "  [$core_name] size=$size threads=$thread"

            echo "  --- VTune Profiling ---"
            run_vtune_profile "$core_name" "$cpu_set" "$size" "$thread" "$core_dir"

            echo -n "  --- 100-Run Benchmark (Progress: "
            run_benchmark "$core_name" "$cpu_set" "$size" "$thread" "$log_file"
            echo "  [TAMAM] Benchmark loglandı"
        done
    done

    echo ""
    echo "[$core_name] tüm testler tamamlandı. Log: $log_file"
    echo ""
done

# ============================================================
# Son Özet
# ============================================================

echo "============================================================"
echo "TÜM TESTLER TAMAMLANDI!"
echo "============================================================"
echo ""

echo "Sonuç dizinleri:"

for core_name in "${CORE_ORDER[@]}"; do

    safe_name=$(echo "$core_name" | tr ' ' '_' | tr '-' '_')

    echo "  $core_name : ${BASE_DIR}/${safe_name}/"

    echo "    Benchmark log : ${BASE_DIR}/${safe_name}/${safe_name}_benchmark.log"

    echo "    VTune sonuçları: ${BASE_DIR}/${safe_name}/vtune/"
done

echo ""

echo "En son sonuç symlink: ./vtune_results/latest"

echo ""

echo "GUI ile görüntülemek için:"
echo "  vtune-gui ${BASE_DIR}/<core>/vtune/<config>/<analysis>"

echo ""

echo "CSV raporları her analiz dizininde summary.csv ve hw_events.csv"

echo "============================================================"