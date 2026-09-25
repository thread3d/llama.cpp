#!/usr/bin/env bash
#
# Benchmark this macOS/AMD build of llama.cpp across CPU and 1/2/3-GPU Metal
# configurations, with repetitions, and emit a TSV plus a markdown summary.
#
# See docs/build-macos-amd.md for how to produce build-cpu and build-amd.
#
# Usage:
#   scripts/bench-macos-amd.sh
#
# Environment:
#   MODELS_DIR   where the GGUF files live     (default: $HOME/models)
#   REPS         repetitions per measurement   (default: 3)
#   OUT          TSV output path               (default: <repo>/bench-macos-amd.tsv)
#   RAW          raw JSON directory            (default: <repo>/bench-macos-amd-raw)
#   EXTRA_MODELS newline separated "label|path|n1|n2|n3" entries appended to the list

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
REPS="${REPS:-3}"
OUT="${OUT:-$ROOT/bench-macos-amd.tsv}"
RAW="${RAW:-$ROOT/bench-macos-amd-raw}"

CPU_BIN="$ROOT/build-cpu/bin/llama-bench"
AMD_BIN="$ROOT/build-amd/bin/llama-bench"

# smallest model in the sweep, used to probe device order cheaply
SMALL="${SMALL:-$HOME/Library/Caches/llama.cpp/hugging-quants_Llama-3.2-1B-Instruct-Q8_0-GGUF_llama-3.2-1b-instruct-q8_0.gguf}"

# label|path|n_cpu_moe for 1 GPU|for 2 GPUs|for 3 GPUs
# n_cpu_moe keeps MoE experts on the host so the model fits the VRAM of each
# configuration. These values are the ones the hardware needs, not tunables.
MODELS="
llama-3.2-1B-Q8_0|$SMALL|0|0|0
Ornith-1.5-9B-BF16|$MODELS_DIR/Ornith-1.5-9B-BF16.gguf|0|0|0
gemma-4-26B-A4B-Q4_K_M|$MODELS_DIR/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf|0|0|0
Ornith-1.5-35B-BF16|$MODELS_DIR/Ornith-1.5-35B-BF16.gguf|28|8|0
gpt-oss-120b-MXFP4|$MODELS_DIR/gpt-oss-120b-MXFP4-00001-of-00002.gguf|22|4|0
"
if [ -n "${EXTRA_MODELS:-}" ]; then
    MODELS="$MODELS
$EXTRA_MODELS"
fi

for f in "$CPU_BIN" "$AMD_BIN"; do
    if [ ! -x "$f" ]; then
        echo "missing $f -- build build-cpu and build-amd first (see docs/build-macos-amd.md)" >&2
        exit 1
    fi
done

mkdir -p "$RAW"

# ---------------------------------------------------------------------------
# The Metal device order is not stable across reboots, so probe it once and
# label the cards by name.
# ---------------------------------------------------------------------------
probe_devices() {
    TOSH_FA_AMD=1 GGML_METAL_DEVICE_LIST=0,1,2 timeout 900 "$AMD_BIN" \
        -m "$SMALL" -p 0 -n 1 -ngl 99 -r 1 2>&1 >/dev/null |
        sed -n 's/.*device \([0-9][0-9]*\): \([^(]*\) (.*/\1\t\2/p'
}

DEVFILE="$RAW/devices.tsv"
probe_devices >"$DEVFILE" 2>/dev/null || true

DUO1=$(awk -F'\t' '/W6800X/{print $1; exit}' "$DEVFILE")
DUO2=$(awk -F'\t' '/W6800X/{n++; if (n == 2) {print $1; exit}}' "$DEVFILE")
VEGA=$(awk -F'\t' '/Vega/{print $1; exit}' "$DEVFILE")
ALL=$(awk -F'\t' 'NF{printf "%s%s", sep, $1; sep=","}' "$DEVFILE")

if [ -z "$DUO1" ]; then
    echo "could not probe Metal devices:" >&2
    cat "$DEVFILE" >&2
    exit 1
fi

echo "== devices =="
cat "$DEVFILE"
echo "== configs =="
echo "  cpu  : CPU only, build-cpu"
echo "  gpu1 : W6800X Duo index $DUO1"
echo "  gpu2 : W6800X Duo indices $DUO1,$DUO2"
echo "  gpu3 : all indices ${ALL:-$DUO1}${VEGA:+, Vega II index $VEGA}"

# ---------------------------------------------------------------------------
# One measurement per (model, config). Resumable: an existing JSON is reused.
# ---------------------------------------------------------------------------
run_one() {
    local label="$1" path="$2" cfg="$3" ncmoe="$4" idxs="$5"
    local json="$RAW/${label}_${cfg}.json"
    local -a envs args

    if [ -s "$json" ]; then
        echo "  [skip] $label $cfg (have $json)"
        return 0
    fi

    if [ ! -f "$path" ]; then
        echo "  [miss] $label -- $path not found"
        return 0
    fi

    args=(-m "$path" -ngl 99 --load-mode none -fa auto -p 512 -n 128 -r "$REPS" -o json)

    case "$cfg" in
        cpu)
            args=(-m "$path" -ngl 0 -t 14 --load-mode none -p 512 -n 128 -r "$REPS" -o json)
            echo "  [run ] $label cpu (t=14)"
            timeout 3600 "$CPU_BIN" "${args[@]}" >"$json" 2>"$RAW/${label}_${cfg}.err" || rm -f "$json"
            return 0
            ;;
        gpu1) envs=(TOSH_FA_AMD=1 GGML_METAL_DEVICE_INDEX="$idxs") ;;
        *)    envs=(TOSH_FA_AMD=1 GGML_METAL_DEVICE_LIST="$idxs" TOSH_MGPU_EVENTS=1)
              args+=(--split-mode layer) ;;
    esac
    [ "$ncmoe" != "0" ] && args+=(-ncmoe "$ncmoe")

    echo "  [run ] $label $cfg (idx=$idxs ncmoe=$ncmoe)"
    env "${envs[@]}" timeout 3600 "$AMD_BIN" "${args[@]}" \
        >"$json" 2>"$RAW/${label}_${cfg}.err" || rm -f "$json"
}

: >"$OUT"
while IFS='|' read -r label path n1 n2 n3; do
    [ -z "${label:-}" ] && continue
    echo "-- $label"
    run_one "$label" "$path" cpu  "$n1" ""
    run_one "$label" "$path" gpu1 "$n1" "$DUO1"
    [ -n "$DUO2" ] && run_one "$label" "$path" gpu2 "$n2" "$DUO1,$DUO2"
    [ -n "$ALL" ]  && run_one "$label" "$path" gpu3 "$n3" "$ALL"
done <<EOF
$MODELS
EOF

# ---------------------------------------------------------------------------
# Collapse the raw JSON into the TSV and a markdown table.
# ---------------------------------------------------------------------------
python3 - "$RAW" "$OUT" "$REPS" <<'PY'
import json, os, sys

raw, out, reps = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for name in sorted(os.listdir(raw)):
    if not name.endswith(".json"):
        continue
    label, cfg = name[:-5].rsplit("_", 1)
    try:
        data = json.load(open(os.path.join(raw, name)))
    except Exception as e:
        rows.append((label, cfg, "?", "FAIL", str(e)))
        continue
    for r in data:
        n_p, n_g = r.get("n_prompt", 0), r.get("n_gen", 0)
        test = "pp%d" % n_p if n_p else "tg%d" % n_g if n_g else "?"
        rows.append((label, cfg, test, "%.2f" % r.get("avg_ts", 0.0),
                     "%.2f" % r.get("stddev_ts", 0.0)))

order = {"cpu": 0, "gpu1": 1, "gpu2": 2, "gpu3": 3}
rows.sort(key=lambda r: (r[0], order.get(r[1], 9), r[2]))

with open(out, "w") as f:
    f.write("model\tconfig\ttest\tt_s\tstddev\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

labels, cfgs = [], []
for r in rows:
    if r[0] not in labels: labels.append(r[0])
    if r[1] not in cfgs:  cfgs.append(r[1])
look = {(r[0], r[1], r[2]): r[3] for r in rows}

print()
print("| model | test | " + " | ".join(cfgs) + " |")
print("|---|---|" + "---|" * len(cfgs))
for lab in labels:
    for test in ("pp512", "tg128"):
        cells = [look.get((lab, c, test), "-") for c in cfgs]
        if all(x == "-" for x in cells):
            continue
        print("| %s | %s | %s |" % (lab, test, " | ".join(cells)))
print()
print("repetitions per cell: %s   (raw JSON in %s)" % (reps, raw))
print("t/s; TSV written to %s" % out)
PY
