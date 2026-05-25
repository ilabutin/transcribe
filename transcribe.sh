#!/bin/bash
# Transcribe any audio/video file with speaker diarization (WhisperX)
#
# Usage:
#   ./transcribe.sh <input_file> [<input_file2> ...] [options]
#
# Options:
#   --lang LANG        Language code (auto-detect if omitted)
#   --speakers N       Number of speakers (auto-detect if omitted)
#   --hf-token TOKEN   HuggingFace token (or set HF_TOKEN env var)
#   --no-diarize       Skip speaker diarization (faster, single-speaker talks)
#   --save-json        Save raw WhisperX segments as JSON alongside output
#   --model MODEL      Whisper model (default: large-v3)
#   --out FILE         Output file path (single-file mode only)
#
# Examples:
#   ./transcribe.sh ~/Downloads/radiodotnet-120.mp3
#   ./transcribe.sh ep1.mp3 ep2.mp3 ep3.mp3 --speakers 2
#   ./transcribe.sh ~/podcasts/*.mp3 --speakers 2
#   ./transcribe.sh meeting.mp4 --speakers 3 --lang en
#   ./transcribe.sh interview.m4a --out ~/Desktop/interview_transcript.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv_wx/bin/python"
PYTHON_SCRIPT="$SCRIPT_DIR/transcribe_wx.py"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file> [<input_file2> ...] [--lang LANG] [--speakers N] [--no-diarize] [--save-json] [--hf-token TOKEN] [--model large-v3] [--out FILE]"
    exit 1
fi

# Separate input files (positional args) from options
INPUT_FILES=()
EXTRA_ARGS=()
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --speakers|--lang|--model|--hf-token)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            INPUT_FILES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "Error: no input files specified"
    exit 1
fi

if [[ ${#INPUT_FILES[@]} -gt 1 && -n "$OUTPUT_FILE" ]]; then
    echo "Warning: --out ignored when processing multiple files"
    OUTPUT_FILE=""
fi

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Error: file not found: $INPUT_FILE"
        continue
    fi

    if [[ -n "$OUTPUT_FILE" ]]; then
        FILE_OUTPUT="$OUTPUT_FILE"
    else
        INPUT_DIR="$(dirname "$INPUT_FILE")"
        INPUT_BASE="$(basename "${INPUT_FILE%.*}")"
        FILE_OUTPUT="$INPUT_DIR/$INPUT_BASE.txt"
    fi

    echo "Input:  $INPUT_FILE"
    echo "Output: $FILE_OUTPUT"

    "$PYTHON" "$PYTHON_SCRIPT" "$INPUT_FILE" "$FILE_OUTPUT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
done
