#!/usr/bin/env python3
"""
Transcribe audio/video with speaker diarization using WhisperX.

Usage:
    python transcribe_wx.py <input_file> [output_file] [--lang ru] [--speakers N] [--hf-token TOKEN]

Output format (same as before):
    start_sec end_sec SPEAKER_XX text
"""

import sys
import os
import warnings
import argparse
import json
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

def combine_segments(segments):
    """Merge consecutive segments from the same speaker into longer phrases."""
    combined = []
    cur_speaker = None
    cur_text = []
    cur_start = None
    cur_end = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg["start"]
        end = seg["end"]

        if speaker != cur_speaker or (cur_text and cur_text[-1][-1] in ".!?"):
            if cur_text:
                combined.append({
                    "start": cur_start,
                    "end": cur_end,
                    "speaker": cur_speaker,
                    "text": " ".join(cur_text),
                })
            cur_speaker = speaker
            cur_text = [text]
            cur_start = start
            cur_end = end
        else:
            cur_text.append(text)
            cur_end = end

    if cur_text:
        combined.append({
            "start": cur_start,
            "end": cur_end,
            "speaker": cur_speaker,
            "text": " ".join(cur_text),
        })

    return combined


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video with speaker diarization")
    parser.add_argument("input", help="Input audio/video file (mp3, mp4, wav, m4a, ...)")
    parser.add_argument("output", nargs="?", help="Output text file (default: input path with .txt)")
    parser.add_argument("--lang", default=None, help="Language code, e.g. ru, en (default: auto-detect)")
    parser.add_argument("--speakers", type=int, default=None, help="Number of speakers (auto-detect if omitted)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for diarization models")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization (transcription only)")
    parser.add_argument("--save-json", action="store_true", help="Save raw WhisperX segments as JSON")
    parser.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    base = os.path.splitext(input_path)[0]
    output_path = args.output or (base + ".txt")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not args.no_diarize and not hf_token:
        # Try reading from huggingface cache
        token_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()

    import whisperx
    import torch

    # CTranslate2 (Whisper transcription) does not support MPS — use CPU on Apple Silicon.
    # PyTorch models (alignment, diarization) support MPS and run much faster on Apple Silicon.
    if torch.cuda.is_available():
        ct2_device = "cuda"
        compute_type = "float16"
        torch_device = "cuda"
    else:
        ct2_device = "cpu"
        compute_type = "int8"
        torch_device = "mps" if torch.backends.mps.is_available() else "cpu"

    log(f"CT2 device: {ct2_device}, torch device: {torch_device}, model: {args.model}, language: {args.lang or 'auto-detect'}")

    # Step 1: Transcribe
    log("Loading Whisper model...")
    model = whisperx.load_model(args.model, ct2_device, compute_type=compute_type, language=args.lang)

    log("Loading audio...")
    audio = whisperx.load_audio(input_path)
    duration_sec = len(audio) / 16000
    duration_min = int(duration_sec // 60)
    log(f"Audio duration: {duration_min}m {duration_sec % 60:.0f}s")

    log("Transcribing...")
    result = model.transcribe(audio, batch_size=16, language=args.lang, print_progress=True)
    detected_lang = result.get("language", args.lang or "ru")
    log(f"Transcribed {len(result['segments'])} segments, detected language: {detected_lang}")

    del model
    if ct2_device == "cuda":
        torch.cuda.empty_cache()

    # Step 2: Align (word-level timestamps)
    log("Aligning word timestamps...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=torch_device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, torch_device,
                                return_char_alignments=False)
        del model_a
    except Exception as e:
        log(f"Alignment skipped: {e}")

    # Step 3: Diarize
    if not args.no_diarize:
        if not hf_token:
            log("Warning: no HF_TOKEN found — skipping diarization. "
                "Set HF_TOKEN env var or pass --hf-token.")
        else:
            log("Running speaker diarization...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(token=hf_token, device=torch_device)
                kwargs = {}
                if args.speakers:
                    kwargs["num_speakers"] = args.speakers
                diarize_segments = diarize_model(audio, **kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                log("Diarization complete")
            except Exception as e:
                log(f"Diarization failed: {e}")

    # Save intermediate JSON (only if explicitly requested)
    if args.save_json:
        json_path = base + "_whisperx.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result["segments"], f, ensure_ascii=False, indent=2)
        log(f"Raw segments saved to {json_path}")

    # Step 4: Combine and write output
    segments = result["segments"]
    combined = combine_segments(segments)

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in combined:
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(f"{seg['start']:.2f} {seg['end']:.2f} {speaker} {seg['text']}\n")

    log(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
