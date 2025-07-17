#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
# ]
# ///
import argparse
from faster_whisper import WhisperModel, BatchedInferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with faster-whisper from the terminal."
    )
    parser.add_argument(
        "audio",
        help="Path to the audio file (mp3, wav, etc.)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Model size: tiny, base, small, medium, large, large-v2, large-v3, distil-large-v3",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Run on CPU or GPU",
    )
    parser.add_argument(
        "--compute_type",
        default="float16",
        choices=["float32", "float16", "int8"],
        help="Precision / quantization mode",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size for batched mode (0 = disable batched pipeline)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force decoding in a specific language (e.g. en, fr, de)",
    )
    parser.add_argument(
        "--word_timestamps", action="store_true", help="Output word-level timestamps"
    )
    parser.add_argument(
        "--vad_filter",
        action="store_true",
        help="Filter out non-speech segments via Silero VAD",
    )
    parser.add_argument(
        "--no-timestamps",
        dest="print_timestamps",
        action="store_false",
        help="Suppress all timestamps in the output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # load the model
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    # choose batched vs. standard transcription
    if args.batch_size > 0:
        pipeline = BatchedInferencePipeline(model=model)
        segments, info = pipeline.transcribe(
            args.audio,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            language=args.language,
            word_timestamps=args.word_timestamps,
            vad_filter=args.vad_filter,
        )
    else:
        segments, info = model.transcribe(
            args.audio,
            beam_size=args.beam_size,
            language=args.language,
            word_timestamps=args.word_timestamps,
            vad_filter=args.vad_filter,
        )

    # print detected language
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})\n")

    # iterate and print segments
    for segment in segments:
        if args.word_timestamps:
            for word in segment.words:
                if args.print_timestamps:
                    print(f"[{word.start:.2f}s → {word.end:.2f}s] {word.word}")
                else:
                    print(word.word)
        else:
            if args.print_timestamps:
                print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
            else:
                print(segment.text)


if __name__ == "__main__":
    main()
