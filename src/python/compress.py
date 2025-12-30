# /// script
# requires-python = ">=3.14"
# ///
import argparse
import tarfile
import time
from pathlib import Path
from compression import zstd


def archive(source: str, output: str, level: int, threads: int):
    src_path = Path(source)

    # If no output name is provided, append .tar.zst to the source name
    if not output:
        output = f"{src_path.name}.tar.zst"

    # Setup Advanced Parameters from 3.14 Docs
    # nb_workers > 0 enables multi-threaded compression
    # checksum_flag adds an XXHash64 for data integrity
    options = {
        zstd.CompressionParameter.nb_workers: threads,
        zstd.CompressionParameter.checksum_flag: 1,
    }

    print(f"Archiving: {src_path} -> {output}")
    print(f"Settings: Level {level}, Threads: {threads if threads > 0 else 'Auto'}")

    start_time = time.perf_counter()

    # Open the zstd stream using the new standard library module
    with zstd.open(output, mode="wb", level=level, options=options) as zstd_stream:
        # Wrap the zstd stream in a Tarfile to handle folders/metadata
        with tarfile.open(fileobj=zstd_stream, mode="w|") as tar:
            tar.add(src_path, arcname=src_path.name)

    end_time = time.perf_counter()

    size_mb = Path(output).stat().st_size / (1024 * 1024)
    print(f"Successfully archived in {end_time - start_time:.2f} seconds.")
    print(f"Final size: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python 3.14 Zstandard Archiver")
    parser.add_argument(
        "-i", "--input", required=True, help="Folder or file to compress"
    )
    parser.add_argument(
        "-o", "--output", help="Output filename (default: input.tar.zst)"
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=3,
        help="Compression level (1-19, or 20+ for Ultra). Default is 3.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="Number of CPU threads (0 for auto-detect).",
    )

    args = parser.parse_args()

    try:
        archive(args.input, args.output, args.level, args.threads)
    except Exception as e:
        print(f"Error: {e}")
