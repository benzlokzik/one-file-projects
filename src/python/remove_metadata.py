#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow",
#     "pypdf2",
# ]
# ///
import sys
import os
import mimetypes
import shutil
import random
import string
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter


def random_name(ext):
    return (
        "stripped_"
        + "".join(
            random.choices(
                string.ascii_lowercase + string.digits,
                k=8,
            )
        )
        + ext
    )


def strip_image(input_file, output_file):
    img = Image.open(input_file)
    data = list(img.getdata())
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    clean.save(output_file)


def strip_pdf(input_file, output_file):
    reader = PdfReader(input_file)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({})  # remove all metadata
    with open(output_file, "wb") as f:
        writer.write(f)


def strip_generic(input_file, output_file):
    # fallback: just copy raw bytes (no guarantee metadata removed)
    shutil.copyfile(input_file, output_file)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file_to_remove_metadata>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print("Error: file not found")
        sys.exit(1)

    mime, _ = mimetypes.guess_type(input_file)
    ext = os.path.splitext(input_file)[1] or ".dat"
    output_file = random_name(ext)

    try:
        if mime and mime.startswith("image/"):
            strip_image(input_file, output_file)
            print(f"Image metadata removed → {output_file}")
        elif mime == "application/pdf":
            strip_pdf(input_file, output_file)
            print(f"PDF metadata removed → {output_file}")
        else:
            strip_generic(input_file, output_file)
            print(f"Generic file copied without filename leak → {output_file}")
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
