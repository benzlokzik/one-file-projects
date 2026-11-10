# /// script
# requires-python = ">=3.9"
# dependencies = ["python-dotenv", "mutagen"]
# ///
"""List (and optionally download) the songs on a Telegram user's profile via Bot API.

Usage:
  uv run ./tg_profile_songs.py USER_ID [--json | --md] [--no-numbering] [--no-duration] [--no-header] [--download DIR]
  BOT_TOKEN is read from the environment or a .env file.
"""
import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import mutagen
from dotenv import find_dotenv, load_dotenv

API = "https://api.telegram.org"


def call(token, method, **params):
    url = f"{API}/bot{token}/{method}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        data = json.load(e)
    if not data.get("ok"):
        sys.exit(f"{method} failed: {data.get('error_code')} {data.get('description')}")
    return data["result"]


def fetch_all(token, user_id):
    audios, offset = [], 0
    while True:
        res = call(token, "getUserProfileAudios", user_id=user_id, offset=offset, limit=100)
        batch = res.get("audios", [])
        audios += batch
        offset += len(batch)
        if not batch or offset >= res.get("total_count", 0):
            return audios


def restore_cut_tags(token, audio):
    """Telegram cuts long titles/performers to 63 chars + '…'; read the full ones from the file's tags."""
    cut = [k for k in ("title", "performer") if str(audio.get(k, "")).endswith("…")]
    if not cut:
        return
    f = call(token, "getFile", file_id=audio["file_id"])
    if not f.get("file_path"):
        print(f"  can't restore (file over 20 MB): {audio.get('title')}", file=sys.stderr)
        return
    try:
        with urllib.request.urlopen(f"{API}/file/bot{token}/{f['file_path']}", timeout=120) as r:
            tags = mutagen.File(io.BytesIO(r.read()), easy=True)
    except (OSError, mutagen.MutagenError) as e:
        print(f"  can't read tags ({e}): {audio.get('title')}", file=sys.stderr)
        return
    for k, tag in (("title", "title"), ("performer", "artist")):
        full = ", ".join((tags or {}).get(tag, []))
        if k in cut and len(full) > len(audio[k]) - 1:
            audio[k] = full
        elif k in cut:
            print(f"  no full {k} in file tags: {audio[k]}", file=sys.stderr)


def download(token, audio, out_dir):
    f = call(token, "getFile", file_id=audio["file_id"])
    path = f.get("file_path")
    if not path:  # files over 20 MB have no file_path
        print(f"  skip (too big for Bot API): {audio.get('title')}", file=sys.stderr)
        return
    ext = os.path.splitext(path)[1] or ".mp3"
    performer, title = audio.get("performer"), audio.get("title")
    if performer or title:
        stem = " - ".join(s for s in (performer, title) if s)
    else:
        stem = os.path.splitext(audio.get("file_name") or audio["file_unique_id"])[0]
    stem = "".join(c for c in stem if c not in '/\\:*?"<>|').strip()
    # filesystems cap a name at 255 bytes; only trim if it would not fit
    while len((stem + ext).encode()) > 255:
        stem = stem[:-1]
    name = stem + ext
    dest = os.path.join(out_dir, name)
    urllib.request.urlretrieve(f"{API}/file/bot{token}/{path}", dest)
    print(f"  saved {dest}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("user_id", type=int)
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="print raw Audio objects as JSON")
    fmt.add_argument("--md", action="store_true", help="print as a Markdown table")
    p.add_argument("--no-numbering", action="store_true", help="omit the track number")
    p.add_argument("--no-duration", action="store_true", help="omit the duration")
    p.add_argument("--no-header", action="store_true", help="omit the Markdown table header")
    p.add_argument("--download", metavar="DIR", help="download audio files into DIR")
    a = p.parse_args()

    load_dotenv(find_dotenv(usecwd=True))  # .env in current dir wins
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    token = os.environ.get("BOT_TOKEN") or sys.exit("set BOT_TOKEN in env or .env")
    audios = fetch_all(token, a.user_id)
    for x in audios:
        restore_cut_tags(token, x)

    if a.json:
        print(json.dumps(audios, ensure_ascii=False, indent=2))
    elif a.md:
        cell = lambda s: str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
        cols = [("#", "---:")] * (not a.no_numbering) + [("Performer", "---"), ("Title", "---")] \
            + [("Duration", "---:")] * (not a.no_duration)
        if not a.no_header:
            print("| " + " | ".join(c[0] for c in cols) + " |")
            print("|" + "|".join(c[1] for c in cols) + "|")
        for i, x in enumerate(audios, 1):
            d = x.get("duration", 0)
            row = [str(i)] * (not a.no_numbering) + [cell(x.get("performer", "")),
                   cell(x.get("title", x.get("file_name", "")))] + [f"{d // 60}:{d % 60:02}"] * (not a.no_duration)
            print("| " + " | ".join(row) + " |")
    else:
        for i, x in enumerate(audios, 1):
            d = x.get("duration", 0)
            line = f"{x.get('performer', '?')} — {x.get('title', x.get('file_name', '?'))}"
            if not a.no_numbering:
                line = f"{i:3}. {line}"
            if not a.no_duration:
                line += f" ({d // 60}:{d % 60:02})"
            print(line)
        print(f"total: {len(audios)}", file=sys.stderr)

    if a.download:
        os.makedirs(a.download, exist_ok=True)
        for x in audios:
            download(token, x, a.download)


if __name__ == "__main__":
    main()
