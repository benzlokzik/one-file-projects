# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "pytest-playwright",
# ]
# ///

"""
Save every element matching a selector (default: div.marimo-cell) as PNGs.

Prerequisites:
  uvx --with pytest-playwright playwright install

Usage:
  uv run html_tag_to_png.py ./report.html
  uv run html_tag_to_png.py "https://example.com/page" -o out/ -s "div.marimo-cell" --scale 2 --transparent
"""

from pathlib import Path
from urllib.parse import urlparse
import math
import click
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def to_url(s: str) -> str:
    p = urlparse(s)
    if p.scheme in ("http", "https"):
        return s
    return Path(s).resolve().as_uri()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("page", metavar="PAGE")
@click.option("-o", "--outdir", default="marimo_cells_png", show_default=True, help="Directory for PNGs.")
@click.option("-s", "--selector", default="div.marimo-cell", show_default=True, help="CSS selector to capture.")
@click.option("--scale", type=float, default=1.5, show_default=True, help="Device scale factor for sharp PNGs.")
@click.option(
    "--transparent/--no-transparent",
    default=False,
    show_default=True,
    help="Transparent background PNGs."
)
def main(page: str, outdir: str, selector: str, scale: float, transparent: bool):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    target_url = to_url(page)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        # viewport=None removes the fixed 1280x720 constraint:
        # Playwright will use the page's own size and scroll as needed.
        context = browser.new_context(viewport=None, device_scale_factor=scale)
        p = context.new_page()

        p.goto(target_url, wait_until="networkidle")

        # Optional transparent background for all screenshots
        if transparent:
            p.evaluate(
                """
                () => {
                  const st = document.createElement('style');
                  st.setAttribute('data-marimo-capture', '1');
                  st.textContent = `html, body { background: transparent !important; }`;
                  document.documentElement.appendChild(st);
                }
                """
            )

        cells = p.locator(selector)
        count = cells.count()
        print(f"Found {count} element(s) matching selector: {selector}")
        if count == 0:
            browser.close()
            return

        saved = 0
        for i in range(count):
            el = cells.nth(i)

            # Skip disconnected nodes early
            try:
                if not el.evaluate("n => !!n.isConnected"):
                    print(f"Skipping #{i+1:03d}: node is not connected.")
                    continue
            except Exception:
                # If evaluation fails, continue best-effort
                pass

            # Ensure the element is brought into view (handles nested scroll containers)
            try:
                el.scroll_into_view_if_needed(timeout=5000)
            except PWTimeout:
                # Not fatal; we can still try to capture via clip fallback
                pass
            except Exception:
                pass

            # --- First attempt: element.screenshot (the most robust, no manual clip math) ---
            try:
                filename = outdir_path / f"marimo-cell-{i+1:03d}.png"
                el.screenshot(path=str(filename), omit_background=transparent)
                print(f"Saved: {filename}")
                saved += 1
                continue
            except Exception as e:
                # Fall through to clip-based approach
                # print(f"element.screenshot failed on #{i+1:03d}: {e}")
                pass

            # --- Fallback: page.screenshot with clip computed from getBoundingClientRect ---
            try:
                # Measure element rect in *viewport* coordinates
                vr = el.evaluate(
                    """(node) => {
                        const r = node.getBoundingClientRect();
                        return { x: r.left, y: r.top, width: r.width, height: r.height };
                    }"""
                )
                vx = float(vr.get("x", 0) or 0.0)
                vy = float(vr.get("y", 0) or 0.0)
                vw = float(vr.get("width", 0) or 0.0)
                vh = float(vr.get("height", 0) or 0.0)

                if vw < 1 or vh < 1:
                    print(f"Skipping #{i+1:03d}: zero-size after scroll.")
                    continue

                # If element is partially off-screen (due to sticky headers/sidebars),
                # intersect with current viewport to avoid out-of-bounds errors.
                vp = p.viewport_size
                # When viewport=None, Playwright picks content size; still returns a dict.
                if not vp:
                    # As a safety net, pick a reasonable default; but this should rarely happen.
                    vp = {"width": 1920, "height": 1080}

                vp_w = float(vp["width"])
                vp_h = float(vp["height"])

                # Intersect rect with viewport
                clip_x = max(0.0, vx)
                clip_y = max(0.0, vy)
                clip_w = min(vp_w - clip_x, vw)
                clip_h = min(vp_h - clip_y, vh)

                # Slightly expand & round to integers for stability
                EPS = 0.5
                clip = {
                    "x": int(math.floor(clip_x - EPS)) if clip_x > 0 else 0,
                    "y": int(math.floor(clip_y - EPS)) if clip_y > 0 else 0,
                    "width": int(math.ceil(clip_w + 2 * EPS)),
                    "height": int(math.ceil(clip_h + 2 * EPS)),
                }

                if clip["width"] < 1 or clip["height"] < 1:
                    print(f"Skipping #{i+1:03d}: clip out of bounds.")
                    continue

                filename = outdir_path / f"marimo-cell-{i+1:03d}.png"
                p.screenshot(path=str(filename), clip=clip, omit_background=transparent)
                print(f"Saved (fallback): {filename}")
                saved += 1
            except Exception:
                print(f"Skipping #{i+1:03d}: clip failed.")
                continue

        print(f"Done. Saved {saved} PNG file(s) to: {outdir_path.resolve()}")
        browser.close()


if __name__ == "__main__":
    main()
