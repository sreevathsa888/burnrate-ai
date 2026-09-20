"""
Capture dashboard screenshots for the README, from YOUR data.

    pip install playwright
    playwright install chromium

Then, in one terminal:
    streamlit run app.py

And in another, from the project root:
    python scripts/capture_screenshots.py

Writes docs/dashboard.png and docs/scenario.png. Pick an interesting venture in
the sidebar before running this -- the highest-risk entries produce a visibly
curved survival plot, which reads far better than a flat one.
"""
import os
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = "docs"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1060},
                            device_scale_factor=2)   # retina-quality for the README
    print(f"[open] {URL}")
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(12000)                     # let Streamlit finish rendering

    page.screenshot(path=f"{OUT}/dashboard.png", full_page=True)
    print(f"[saved] {OUT}/dashboard.png")

    # drag the burn slider so the second shot shows the simulator responding
    try:
        # Streamlit renders the thumb inside .stSlider; grab the track and click
        # part-way along it, which moves the handle without needing the thumb node.
        track = page.locator('div[data-testid="stSlider"]').first
        track.wait_for(state="visible", timeout=15000)
        box = track.bounding_box()
        y = box["y"] + box["height"] * 0.78          # the track sits low in the block
        page.mouse.click(box["x"] + box["width"] * 0.45, y)
        page.wait_for_timeout(6000)
        page.screenshot(path=f"{OUT}/scenario.png", full_page=True)
        print(f"[saved] {OUT}/scenario.png")
    except Exception as exc:
        print(f"[skip] could not drag the slider ({exc}); move it by hand and re-run")

    browser.close()
print("\n[done] reference these from README.md")