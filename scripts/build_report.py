"""Build the PDF report from report/REPORT.md (English) and report/RAPOR.md (Turkish).

    python scripts/build_report.py            # both
    python scripts/build_report.py REPORT     # one

Markdown -> HTML (markdown-it-py, installed with Jupyter) -> PDF with a headless Chromium browser
(Microsoft Edge or Google Chrome). Figures are referenced from ../results/, so run it after the experiments.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.33; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }
p { margin: 3pt 0; }
li { margin: 1.5pt 0; }
ul, ol { margin: 3pt 0; padding-left: 16pt; }
table { border-collapse: collapse; margin: 5pt 0; font-size: 8.6pt; break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2pt 5pt; vertical-align: top; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
em { color: #333; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #e0e0e0; padding: 5pt 7pt; font-size: 7.9pt; line-height: 1.25;
      break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 4pt 0; padding: 2pt 8pt; border-left: 3px solid #bbb; color: #333; }
.pagebreak { break-before: page; }
"""


def find_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("no Edge/Chrome found for printing to PDF")


def to_html(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)          # authoring notes are not part of the report
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(text)
    title = text.splitlines()[0].lstrip("# ").strip()
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html_path: Path) -> Path:
    pdf = html_path.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    # a separate profile, otherwise a browser that is already running takes over the call and nothing is printed
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
                        "--no-pdf-header-footer", "--print-to-pdf-no-header", f"--print-to-pdf={pdf}",
                        html_path.as_uri()],
                       check=True, capture_output=True, timeout=180)
    if not pdf.exists():
        raise SystemExit(f"the browser did not write {pdf}")
    return pdf


def page_count(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


def main_body_pages(md_path: Path) -> int:
    """Pages before the appendix (the 10-15 page limit applies to the main body)."""
    text = md_path.read_text(encoding="utf-8")
    cut = re.search(r'(<div class="pagebreak"></div>\s*)?^## (Appendix|Ek)\b', text, flags=re.M)
    if not cut:
        return 0
    tmp = md_path.with_name(f"_{md_path.stem}_main.md")
    tmp.write_text(text[:cut.start()], encoding="utf-8")
    html = to_html(tmp)
    pdf = to_pdf(html)
    n = page_count(pdf)
    for p in (tmp, html, pdf):
        p.unlink()
    return n


def main():
    names = sys.argv[1:] or ["REPORT", "RAPOR"]
    for name in names:
        html = to_html(REPORT_DIR / f"{name}.md")
        pdf = to_pdf(html)
        html.unlink()
        print(f"{pdf.relative_to(ROOT)}: {page_count(pdf)} pages "
              f"(main body {main_body_pages(REPORT_DIR / f'{name}.md')} + appendix)")


if __name__ == "__main__":
    main()
