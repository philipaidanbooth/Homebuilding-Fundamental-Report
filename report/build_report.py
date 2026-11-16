#!/usr/bin/env python3
"""Assemble the full homebuilding fundamentals report.

Order: (executive summary, if present) -> primer -> driver 1-4 bodies -> section 8
-> Data & Methodology (methodology text, then the Supporting analysis divider page) ->
supporting analysis (driver 1, 2, 4 exhibit pages) -> Sources.

Driver PDFs are split at the page where their appendix starts, found by text search,
so nothing in the driver Rmds has to change. Supporting-analysis pages are then packed:
each page's content (below the running header) is cropped and stacked onto as few pages
as it fits, so a short appendix no longer leaves most of a page blank. Packing keeps the
pages as vector graphics (xelatex + graphicx crop); if xelatex is missing it is skipped. Requires poppler (pdfinfo, pdftotext)
and qpdf, both standard. Run from anywhere:  python3 build_report.py
Output: report/homebuilding_fundamentals_report_final.pdf
"""
import os
import re
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE) == "report":
    BASE = os.path.dirname(BASE)

OUT = os.path.join(BASE, "report", "homebuilding_fundamentals_report_final.pdf")

# (path, marker text where the appendix starts, or None to include the whole file)
SECTIONS = [
    ("executive_summary/executive_summary.pdf", None),  # optional, skipped until it exists
    ("00_primer/primer_section.pdf", None),
    ("01_affordability/driver1_section.pdf", "Supporting analysis: Driver 1"),
    ("02_inventory/driver2_section.pdf", "Supporting analysis: Driver 2"),
    ("03_costs/driver3_section.pdf", None),
    ("04_margins/driver4_section.pdf", "Supporting analysis: Driver 4"),
    ("05_scenarios_signposts/section8_section.pdf", None),
    ("07_methodology/methodology_section_v2.pdf", None),
]
SOURCES = "06_appendix_sources/sources_section_v2.pdf"
OPTIONAL = {"executive_summary/executive_summary.pdf", "07_methodology/methodology_section_v2.pdf"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(cmd), r.stderr.strip()))
    return r.stdout


def n_pages(pdf):
    m = re.search(r"Pages:\s+(\d+)", run(["pdfinfo", pdf]))
    return int(m.group(1))


def find_marker(pdf, text):
    for p in range(1, n_pages(pdf) + 1):
        t = run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"])
        if text in t:
            return p
    sys.exit("FAILED: marker %r not found in %s" % (text, pdf))


def number_pages(pdf):
    """Stamp continuous page numbers bottom right. A white strip first covers each
    section's own footer, so sections can be merged whether or not they were re-knit."""
    import io
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    reader = PdfReader(pdf)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages, start=1):
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, 0, w, 38, stroke=0, fill=1)          # bottom margin is 0.6in (43pt)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.setFont("Helvetica", 8)
        c.drawRightString(w - 0.6 * 72, 24, str(i))
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    tmp = pdf + ".tmp"
    with open(tmp, "wb") as fh:
        writer.write(fh)
    os.replace(tmp, pdf)


HEADER_BOTTOM = 22    # pt from top: running header and its rule end above this line
BODY_BOTTOM = 745     # pt from top: lowest point content may reach (0.65in bottom margin)
BLOCK_GAP = 16        # pt of space between stacked blocks


def content_bottom(pdf, page):
    """Lowest inked row (pt from top) on a page, from a 72-dpi render."""
    import io
    import numpy as np
    from PIL import Image
    png = subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "72",
                          "-gray", "-png", pdf], capture_output=True, check=True).stdout
    a = np.array(Image.open(io.BytesIO(png)).convert("L"))
    rows = np.where((a[:BODY_BOTTOM] < 245).any(axis=1))[0]
    return int(rows.max()) + 4


def pack_appendix(appendices, workdir):
    """Stack appendix page contents onto as few pages as they fit. Returns a pdf path,
    or None if xelatex is unavailable (caller falls back to whole pages)."""
    import shutil
    if not shutil.which("xelatex"):
        print("xelatex not found: appendix pages left unpacked")
        return None
    blocks = []  # (local pdf name, page, top, bottom)
    for k, (pdf, rng) in enumerate(appendices):
        local = "src%d.pdf" % k
        shutil.copy(pdf, os.path.join(workdir, local))
        a, b = map(int, rng.split("-"))
        for p in range(a, b + 1):
            blocks.append((local, p, HEADER_BOTTOM, content_bottom(pdf, p)))
    pages, cur, used = [], [], HEADER_BOTTOM
    for blk in blocks:
        h = blk[3] - blk[2]
        need = h + (BLOCK_GAP if cur else 0)
        if cur and used + need > BODY_BOTTOM:
            pages.append(cur)
            cur, used = [], HEADER_BOTTOM
            need = h
        cur.append(blk)
        used += need
    if cur:
        pages.append(cur)
    tex = [r"\documentclass{article}",
           r"\usepackage[paperwidth=612bp,paperheight=792bp,margin=0pt]{geometry}",
           r"\usepackage{graphicx}", r"\pagestyle{empty}",
           r"\setlength{\parindent}{0pt}\setlength{\parskip}{0pt}",
           r"\begin{document}\offinterlineskip"]
    for pg in pages:
        src, p = pg[0][0], pg[0][1]
        # running header, cropped from the first block's own page
        tex.append(r"\includegraphics[page=%d,trim=0bp %dbp 0bp 0bp,clip]{%s}\par"
                   % (p, 792 - HEADER_BOTTOM, src))
        for i, (src, p, top, bot) in enumerate(pg):
            if i:
                tex.append(r"\vspace{%dbp}" % BLOCK_GAP)
            tex.append(r"\includegraphics[page=%d,trim=0bp %dbp 0bp %dbp,clip]{%s}\par"
                       % (p, 792 - bot, top, src))
        tex.append(r"\newpage")
    tex.append(r"\end{document}")
    with open(os.path.join(workdir, "appendix.tex"), "w") as fh:
        fh.write("\n".join(tex))
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "appendix.tex"],
                       cwd=workdir, capture_output=True, text=True)
    out = os.path.join(workdir, "appendix.pdf")
    if r.returncode != 0 or not os.path.exists(out):
        sys.exit("FAILED: appendix packing (see %s/appendix.log)" % workdir)
    print("supporting analysis: %d pages packed onto %d" % (len(blocks), len(pages)))
    return out


def main():
    bodies, appendices = [], []  # (pdf, page_range)
    for rel, marker in SECTIONS:
        pdf = os.path.join(BASE, rel)
        if not os.path.exists(pdf):
            if rel in OPTIONAL:
                print("skipping (not built yet): %s" % rel)
                continue
            sys.exit("missing: %s%s" % (rel, " - knit methodology_section_v2.Rmd first"
                                        if "methodology" in rel else ""))
        total = n_pages(pdf)
        if marker is None:
            bodies.append((pdf, "1-%d" % total))
            print("%-45s whole (%d pages)" % (rel, total))
        else:
            split = find_marker(pdf, marker)
            if split <= 1:
                sys.exit("FAILED: %s appendix starts on page 1?" % rel)
            bodies.append((pdf, "1-%d" % (split - 1)))
            appendices.append((pdf, "%d-%d" % (split, total)))
            print("%-45s body 1-%d, appendix %d-%d" % (rel, split - 1, split, total))

    src = os.path.join(BASE, SOURCES)
    if not os.path.exists(src):
        sys.exit("missing: %s" % SOURCES)
    tail = [(src, "1-%d" % n_pages(src))]

    with tempfile.TemporaryDirectory() as tmp:
        packed = pack_appendix(appendices, tmp) if appendices else None
        middle = [(packed, "1-%d" % n_pages(packed))] if packed else appendices
        args = ["qpdf", "--empty", "--pages"]
        for pdf, rng in bodies + middle + tail:
            args += [pdf, rng]
        args += ["--", OUT]
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        run(args)
    number_pages(OUT)
    print("\nwrote %s (%d pages)" % (os.path.relpath(OUT, BASE), n_pages(OUT)))


if __name__ == "__main__":
    main()
