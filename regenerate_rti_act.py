import re

import pdfplumber

from text_cleaning import clean_line_list, collapse_blanks
from fix_subsection_markers import fix_slash_markers

SRC = "raw/RTI-Act_English.pdf"
OUT = "processed/rti_act_2005_final.txt"

# standalone legal-marker lines like "(a)", "(b)", "(1)", "(i)" that live in a
# separate left column and are dropped by is_noise() as tiny fragments
MARKER_RE = re.compile(r"^\(\s*(?:[0-9]{1,3}|[a-z]{1,3}|[ivxlcdm]{1,6})\s*\)\s*$", re.I)


def extract(src):
    with pdfplumber.open(src) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def keep_line(line):
    return (not line.strip()) or not _is_noise_but_marker(line)


def _is_noise_but_marker(line):
    if MARKER_RE.match(line.strip()):
        return False          # legal marker column - always keep
    from text_cleaning import is_noise
    return is_noise(line)


def main():
    raw = extract(SRC)
    lines = raw.splitlines()

    kept, removed = [], 0
    for line in lines:
        if keep_line(line):
            kept.append(line.rstrip())
        else:
            removed += 1

    kept = collapse_blanks(kept)
    text = "\n".join(kept).rstrip() + "\n"
    text, fixes = fix_slash_markers(text)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    marker_count = sum(1 for l in kept if MARKER_RE.match(l.strip()))
    print(f"source pages : extracted from {SRC}")
    print(f"noise removed: {removed}")
    print(f"slash fixed  : {len(fixes)}")
    print(f"marker lines : {marker_count} preserved")
    print(f"saved        : {OUT} ({len(text)} chars, {len(kept)} lines)")


if __name__ == "__main__":
    main()