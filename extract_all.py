import re
from collections import Counter

import pdfplumber

from text_cleaning import clean_line_list, collapse_blanks
from fix_subsection_markers import fix_slash_markers

FILES = [
    {
        "src": "raw/Delhi Right to Information Act 2001.pdf",
        "out": "processed/delhi_rti_2001_final.txt",
        "label": "Delhi RTI Act, 2001",
        "duplicate": False,
    },
    {
        "src": "raw/Delhi_HC_Judgement_dated_22.01.2021_-_Sh._"
               "Rakesh_Kumar_Gupta_Erstwhile_CPIO_Union_Bank_of_India__"
               "Ors._Vs._Central_Information_Commision__Anr.pdf",
        "out": "processed/hc_judgment_2021_final.txt",
        "label": "Delhi HC Judgment (22.01.2021)",
        "duplicate": False,
    },
    {
        "src": "raw/CIC_Decision_dated_23.04.2026_on_2nd_Appeal_filed_"
               "by_Prasanta_Kumar_Sahoo_Vs._CPIO_Ministry_of_Labour__"
               "Employment_New_Delhi.pdf",
        "out": "processed/cic_decision_2026_final.txt",
        "label": "CIC Decision (23.04.2026)",
        "duplicate": False,
    },
    {
        "src": "raw/02.The RTI Act, 2005_As on 01022011_English.pdf",
        "out": "processed/rti_act_2005_v2_final.txt",
        "label": "RTI Act, 2005 v2",
        "duplicate": True,
    },
]

PAREN_RE = re.compile(r"\([^()]*\)")
# the digit-1->slash corruption: parens holding only slash-tally marks
# (optional leading space, optional trailing '0'):  (/), (//), ( /), (/0), ( /0)
CORRUPT_RE = re.compile(r"\(\s*/+0?\s*\)")


def slash_shapes(text):
    """Return Counter of corrupted slash-token shapes (the digit->slash bug)."""
    return Counter(CORRUPT_RE.findall(text))


def legit_slash_examples(text, limit=5):
    """Return up to 'limit' paren tokens with slashes that are NOT the bug."""
    out = []
    for tok in PAREN_RE.findall(text):
        if "/" in tok and not CORRUPT_RE.fullmatch(tok):
            out.append(tok.replace("\n", " "))
            if len(out) >= limit:
                break
    return out


def extract_pdf(path):
    """Extract full text via pdfplumber. Returns (text, n_pages, n_pages_with_text)."""
    pages_text = []
    n_pages = 0
    n_with_text = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception as exc:                      # skip bad pages
                t = ""
                print(f"      [warn] page failed: {exc}")
            if t.strip():
                n_with_text += 1
            pages_text.append(t)
    return "\n".join(pages_text), n_pages, n_with_text


def process_file(cfg):
    src, out = cfg["src"], cfg["out"]

    raw_text, n_pages, n_with_text = extract_pdf(src)
    raw_chars = len(raw_text)

    # scanned / image-only PDFs have no usable text layer
    scanned = n_with_text == 0 or raw_chars < 300
    if scanned:
        with open(out, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"---+ {cfg['label']} {'(DUPLICATE/ALTERNATE RTI SOURCE)' if cfg['duplicate'] else ''}")
        print(f"    source : {src}")
        print(f"    pages  : {n_pages} total, {n_with_text} with extractable text")
        print(f"    !! SCANNED / IMAGE-ONLY PDF -- no text layer ({raw_chars} chars)")
        print(f"    !! requires OCR (e.g. pytesseract/Tesseract) instead of pdfplumber")
        print(f"    saved  : {out}")
        return {
            "out": out,
            "label": cfg["label"],
            "duplicate": cfg["duplicate"],
            "raw_chars": raw_chars,
            "final_chars": raw_chars,
            "removed": None,
            "slash_found": False,
            "slash_fixed": 0,
            "slash_shapes": {},
            "note": "SCANNED/OCR REQUIRED",
        }

    # 2) remove noise lines
    kept, removed = clean_line_list(raw_text.splitlines())
    kept = collapse_blanks(kept)
    cleaned_text = "\n".join(kept).rstrip() + "\n"

    # 3) check slash corruption and fix if present
    shapes_before = slash_shapes(cleaned_text)
    n_slash_tokens = sum(shapes_before.values())
    n_fixed = 0
    if n_slash_tokens:
        fixed_text, fixes = fix_slash_markers(cleaned_text)
        n_fixed = len(fixes)
        remaining = slash_shapes(fixed_text)
        final_text = fixed_text
    else:
        final_text = cleaned_text
        remaining = Counter()

    with open(out, "w", encoding="utf-8") as f:
        f.write(final_text)

    final_chars = len(final_text)

    # report details
    print(f"---+ {cfg['label']} {'(DUPLICATE/ALTERNATE RTI SOURCE)' if cfg['duplicate'] else ''}")
    print(f"    source : {src}")
    print(f"    pages  : {n_pages} total, {n_with_text} with extractable text")
    print(f"    chars  : raw={raw_chars}  ->  final={final_chars}")
    print(f"    noise lines removed: {removed}")
    if n_slash_tokens:
        print(f"    slash-corruption: FOUND ({n_slash_tokens}) shapes {dict(shapes_before)}")
        print(f"    slash tokens fixed: {n_fixed}")
        if remaining:
            print(f"    !! unhandled corruption remaining: {dict(remaining)}")
    else:
        print(f"    slash-corruption: none found - not applicable to this file")
        legit = legit_slash_examples(cleaned_text)
        if legit:
            print(f"    (other '/' appeared only in legitimate refs e.g. {legit})")

    return {
        "out": out,
        "label": cfg["label"],
        "duplicate": cfg["duplicate"],
        "raw_chars": raw_chars,
        "final_chars": final_chars,
        "removed": removed,
        "slash_found": n_slash_tokens > 0,
        "slash_fixed": n_fixed,
        "slash_shapes": dict(shapes_before),
        "note": "",
    }


def main():
    results = []
    for cfg in FILES:
        print("=" * 90)
        try:
            results.append(process_file(cfg))
        except Exception as exc:
            print(f"!!! FAILED for {cfg['out']}: {exc}")
            results.append({
                "out": cfg["out"],
                "label": cfg["label"],
                "duplicate": cfg["duplicate"],
                "raw_chars": 0,
                "final_chars": 0,
                "removed": 0,
                "slash_found": False,
                "slash_fixed": 0,
                "slash_shapes": {},
                "error": str(exc),
            })

    print()
    print("=" * 100)
    print("SUMMARY TABLE")
    print(f"{'output file':<38} {'raw_chars':>10} {'final_chars':>12} {'noise_rm':>9} {'slash_bug':>16}  note")
    print("-" * 100)
    for r in results:
        if r.get("error"):
            print(f"{r['out']:<38} {r['raw_chars']:>10} {r['final_chars']:>12} {'-':>9} {'ERROR':>16}  ")
            continue
        rm = r["removed"] if r["removed"] is not None else "-"
        slash = f"YES ({r['slash_fixed']} fixed)" if r["slash_found"] else "no"
        note = []
        if r.get("duplicate"):
            note.append("DUPLICATE RTI SRC")
        if r.get("note"):
            note.append(r["note"])
        print(f"{r['out']:<38} {r['raw_chars']:>10} {r['final_chars']:>12} {rm:>9} {slash:>16}  {' | '.join(note) if note else ''}")


if __name__ == "__main__":
    main()