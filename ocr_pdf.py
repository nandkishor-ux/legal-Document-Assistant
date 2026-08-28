import difflib
import re
from collections import Counter

import pdf2image
import pdfplumber
import pytesseract

from text_cleaning import clean_line_list, collapse_blanks
from fix_subsection_markers import fix_slash_markers

OCR_DPI = 300
MAX_OCR_PAGES = 0  # 0 = all pages; set to a number to limit for testing

TESSERACT_HELP = (
    "Tesseract OCR engine binary was NOT found on this system.\n"
    "It is a separate program (not a Python package); pytesseract is just the wrapper.\n"
    "\n"
    "Install it (one of these):\n"
    "  A) winget:      winget install --id UB-Mannheim.TesseractOCR -e\n"
    "     (installs to C:\\Program Files\\Tesseract-OCR by default)\n"
    "  B) Manual:      download the 64-bit installer from the UB Mannheim build page\n"
    "     https://github.com/UB-Mannheim/tesseract/wiki  and run it.\n"
    "     Tick the checkbox 'Add Tesseract to system PATH' during install.\n"
    "\n"
    "Add Tesseract to PATH (if not done at install):\n"
    "  1. Open Start menu -> search 'environment variables' ->\n"
    "     'Edit the system environment variables'.\n"
    "  2. Environment Variables -> under 'User variables' select 'Path' -> Edit.\n"
    "  3. New -> paste:  C:\\Program Files\\Tesseract-OCR   -> OK -> OK.\n"
    "  4. Open a NEW terminal so PATH picks it up, then verify:\n"
    "       tesseract --version\n"
    "\n"
    "If you do not want to touch PATH, run the script with the path set explicitly:\n"
    "   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n"
)

LEGAL_VOCAB = {
    "information", "commissioner", "commission", "authority", "authorities",
    "government", "public", "officer", "section", "subsection", "appeal",
    "rejection", "grounds", "disclosure", "exemption", "official", "register",
    "request", "charges", "penalty", "penalties", "prescribed", "provided",
    "notification", "representative", "appropriate", "constitution", "article",
    "statutory", "causes", "liability", "shall", "such", "said", "thereto",
    "thereof", "thereunder", "notwithstanding", "competent", "principal",
    "concerned", "reply", "complainant", "applicant", "citizen", "reasonably",
    "liable", "false", "sovereign", "expenditure", "functions", "witnesses",
}

TOKEN_TRIM = ".,;:()\"'`/|-_\u2018\u2019\u201c\u201d\u2013\u2014\u00a0\u2022\u2026\ufffd"

COMMON_WORDS = {
    "government", "legislative", "assembly", "territory", "provision", "republic",
    "governor", "published", "authority", "authorities", "reference", "following",
    "question", "stated", "making", "matters", "official", "gazette",
    "prescribed", "purposes", "general", "exception", "manner", "person",
    "persons", "statement", "information", "possession", "proceedings",
    "reasonable", "cause", "charge", "charges", "provide", "provided",
    "sub-section", "subsection", "constitutional", "constitution", "liable", "false",
    "office", "offices", "clauses", "ground", "grounds", "cases", "case",
    "monthly", "remuneration", "sub-sections", "chapter", "complaint", "complaints",
}

CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")
DIGIT_LETTER_RE = re.compile(r"[lI][0-9]|[0-9][lI]|[oO][0-9]|[0-9][oO]")


def norm_token(tok):
    return tok.strip(TOKEN_TRIM).lower()


def tesseract_available():
    try:
        pytesseract.pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def render_pages(path, dpi=OCR_DPI):
    try:
        images = pdf2image.convert_from_path(path, dpi=dpi, fmt="png")
        return images, "pdf2image/poppler"
    except Exception:
        images = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                images.append(page.to_image(resolution=dpi).original.convert("RGB"))
        return images, "pdfplumber/pypdfium2 (fallback, poppler not found)"


def ocr_pages(images, limit=MAX_OCR_PAGES):
    pages_text = []
    for i, img in enumerate(images, 1):
        if limit and i > limit:
            break
        txt = pytesseract.image_to_string(img)
        pages_text.append(txt)
        print(f"  OCR page {i}/{len(images)}: {len(txt)} chars")
    return pages_text


def is_derived_form(low, base):
    """True if 'low' is a normal English inflection/derivation of 'base'."""
    if low.startswith(base):
        return True
    if low == base + "s" or low == base + "es" or low == base + "ed":
        return True
    if low == base + "d" or low == base + "ing" or low == base + "ly":
        return True
    if low.endswith("s") and low[:-1] == base:
        return True
    return False


def flag_suspicious(line):
    """Return non-empty list of (token, reason) if line hints at OCR garble."""
    tokens = line.split()
    reasons = []
    for tok in tokens:
        low = norm_token(tok)
        if len(low) < 5:
            continue
        if low.isdigit():
            continue
        if low in LEGAL_VOCAB or low in COMMON_WORDS:
            continue
        run = max((len(m) for m in CONSONANT_RUN_RE.findall(low)), default=0)
        if run >= 5 and not any(v in low for v in ("ment", "ction", "sion", "tion")):
            reasons.append((tok, "long consecutive-consonant run"))
            continue
        if DIGIT_LETTER_RE.search(low):
            reasons.append((tok, "digit/letter confusion (l/I/o for 1/0)"))
            continue
        close = difflib.get_close_matches(low, LEGAL_VOCAB, n=1, cutoff=0.9)
        if close and not is_derived_form(low, close[0]):
            reasons.append((tok, f"possible OCR garble of '{close[0]}'"))
    return reasons


def process(src_pdf, out_txt, label="", limit=MAX_OCR_PAGES):
    """Run the full OCR pipeline on src_pdf and save cleaned text to out_txt.

    Returns (out_path, cleaned_text, flags) where flags is a list of
    (line_no, line, token, reason) tuples for manual review.
    """
    print(f"=== OCR: {label}")
    if not tesseract_available():
        print(TESSERACT_HELP)
        return None, "", []

    print(f"Rendering {src_pdf}")
    images, engine = render_pages(src_pdf)
    print(f"  rendered {len(images)} pages via {engine}")
    raw_pages = ocr_pages(images, limit=limit)
    raw_text = "\n".join(raw_pages)

    kept, n_removed = clean_line_list(raw_text.splitlines())
    kept = collapse_blanks(kept)
    cleaned_text = "\n".join(kept).rstrip() + "\n"

    slash_shapes = re.findall(r"\(\s*/+0?\s*\)", cleaned_text)
    if slash_shapes:
        fixed_text, fixes = fix_slash_markers(cleaned_text)
        n_fixed = len(fixes)
        cleaned_text = fixed_text
        print(f"\nSlash-marker corruption: found {len(slash_shapes)} shapes {dict(Counter(slash_shapes))}")
        print(f"Slash-marker corruption fixed: {n_fixed}")
    else:
        n_fixed = 0

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"\nOCR raw characters : {len(raw_text)}")
    print(f"Noise lines removed: {n_removed}")
    print(f"Cleaned saved      : {out_txt} ({len(cleaned_text)} chars)")

    flags = []
    for i, ln in enumerate(kept, 1):
        for tok, reason in flag_suspicious(ln):
            flags.append((i, ln, tok, reason))
    return out_txt, cleaned_text, flags


def print_report(cleaned_text, flags, preview=1500):
    print("=" * 80)
    print(f"First {preview} characters:")
    print("=" * 80)
    print(cleaned_text[:preview])
    print("=" * 80)
    if flags:
        print(f"Suspicious lines flagged for manual review: {len(flags)}")
        for i, ln, tok, reason in flags[:40]:
            print(f"  L{i:4d} | {ln[:90]}")
            print(f"        ^ '{tok}': {reason}")
    else:
        print("No suspicious lines flagged - OCR output looks clean.")