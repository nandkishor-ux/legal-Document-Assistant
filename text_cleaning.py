import re

WORD_RE = re.compile(r"[A-Za-z]+")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
ORD_RE = re.compile(r"\d+(?:st|nd|rd|th)", re.IGNORECASE)

VOWELS = set("aeiouy")
SHORT_WORDS = {
    "ad", "am", "an", "as", "at", "be", "by", "do", "er", "go", "he", "hi",
    "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on", "or",
    "ox", "so", "to", "up", "us", "we", "yo",
}
ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}


def plausible(word):
    wl = word.lower()
    if len(wl) == 1:
        return wl in "ai"
    if len(wl) == 2:
        return wl in SHORT_WORDS
    return any(c in VOWELS for c in wl)


def shape_ok(word):
    """word looks normally cased: lowercase, ALL-CAPS, or Title case."""
    if len(word) == 1:
        return word.isalpha()
    return word.islower() or word.isupper() or word == word.title()


def is_noise(line):
    """Return True if the line is garbled Hindi-glyph noise / extraction junk."""
    if not line.strip():
        return False
    words = WORD_RE.findall(line)
    alnums = ALNUM_RE.findall(line)
    letters = sum(len(w) for w in words)
    digits = sum(c.isdigit() for c in line)
    alnum_len = letters + digits
    best_token = max(len(t) for t in alnums) if alnums else 0

    if alnum_len == 0:
        return True            # pure punctuation / replacement chars
    if not words:
        return True            # digits only: '2011', '21.', '(2)', page numbers
    if best_token < 3:
        return True            # tiny line-split fragments: 'to', 'be;', 'W', '(a)'

    n = len(words)
    plau = sum(plausible(w) for w in words)
    ordinals = ORD_RE.findall(line)        # '12th', '1st' - valid English pieces
    n += len(ordinals)
    plau += len(ordinals)
    frac = plau / n if n else 0.0

    normal = sum(shape_ok(w) for w in words)
    normal_frac = normal / len(words)

    # structural ALL-CAPS headers: "CHAPTER I", "GOVERNMENT OF INDIA", "PREFACE"
    all_caps_struct = (
        digits == 0
        and n >= 2
        and all(w.isupper() or w in ROMAN for w in words)
        and frac >= 0.4
    )
    if all_caps_struct:
        return False

    # 3+ letter token with a mid-word capital ('riTon', 'sictarDcf')
    # is a hallmark of Hindi-glyph-to-Latin mapping -- but keep ALL-CAPS headers
    if any(len(w) >= 3 and not shape_ok(w) for w in words):
        if not all_caps_struct and len(words) >= 2:
            return True

    if normal_frac < 0.75 and len(words) >= 2:
        return True            # camelcase gibberish: 'TRAttli itUti'
    if frac < 0.55:
        return True            # too many non-word tokens
    if alnum_len:
        digit_ratio = digits / alnum_len
        if digit_ratio > 0.30 and (plau < 2 or letters / len(line) < 0.25):
            return True        # digit-dense junk unless it's real text w/ numbers
    return False


def clean_line_list(lines):
    """Filter lines with is_noise(); return (kept_lines, removed_count)."""
    kept = []
    removed = 0
    for line in lines:
        if is_noise(line):
            removed += 1
        else:
            kept.append(line.rstrip())
    return kept, removed


def collapse_blanks(lines):
    """Collapse runs of blank lines into a single blank line."""
    final = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run == 1:
                final.append("")
        else:
            blank_run = 0
            final.append(line)
    return final