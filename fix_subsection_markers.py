import re

INPUT = "processed/rti_act_2005_clean.txt"
OUTPUT = "processed/rti_act_2005_final.txt"


def fix_slash_markers(text):
    """Convert slash-corrupted parenthetical markers back to their numbers.

    The source PDF's body font maps the digit '1' to '/', so "(1)" extract as
    "(/)". Two shapes appear:

      * tally style  (/) (//) (///) ...  ->  (1) (2) (3) ...
      * '1'+digit    (/0) and ( /0)      ->  (10)   (the trailing digit is intact)
    """
    fixes = []

    # 1) pure slash runs: count slashes -> numeral
    def tally(m):
        n = len(m.group(1))
        fixes.append((m.group(0), f"({n})"))
        return f"({n})"

    text = re.sub(r"\((/+)\)", tally, text)

    # 2) variants with spacing or a surviving digit: "( /)" -> (1), "(/0)"/"( /0)" -> (10)
    def variant(m):
        tok = m.group(0)
        n = 1 if "0" not in tok else 10
        fixes.append((tok, f"({n})"))
        return f"({n})"

    text = re.sub(r"\(\s*/0?\)", variant, text)

    return text, fixes


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        text = f.read()

    fixed, fixes = fix_slash_markers(text)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(fixed)

    print(f"Total slash-marker corrections: {len(fixes)}")
    print(f"Saved corrected text to: {OUTPUT}")
    print("-" * 60)
    print("Example before/after conversions (first 10, in document order):")
    print(f"  {'before':>10} -> {'after'}")
    for before, after in fixes[:10]:
        print(f"  {before!r:>10} -> {after!r}")

    special = [f for f in fixes if f[0] != "(/)"]
    if special:
        print(f"  Special variants ({len(special)} non-'(/)' shapes):")
        for before, after in special:
            print(f"    {before!r} -> {after!r}")

    print("-" * 60)
    print("Audit of the corrected file:")
    print(f"  remaining '(/' anywhere: {len(re.findall(r'\(/', fixed))}")
    for n in range(1, 11):
        pat = rf"(?<!\d)\({n}\)"
        print(f"  ({n}) x{len(re.findall(pat, fixed))}")
    print(f"  lettered (a)-(z): {len(re.findall(r'\([a-z]\)', fixed))}")
    headers = re.findall(r"(?m)^\s*\d{1,2}\.\s", fixed)
    print(f"  main-section headers 'N. Title': {len(headers)}")
    print(f"  chars: {len(text)} -> {len(fixed)}")


if __name__ == "__main__":
    main()