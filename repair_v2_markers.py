import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REF = "processed/rti_act_2005_final.txt"
V2 = "processed/rti_act_2005_v2_ocr_final.txt"
OUT = "processed/rti_act_2005_v2_repaired_final.txt"

SEC_RE = re.compile(r"^\s*(\d{1,2})\s*[.,;:)\s]+\s*(?=[A-Z])")
SLOT_RE = re.compile(r"^\(([0-9]{1,2}|[a-z]|[ivxlcdm]{2,6})\)(.*)$")
SCHED_RE = re.compile(r"^\(?\s*(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH)\s+SCHEDULE\b", re.I)
TITLE = "THE RIGHT TO INFORMATION ACT, 2005"


def toks(s):
    return re.findall(r"[a-z]{4,}", s.lower())


def tokens(text):
    return toks(text)


def body_start(text):
    idxs = [i for i, l in enumerate(text.splitlines()) if l.strip() == TITLE]
    return (idxs[-1] + 1) if idxs else 0


def ref_sections(text):
    """section number -> ordered list of clause letters and their text anchors."""
    start = body_start(text)
    secs, cur = {}, None
    for raw in text.splitlines()[start:]:
        line = raw.strip()
        if not line or SCHED_RE.match(line):
            continue
        m = SEC_RE.match(line)
        if m:
            cur = {"num": m.group(1), "clauses": [], "inline1": False}
            secs.setdefault(cur["num"], cur)
            if "(" in line[line.find(m.group(1)) + len(m.group(1)):]:
                cur["inline1"] = bool(re.search(r"\u2014?[.-]?\(1\)|\(1\)", line))
            continue
        if cur is None:
            continue
        s = SLOT_RE.match(line)
        if s and re.fullmatch(r"[a-z]", s.group(1)):
            cur["clauses"].append((s.group(1), tokens(line[s.end():] or "")))
        elif cur["clauses"]:
            cur["clauses"][-1] = (cur["clauses"][-1][0], cur["clauses"][-1][1] + tokens(line))
    return secs


def v2_sections(lines, start):
    """yield (section_number, header_idx, header_line) and slot list (idx, marker, snippet tokens)."""
    cur = None
    for idx in range(start, len(lines)):
        line = lines[idx].strip()
        if not line:
            continue
        if SCHED_RE.match(line):
            cur = None
            continue
        m = SEC_RE.match(line)
        if m:
            if cur:
                yield cur
            cur = {"num": m.group(1), "header_idx": idx, "header": lines[idx], "slots": []}
            continue
        if cur is None:
            continue
        s = SLOT_RE.match(line)
        if s:
            cur["slots"].append((idx, s.group(1), tokens(s.group(2)), []))
        elif cur["slots"]:
            cur["slots"][-1][3].extend(tokens(line[:400]))
    if cur:
        yield cur


def main():
    ref_text = open(REF, encoding="utf-8").read()
    ref = ref_sections(ref_text)

    text = open(V2, encoding="utf-8").read()
    n0 = text.count("(")

    # 1. em-dash family corruption
    for a, b in [
        ("./)", ".—"),
        ("_(", "—("),
        ("(/}", "(1)"),
        ("(@)", "(d)"),
        (".-({/)", ".—(1)"),
        ("—{/)", "—(1)"),
        ("{/)", "(1)"),
    ]:
        if a in text:
            print(f"dash fix: {a!r} -> {b!r}  x{text.count(a)}")
            text = text.replace(a, b)

    lines = text.splitlines()
    start = body_start(text)
    bullet = 0
    for sec in v2_sections(lines, start):
        num = sec["num"]
        heref = ref.get(num)
        # 2. restore inline (1) in header where the clean source has it
        if heref and heref.get("inline1"):
            h = lines[sec["header_idx"]]
            if "\u2014" in h and "(1)" not in h:
                lines[sec["header_idx"]] = h.replace("\u2014", "\u2014(1)", 1)
        # 3. digit -> clause-letter repair (content-anchored)
        if heref and heref["clauses"]:
            k = 0
            for slot in sec["slots"]:
                idx, marker, tail, after = slot
                snippet = tail or tokens(successor(lines, idx)) or after
                if marker.isdigit():
                    best = None
                    for j in range(k, len(heref["clauses"])):
                        letter, anchor = heref["clauses"][j]
                        if overlap(anchor, snippet) >= 2:
                            best = (j, letter)
                            break
                    if best and best[1] != marker:
                        lines[idx] = lines[idx].replace(f"({marker})", f"({best[1]})", 1)
                        bullet += 1
                        k = best[0] + 1
                elif re.fullmatch(r"[a-z]", marker):
                    for j in range(k, len(heref["clauses"])):
                        letter, anchor = heref["clauses"][j]
                        if letter == marker:
                            k = j + 1
                            break

    print(f"digit-clause markers repaired: {bullet}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved {OUT}")


def successor(lines, idx):
    buf = []
    for l in lines[idx + 1:]:
        l = l.strip()
        if not l or SLOT_RE.match(l) or SEC_RE.match(l) or SCHED_RE.match(l):
            break
        buf.append(l)
        if sum(map(len, buf)) > 300:
            break
    return " ".join(buf)


def overlap(a, b):
    return len(set(a) & set(b))


if __name__ == "__main__":
    main()