import json
import re

from index import CHUNKS_JSON

OUT = "graph.json"
CANON_ACT = "RTI Act 2005"

# Matches explicit clause references like "Section 8(1)(d)", "section 20 (1)",
# "Section 5(3)", "sec. 25(5)" (whitespace/newlines tolerated around parens).
REF_RE = re.compile(
    r"(?i)\b(?:section|sec)\.?\s*(\d{1,2})\s*"
    r"(?:\((\d{1,2})\)\s*)?"
    r"(?:\(([a-z])\))?"
)


def clause_key(fields):
    sec, sub, cla, rom = fields
    return (str(sec or ""), str(sub or ""), str(cla or ""), str(rom or ""))


def suffix_for(fields):
    _, sub, cla, rom = fields
    return "".join(f"({k})" for k in (sub, cla, rom) if k)


def build():
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        chunks = json.load(f)

    # Which logical clause keys exist in the Act chunks, and the canonical
    # act document name to label them with (prefer the non-consolidated copy).
    registry = {}
    for c in chunks:
        if c.get("document_type") != "act" or not c.get("section_number"):
            continue
        fields = (c.get("section_number"), c.get("subsection"),
                  c.get("clause"), c.get("sub_clause"))
        key = clause_key(fields)
        if key not in registry:
            registry[key] = c.get("source_document") or CANON_ACT

    # clause -> set of case documents that cite it
    edges = {}
    case_chunks = [c for c in chunks if c.get("document_type") in ("judgment", "decision")]
    for c in case_chunks:
        doc = c.get("source_document", "")
        for m in REF_RE.finditer(c.get("text", "")):
            fields = (m.group(1), m.group(2) or "", m.group(3) or "", "")
            key = clause_key(fields)
            if key in registry:
                edges.setdefault(key, set()).add(doc)

    def sort_key(key):
        sec, sub, cla, rom = key
        def num(v):
            return int(v) if v.isdigit() else 0
        return (num(sec), num(sub), cla, rom)

    entries = []
    for key in sorted(edges, key=sort_key):
        sec, sub, cla, rom = key
        label = f"{registry[key]}, section {sec}{suffix_for(key)}"
        entries.append({
            "source_clause": label,
            "cited_by": sorted(edges[key]),
            "relationship": "interpreted_by",
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"graph.json written with {len(entries)} Act-clause citation edges\n")
    for e in entries:
        docs = ", ".join(e["cited_by"])
        print(f"  {e['source_clause']:<52} <- {docs}")


if __name__ == "__main__":
    build()