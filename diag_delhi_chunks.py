import index

cfg = next(c for c in index.CORPUS if c["source_document"] == "Delhi RTI Act 2001")
text = open(cfg["path"], encoding="utf-8").read()
parts = index.chunk_act(index.repair_delhi_sections(text))

parents = [c for c in parts if c["chunk_type"] == "parent"]
parents.sort(key=lambda c: (int(c["section_number"]), c["subsection"] or ""))
print(f"parents: {len(parents)}")
for c in parents:
    head = c["text"].strip().replace("\n", " ")
    print(f"  sec {c['section_number']:>2}  {len(c['text']):>6} ch  {head[:80]}")

by_sec = {}
for c in parts:
    if c["chunk_type"] == "child":
        by_sec.setdefault(c["section_number"], []).append(c)

for sec in ("5", "6", "9"):
    kids = by_sec.get(sec, [])
    p = next((c for c in parents if c["section_number"] == sec), None)
    print(f"\nsection {sec} children: {len(kids)}")
    if p:
        print(f"  parent ({len(p['text'])} ch): {p['text'][:220]}")
    for c in kids:
        sub = c["subsection"] or "-"
        cla = c["clause"] or "-"
        rom = c["sub_clause"] or "-"
        print(f"  ({sub})({cla})({rom}) {len(c['text']):>5} ch: {c['text'][:200]}")

kids_by_phrase = {
    "trade and commercial secrets": "section 6",
    "within thirty days": "section 5",
    "within 15 days": "section 5",
    "penalty": "section 9",
    "personally liable for furnishing": "section 9",
}
for phrase, where in kids_by_phrase.items():
    hits = [c for c in parts if phrase.lower() in c["text"].lower()]
    print(f"\n'{phrase}' ({where}):")
    for c in hits:
        print(f"  sec {c['section_number']} {c['chunk_type']} ({len(c['text'])} ch): {c['text'][:120]}")