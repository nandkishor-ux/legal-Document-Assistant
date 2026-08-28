import json
import os
import re
import shutil

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
COLLECTION = "rti_docs"
QDRANT_PATH = "vectorstore/qdrant"
CHUNKS_JSON = "vectorstore/chunks.json"

TARGET_CHUNK = 1200       # soft target for paragraph (case-doc) chunks
MAX_CHUNK = 2200          # a single runaway paragraph never gets hard-split

CORPUS = [
    {"path": "processed/rti_act_2005_final.txt",
     "source_document": "RTI Act 2005",
     "document_type": "act"},
    {"path": "processed/rti_act_2005_v2_repaired_final.txt",
     "source_document": "RTI Act 2005 (consolidated 01.02.2011)",
     "document_type": "act"},
    {"path": "processed/delhi_rti_2001_ocr_final.txt",
     "source_document": "Delhi RTI Act 2001",
     "document_type": "act"},
    {"path": "processed/hc_judgment_2021_final.txt",
     "source_document": "Delhi HC Judgment, 22.01.2021",
     "document_type": "judgment"},
    {"path": "processed/cic_decision_2026_final.txt",
     "source_document": "CIC Decision, 23.04.2026",
     "document_type": "decision"},
]

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "shall", "any", "his",
    "her", "its", "are", "was", "were", "been", "has", "have", "had", "not",
    "but", "all", "who", "whom", "which", "where", "when", "than", "then",
    "may", "can", "will", "would", "should", "could", "into", "upon", "such",
    "also", "more", "most", "some", "other", "others", "before", "after",
    "under", "over", "within", "without", "between", "through", "per", "of",
    "to", "in", "on", "by", "as", "at", "or", "if", "be", "is",
}

ACT_TITLE = "THE RIGHT TO INFORMATION ACT, 2005"

CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+(?:[IVXLCDM]{1,6}|\d{1,2})\b\s*(.*)$", re.IGNORECASE)
SCHED_RE = re.compile(r"^\(?\s*(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH)\s+SCHEDULE\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(\d{1,2})\s*[.,;:)\s]+\s*(?=[A-Z])")
SUB_RE = re.compile(r"^\s*\((\d{1,2})\)\s*(.*)$")
CLA_RE = re.compile(r"^\s*\(([a-z])\)\s*(.*)$")
ROM_RE = re.compile(r"^\s*\(([ivxlcdm]{2,6})\)\s*(.*)$", re.IGNORECASE)
INLINE_RE = re.compile(r"\((\d{1,2}|[a-z]{1,2}|[ivxlcdm]{1,6})\)")
INLINE_SEP = "-—–:.("


def act_body_start(text):
    """Cut ToC/front matter (ARRANGEMENT OF SECTIONS ... body title)."""
    a = text.upper().find("ARRANGEMENT OF SECTIONS")
    if a < 0:
        return 0
    t = text.find(ACT_TITLE, a)
    return t + len(ACT_TITLE) if t >= 0 else 0


def marker_kind(m):
    if m.isdigit():
        return "sub"
    if re.fullmatch(r"[ivxlcdm]{2,6}", m):
        return "rom"
    return "clu"


def inline_starts(s):
    out = []
    for m in INLINE_RE.finditer(s):
        before = s[:m.start()].rstrip()
        if not before or before[-1] in INLINE_SEP:
            out.append((m.start(), marker_kind(m.group(1)), m.group(1)))
    return out


def open_unit(st, section, kind, marker):
    """Open a new (sub, clause, subclause) node inside the current section."""
    if kind == "sub":
        node = {"marker": marker, "text": [], "clauses": [], "subclauses": []}
        section["nodes"].append(node)
        st["sub"], st["cla"], st["rom"] = node, None, None
    elif kind == "clu":
        if st["sub"] is None:
            parent = {"marker": "", "text": [], "clauses": [], "subclauses": []}
            section["nodes"].append(parent)
            st["sub"] = parent
        node = {"marker": marker, "text": [], "subclauses": []}
        st["sub"]["clauses"].append(node)
        st["cla"], st["rom"] = node, None
    else:  # roman sub-clause
        if st["cla"] is None:
            if st["sub"] is None:
                parent = {"marker": "", "text": [], "clauses": [], "subclauses": []}
                section["nodes"].append(parent)
                st["sub"] = parent
            node = {"marker": "", "text": [], "subclauses": []}
            st["sub"]["clauses"].append(node)
            st["cla"] = node
        node = {"marker": marker, "text": []}
        st["cla"]["subclauses"].append(node)
        st["rom"] = node


def parse_act(text):
    body = text[act_body_start(text):]
    sections, cur, chap, sched = [], None, "", None
    st = {"sub": None, "cla": None, "rom": None}

    def flush_sched():
        nonlocal sched
        if sched is not None:
            sections.append({"chapter": "", "number": "", "heading": sched["name"],
                             "intro": sched["lines"], "nodes": [],
                             "schedule": sched["name"]})
            sched = None

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if SCHED_RE.match(line):
            name = line.strip("() ").strip()
            if sched is not None and name.strip(". ") .lower() == sched["name"].strip(".").lower():
                sched["lines"].append(line)
                continue
            flush_sched()
            if cur is not None:
                sections.append(cur)
                cur = None
            st.update(sub=None, cla=None, rom=None)
            sched = {"name": name, "lines": []}
            continue
        if sched is not None:
            sched["lines"].append(line)
            continue
        m = CHAPTER_RE.match(line)
        if m:
            chap = m.group(1).strip(" \t-—–:.)(")
            continue
        m = SECTION_RE.match(line)
        if m:
            if cur is not None:
                sections.append(cur)
            cur = {"chapter": chap, "number": m.group(1),
                   "heading": "", "intro": [], "nodes": []}
            st.update(sub=None, cla=None, rom=None)
            rest = line[m.end():]
            marks = inline_starts(rest)
            if marks:
                cur["heading"] = rest[:marks[0][0]].rstrip()
                for i, (start, kind, marker) in enumerate(marks):
                    end = marks[i + 1][0] if i + 1 < len(marks) else len(rest)
                    piece = rest[start + len(marker) + 2:end].strip()
                    open_unit(st, cur, kind, marker)
                    if piece:
                        st["rom"]["text"].append(piece) if st["rom"] is not None \
                            else (st["cla"]["text"].append(piece) if st["cla"] is not None
                                  else st["sub"]["text"].append(piece))
            else:
                cur["heading"] = rest.rstrip()
            continue
        m = SUB_RE.match(line)
        if m:
            open_unit(st, cur, "sub", m.group(1))
            if m.group(2):
                st["sub"]["text"].append(m.group(2))
            continue
        m = CLA_RE.match(line)
        if m:
            open_unit(st, cur, "clu", m.group(1))
            if m.group(2):
                st["cla"]["text"].append(m.group(2))
            continue
        m = ROM_RE.match(line)
        if m:
            open_unit(st, cur, "rom", m.group(1))
            if m.group(2):
                st["rom"]["text"].append(m.group(2))
            continue
        # plain content line
        if cur is not None:
            if st["rom"] is not None:
                st["rom"]["text"].append(line)
            elif st["cla"] is not None:
                st["cla"]["text"].append(line)
            elif st["sub"] is not None:
                st["sub"]["text"].append(line)
            else:
                cur["intro"].append(line)

    flush_sched()
    if cur is not None:
        sections.append(cur)
    return sections


def compose(lines):
    return " ".join(l.strip() for l in lines if l.strip())


def wrap(marker, body):
    return f"({marker}) {body}".strip() if marker else body


def meta(section, stype, sub="", cla="", rom="", text=""):
    return {
        "source_document": None,
        "document_type": "act",
        "chunk_type": stype,
        "chapter": section["chapter"],
        "section_number": section["number"],
        "subsection": sub,
        "clause": cla,
        "sub_clause": rom,
        "text": text,
    }


def section_children(section):
    children = []
    for sub in section["nodes"]:
        if sub["clauses"]:
            for cla in sub["clauses"]:
                parts = []
                if cla["text"]:
                    parts.append(wrap(cla["marker"], compose(cla["text"])))
                for rom in cla["subclauses"]:
                    if rom["text"]:
                        parts.append(wrap(rom["marker"], compose(rom["text"])))
                if parts:
                    children.append(meta(section, "child", sub["marker"], cla["marker"], "", " ".join(parts).strip()))
                for rom in cla["subclauses"]:
                    t = compose(rom["text"])
                    if t:
                        children.append(meta(section, "child", sub["marker"], cla["marker"], rom["marker"], wrap(rom["marker"], t)))
        elif sub["subclauses"]:
            for rom in sub["subclauses"]:
                t = compose(rom["text"])
                if t:
                    children.append(meta(section, "child", sub["marker"], "", rom["marker"], wrap(rom["marker"], t)))
        else:
            t = compose(sub["text"])
            if t:
                children.append(meta(section, "child", sub["marker"], "", "", wrap(sub["marker"], t)))
    return children


def section_parent(section):
    parts = [section["heading"].strip()]
    if section["intro"]:
        parts.append(compose(section["intro"]))
    for sub in section["nodes"]:
        if sub["text"]:
            parts.append(wrap(sub["marker"], compose(sub["text"])))
        for cla in sub["clauses"]:
            if cla["text"]:
                parts.append(wrap(cla["marker"], compose(cla["text"])))
            for rom in cla["subclauses"]:
                if rom["text"]:
                    parts.append(wrap(rom["marker"], compose(rom["text"])))
        for rom in sub["subclauses"]:
            if rom["text"]:
                parts.append(wrap(rom["marker"], compose(rom["text"])))
    return " ".join(p for p in parts if p)


def chunk_act(text):
    chunks = []
    for section in parse_act(text):
        parent_text = section_parent(section)
        if not (parent_text or section["heading"]):
            continue
        m = meta(section, "parent", text=parent_text)
        chunks.append(m)
        for child in section_children(section):
            chunks.append(child)
    return chunks


def split_paras(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1:
        paras = [l.strip() for l in text.splitlines() if l.strip()]
    return paras


def chunk_case(text, source_document, document_type):
    chunks, cur, first, last = [], [], None, None

    def flush():
        nonlocal cur, first, last
        if not cur:
            return
        para_num = str(first) if first == last else f"{first}-{last}"
        chunks.append({
            "source_document": source_document,
            "document_type": document_type,
            "chunk_type": "paragraph",
            "paragraph_number": para_num,
            "section_number": "", "subsection": "", "clause": "", "sub_clause": "",
            "chapter": "",
            "text": "\n\n".join(cur).strip(),
        })
        cur, first, last = [], None, None

    for idx, p in enumerate(split_paras(text), 1):
        if len(p) > TARGET_CHUNK:
            flush()
            chunks.append({
                "source_document": source_document,
                "document_type": document_type,
                "chunk_type": "paragraph",
                "paragraph_number": str(idx),
                "section_number": "", "subsection": "", "clause": "", "sub_clause": "",
                "chapter": "",
                "text": p,
            })
            continue
        if cur and sum(len(c) for c in cur) + len(p) + 2 > TARGET_CHUNK:
            flush()
        cur.append(p)
        if first is None:
            first = idx
        last = idx
    flush()
    return chunks


def label_for(c):
    """Citation anchor for a chunk, e.g. 'RTI Act 2005, section 8, subsection (1), clause (j)'."""
    doc = c.get("source_document", "") or ""
    if c.get("document_type") == "act":
        parts = [doc]
        if c.get("section_number"):
            suffix = "".join(f"({k})" for k in (c.get("subsection"), c.get("clause"), c.get("sub_clause")) if k)
            parts.append(f"section {c['section_number']}{suffix}")
        return ", ".join(p for p in parts if p)
    if c.get("paragraph_number"):
        return f"{doc}, paragraph {c['paragraph_number']}"
    return doc


def read_corpus():
    records = []
    for cfg in CORPUS:
        with open(cfg["path"], encoding="utf-8") as f:
            text = f.read()
        records.append({**cfg, "text": text})
        print(f"  {cfg['source_document']:<38} {len(text):>8,} chars")
    return records


def tokenize(text):
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def main():
    print("Loading corpus")
    records = read_corpus()

    chunks = []
    for rec in records:
        if rec["document_type"] == "act":
            parts = chunk_act(rec["text"])
            for c in parts:
                c["source_document"] = rec["source_document"]
        else:
            parts = chunk_case(rec["text"], rec["source_document"], rec["document_type"])
        for c in parts:
            c["text"] = label_for(c) + "\n" + c["text"]
        chunks.extend(parts)
        n_parent = sum(1 for c in parts if c["chunk_type"] == "parent") if rec["document_type"] == "act" else "-"
        n_child = sum(1 for c in parts if c["chunk_type"] == "child") if rec["document_type"] == "act" else "-"
        print(f"  {rec['source_document']:<38} -> {len(parts):>4} chunks"
              + (f" ({n_parent} parent / {n_child} child)" if rec["document_type"] == "act" else " (paragraphs)"))

    print(f"\nTotal chunks: {len(chunks)}")

    print(f"Loading embedding model '{MODEL_NAME}'")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedded {len(vectors)} vectors @ dim {EMBED_DIM}")

    os.makedirs("vectorstore", exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    if os.path.isdir(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
    client = QdrantClient(path=QDRANT_PATH)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=i, vector=v.tolist(), payload=c) for i, (c, v) in enumerate(zip(chunks, vectors))],
    )
    n_points = client.count(COLLECTION).count
    if n_points != len(chunks):
        raise RuntimeError(f"Qdrant has {n_points} points, expected {len(chunks)}")
    info = client.get_collection(COLLECTION)
    print(f"\nQdrant collection '{COLLECTION}': {info.points_count} points")
    print(f"Chunks index   : {CHUNKS_JSON}")
    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()