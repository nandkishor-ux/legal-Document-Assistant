import json
import os
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
COLLECTION = "rti_docs"
QDRANT_PATH = "vectorstore/qdrant"
CHUNKS_JSON = "vectorstore/chunks.json"

TARGET_CHUNK = 1200
MAX_CHUNK = 2200

CORPUS = [
    {"path": "processed/rti_act_2005_final.txt", "label": "RTI Act, 2005"},
    {"path": "processed/rti_act_2005_v2_ocr_final.txt", "label": "RTI Act, 2005 (consolidated 01.02.2011)"},
    {"path": "processed/delhi_rti_2001_ocr_final.txt", "label": "Delhi RTI Act, 2001"},
    {"path": "processed/hc_judgment_2021_final.txt", "label": "Delhi HC Judgment, 22.01.2021"},
    {"path": "processed/cic_decision_2026_final.txt", "label": "CIC Decision, 23.04.2026"},
]

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "shall", "any", "his",
    "her", "its", "are", "was", "were", "been", "has", "have", "had", "not",
    "but", "all", "who", "whom", "which", "where", "when", "than", "then",
    "may", "can", "will", "would", "should", "could", "into", "upon", "such",
    "shall", "also", "more", "most", "some", "other", "others", "before",
    "after", "under", "over", "within", "without", "between", "through",
    "per", "of", "to", "in", "on", "by", "as", "at", "or", "if", "be", "is",
}


def read_corpus_files():
    records = []
    for cfg in CORPUS:
        with open(cfg["path"], encoding="utf-8") as f:
            text = f.read()
        records.append({"source": cfg["label"], "text": text})
        print(f"  {cfg['label']:<38} {len(text):>8,} chars")
    return records


def hard_split_long(p, max_len=MAX_CHUNK):
    words = p.split()
    parts, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_len and cur:
            parts.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        parts.append(" ".join(cur))
    return parts or [p]


def chunk_paragraphs(text, target=TARGET_CHUNK, max_chunk=MAX_CHUNK):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 <= target:
            cur += "\n\n" + p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chunk:
                chunks.extend(hard_split_long(p, max_chunk))
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def tokenize(text):
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def main():
    print("Loading corpus")
    records = read_corpus_files()

    chunks = []
    for rec in records:
        parts = chunk_paragraphs(rec["text"])
        for part in parts:
            chunks.append({"source": rec["source"], "text": part})
    print(f"\nTotal chunks: {len(chunks)}")

    print(f"Loading embedding model '{MODEL_NAME}'")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedded {len(vectors)} vectors @ dim {EMBED_DIM}")

    os.makedirs("vectorstore", exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    client = QdrantClient(path=QDRANT_PATH)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={"source": chunks[i]["source"], "text": chunks[i]["text"]},
            )
            for i in range(len(chunks))
        ],
    )
    info = client.get_collection(COLLECTION)
    print(f"\nQdrant collection '{COLLECTION}': {info.points_count} points")
    print(f"Chunks index   : {CHUNKS_JSON}")


if __name__ == "__main__":
    main()