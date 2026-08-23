"""Build the local vector index from the UET Mardan dataset PDFs.

Extracts text from the PDFs, splits it into overlapping chunks, embeds the
chunks with the OpenAI embeddings API, and saves the index (chunks + vectors)
to index/ so the app can answer questions offline from the local index.
"""
import json
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDFS = [
    BASE_DIR / "uet_chatabot_dataset.pdf",   # 394-page prospectus
    BASE_DIR / "uet_chatbot_dataset.pdf",    # 1-page university summary
]
INDEX_DIR = BASE_DIR / "index"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"


def extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, page_text)] for a PDF."""
    import pymupdf

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
            text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank runs
            if text.strip():
                pages.append((i + 1, text.strip()))
    return pages


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            # Back off to the last sentence/paragraph break within the window.
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def main() -> None:
    client = OpenAI()
    all_chunks = []

    for pdf in PDFS:
        print(f"Extracting: {pdf.name}")
        for page_no, text in extract_text(pdf):
            for chunk in chunk_text(text):
                all_chunks.append(
                    {"text": chunk, "source": pdf.name, "page": page_no}
                )
    print(f"Total chunks: {len(all_chunks)}")

    print(f"Embedding with {EMBED_MODEL} ...")
    vectors = []
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = [c["text"] for c in all_chunks[i : i + batch_size]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
        print(f"  embedded {len(vectors)}/{len(all_chunks)}")

    INDEX_DIR.mkdir(exist_ok=True)
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)
    np.save(INDEX_DIR / "vectors.npy", np.array(vectors, dtype=np.float32))
    print(f"Index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
