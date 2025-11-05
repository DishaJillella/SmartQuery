# build_index.py
import os
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------------------------------------
# 🧩 1. PDF Loader (extracts text + page numbers)
# -------------------------------------------------------------
def load_pdfs(folder="papers"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            try:
                reader = PdfReader(path)
                pages = []
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append({"page": page_num, "text": text})
                docs.append({"source": file, "pages": pages})
                print(f"Loaded {file} ({len(pages)} pages)")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
    return docs


# -------------------------------------------------------------
# 🧩 2. Custom Text Splitter (no LangChain required)
# -------------------------------------------------------------
def split_text_custom(text, chunk_size=800, overlap=150):
    """
    Simple manual text splitter.
    Splits long text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# -------------------------------------------------------------
# 🧩 3. Chunk all pages of all documents (with metadata)
# -------------------------------------------------------------
def chunk_documents(docs):
    chunks = []
    for doc in docs:
        source = doc["source"]
        for page in doc["pages"]:
            page_num = page["page"]
            text = page["text"]
            # skip blank pages
            if not text.strip():
                continue
            parts = split_text_custom(text)
            for i, part in enumerate(parts):
                chunks.append({
                    "source": source,
                    "page": page_num,
                    "chunk_id": i,
                    "text": part
                })
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents.")
    return chunks


# -------------------------------------------------------------
# 🧩 4. Embed and Build FAISS Index
# -------------------------------------------------------------
def create_faiss_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    print(f"🔹 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # save index + metadata
    faiss.write_index(index, "vector.index")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("✅ Saved FAISS index (vector.index) and metadata (chunks.json)")


# -------------------------------------------------------------
# 🧩 5. Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    print("📚 Building FAISS index from PDFs...\n")
    docs = load_pdfs("papers")
    chunks = chunk_documents(docs)
    create_faiss_index(chunks)
    print("\n🎉 Index build complete! Ready for query_rag.py.")
