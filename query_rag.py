# query_rag.py
import faiss
import numpy as np
import json
import subprocess
from sentence_transformers import SentenceTransformer

def query_rag(question, k=3, model_name="phi"):
    """
    Offline RAG Query Pipeline with Page Numbers
    --------------------------------------------
    1. Embeds query with SentenceTransformer.
    2. Retrieves top-k chunks from FAISS.
    3. Builds a context including source + page numbers.
    4. Sends context and question to local Ollama model.
    5. Prints the citation-aware answer.
    """

    # --- Load FAISS index and chunk metadata ---
    index = faiss.read_index("vector.index")
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # --- Embed question ---
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb), k)

    # --- Build context from top chunks ---
    context_blocks = []
    for rank, idx in enumerate(I[0]):
        data = chunks[idx]
        src = data["source"]
        page = data.get("page", "?")
        text = data["text"].replace("\n", " ")
        short_text = text[:700] + ("..." if len(text) > 700 else "")
        context_blocks.append(f"[SRC_{rank}] ({src}: p.{page}) {short_text}")

    context = "\n\n".join(context_blocks)

    # --- Prompt ---
    prompt = f"""
You are a research assistant. Use only the following excerpts from research papers to answer the question.
Cite your sources inline like [SRC_0] and include filename + page numbers.
If the information is not in the context, say: "Not enough information found in the provided documents."

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

Answer concisely with citations.
"""

    # --- Run Ollama model ---
    print("\n Running local model via Ollama...")
    result = subprocess.run(
    ["ollama", "run", model_name, prompt],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore"
)


    # --- Display Answer ---
    print("\n=== ANSWER ===\n")
    print(result.stdout.strip())

    # --- Optional: show which documents were used ---
    print("\n=== SOURCES USED ===")
    for rank, idx in enumerate(I[0]):
        src = chunks[idx]["source"]
        page = chunks[idx].get("page", "?")
        print(f"[SRC_{rank}] -> {src} (page {page})")

if __name__ == "__main__":
    print("SmartQuery: Offline Research Paper Q&A System ")
    print("Welcome, please upload your files before you begin")
    while True:
        question = input("Ask a question (or type 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Thankyou for using SmartQurey! Hope to see you again.")
            break
        query_rag(question)
        print("\n" + "-" * 60 + "\n")
