#  SmartQuery: Offline Document-Based Question Answering System (RAG)

**SmartQuery** is a **Retrieval-Augmented Generation (RAG)** system that answers user questions based on the content of uploaded research papers — **completely offline**.

Unlike API-based systems, SmartQuery uses **local embeddings**, **FAISS-based vector search**, and **offline LLMs (via Ollama)** to ensure **data privacy**, **speed**, and **reliability**.  
This makes it ideal for **research**, **academia**, and **enterprise environments** where privacy and transparency are crucial.

---

### 🔍 What It Does
SmartQuery enables users to:
- Upload **PDF research papers** or documents  
- Automatically **extract, embed, and store** their content  
- Ask **natural language questions**  
- Get **short, citation-aware answers** with **page references**

It integrates **semantic search (retrieval)** and **local language modeling (generation)** to simulate a fully autonomous, explainable AI assistant.


## 🧩 System Architecture

```
            ┌────────────────────┐
            │   PDF Documents    │
            │ (Research Papers)  │
            └────────┬───────────┘
                     │
                     ▼
         ┌────────────────────┐
         │ Text Extraction &  │
         │ Chunking (PyPDF2)  │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────────┐
         │ Sentence Embedding │
         │ (SentenceTransformers) │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────────┐
         │ FAISS Vector Index │
         │  (Semantic Search) │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────────┐
         │  LLM (Ollama Phi)  │
         │   Offline Inference│
         └────────────────────┘
```



---

## ⚙️ Tech Stack

| Component | Tool / Library | Purpose |
|------------|----------------|----------|
| **Text Extraction** | PyPDF2 | Extract text from PDFs |
| **Text Chunking** | Custom splitter | Maintain context in chunks |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Convert text to numerical vectors |
| **Vector Search** | FAISS | Retrieve semantically similar chunks |
| **Language Model** | Ollama (`phi`, `llama3`, etc.) | Offline text generation |
| **Language** | Python | Core implementation |

---

## 💡 Key Features

- ⚡ **100% Offline** — No internet or APIs required  
- 🔍 **Semantic Search** — Understands meaning, not just keywords  
- 📄 **Page-Level Citations** — Trace every answer to source pages  
- 🧱 **Modular & Extensible** — Easily switch between models (phi, llama3)  
- 🔐 **Privacy-Focused** — No cloud calls, runs entirely locally  
- 🧠 **Concise, Context-Aware Answers** — Ideal for academic Q&A  

---

## 🛠️ Installation & Setup

### 🖥️ Prerequisites
- Python **3.10+**
- Ollama installed → [https://ollama.ai/download](https://ollama.ai/download)

---

### 1️⃣ Clone this repository
```bash
git clone https://github.com/DishaJillella/SmartQuery.git
cd SmartQuery
````

---

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one:

```text
PyPDF2
faiss-cpu
sentence-transformers
torch
tqdm
numpy
requests
```

---

### 4️⃣ Pull a local model using Ollama

```bash
ollama pull phi
```

or (optional)

```bash
ollama pull llama3
```

---

### 5️⃣ Add your research papers

Place all your PDFs in the `/papers` folder:

```
papers/
 ├── paper1.pdf
 ├── paper2.pdf
 └── paper3.pdf
```

---

### 6️⃣ Build the FAISS index

```bash
python build_index.py
```

This will:

* Extract text from PDFs
* Split it into small overlapping chunks
* Embed and store them in `vector.index` + `chunks.json`

---

### 7️⃣ Run the Q&A system

```bash
python query_rag.py
```



##  How It Works

1. **Document Loading** – Extracts text from PDFs in `/papers/`
2. **Chunking** – Splits text into 800-character segments with overlaps
3. **Embeddings** – Converts chunks into 384-dimensional semantic vectors
4. **FAISS Indexing** – Builds a searchable database of all document embeddings
5. **Query Handling** – User query is embedded → matched → context passed to LLM
6. **Answer Generation** – The LLM (Phi) generates short, cited answers

---



## 🎯 Use Cases

* Research paper summarization
* Legal or medical document Q&A
* Academic literature review assistant
* Private enterprise document analysis
* Offline AI assistant for restricted environments

---



## 🔒 Privacy & Security

All processing — including embedding, retrieval, and generation — happens **locally**.
No data leaves your system.
This ensures maximum privacy and makes SmartQuery suitable for sensitive domains.

---

## 🚧 Future Enhancements

* ✅ Add a **Streamlit UI** for easy interaction
* ✅ Integrate **incremental document updates**
* ✅ Add support for **tables and images** in PDFs
* ✅ Extend to **Agentic RAG** (multi-step reasoning using LangGraph)

---

## 🧾 Credits

**Developed by:** Disha Jillella

**Institution:** CBIT, Hyderabad, Telangana

**Mentor:** [Dr. Y Ramadevi]

**Year:** 2025

**Technologies:** Python, FAISS, SentenceTransformers, Ollama
