# GenAI Lab — Local Llama3/Phi3 (Ollama) Streamlit App

A single-file Streamlit app (`genai_lab.py`/`genai_lab2.py`) that provides four main tools running locally with Ollama + Llama3:

1. 💬 Chat Assistant — chat with local Llama3 model
2. 📄 Summarizer — upload PDF/DOCX or paste text and get a concise summary (uses chunking)
3. 🧠 Semantic Analyzer — TF-IDF based keywords, topic sentences and optional Llama explanation
4. 🔍 Document Q&A — ask questions against an uploaded document (simple retrieval by chunk)

---

## Requirements

- Python 3.9 or newer (3.10/3.11 recommended)
- Local Ollama installation with `llama3` model available (see Ollama docs)

Install Python deps:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
````

> If `ollama` is not available via pip in your environment, keep the Ollama SDK import but install/upgrade the Ollama app separately. The app uses `ollama.chat(...)` so the Ollama Desktop/Server must be running and the `llama3` model present.

---

## Run the app

```bash
streamlit run genai_lab.py
```

Open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

---

## Files

* `genai_lab.py` — single Streamlit app file (contains Chat, Summarizer, Semantic Analyzer, Doc Q&A)---llama3
* `genai_lab2.py` — single Streamlit app file (contains Chat, Summarizer, Semantic Analyzer, Doc Q&A)---phi3
* `requirements.txt` — Python dependencies
* `README.md` — this file

---

## Usage tips & configuration

* **Model name:** The app defaults to `llama3`. If your local model has a different name, change the `model` argument passed to `ollama.chat(...)` or change the default in the UI.

* **Large documents:** For big PDFs use chunking. The app already chunks before summarization; however, extremely large files (>200k tokens) may still require manual pre-processing.

* **Performance:** Local Llama3 can be heavy. If you experience high latency, consider using a smaller local model for development and Llama3 for final runs.

* **Security:** This app runs locally. Do not expose it to the public internet without adding authentication and sanitizing uploaded files.

---

## Troubleshooting

* `ollama` import error: Ensure Ollama Python package is installed (if available) or that the Ollama app is installed and accessible. If the SDK is unavailable, you can replace the SDK call with a subprocess `ollama` CLI invocation.

* PDF text extraction issues: `PyPDF2` may not extract text from scanned PDFs. Use OCR (e.g., `pytesseract`) for scanned documents.

* Memory issues: Reduce chunk sizes in the app or use smaller models.

---

## Next steps (optional)

* Add vector-based retrieval (sentence-transformers + FAISS) for more accurate Doc Q&A.
* Create a lightweight API wrapper (FastAPI / Flask) around your Ollama calls for integration with other services.
* Add a simple login layer if you plan to share the app in a server environment.

---
Here’s a complete, clear **`README.md`** for your `genai_lab.py` project 👇
You can copy this directly into a file named `README.md` in the same folder.

---

````markdown
# 🦙 GenAI Lab — Local Chat, Summarization, and Q&A using Ollama (Llama 3 / Phi 3)

**GenAI Lab** is an all-in-one local AI research and productivity tool that runs directly on your system using **Ollama** models (like `phi3`, `llama3`).  
It provides a local, privacy-friendly alternative to cloud AI platforms — fully offline once models are downloaded.

---

## 🚀 Features

### 💬 Chat Assistant
- Real-time chat with streaming output (ChatGPT-like typing).  
- Supports any local Ollama model (e.g., `phi3`, `llama3`, `mistral`).  
- Adjustable temperature and token limits for control over creativity and length.  

### 📄 Summarizer
- Upload PDF or DOCX documents or paste raw text.  
- Performs multi-chunk summarization using local LLMs.  
- Produces clean bullet-point summaries with intermediate partial summaries.  

### 🧠 Semantic Analyzer
- Extracts **keywords**, **topic sentences**, and **themes** from any text or document.  
- Optional semantic interpretation by your chosen model.  
- TF-IDF–based topic detection for fast, explainable insights.  

### 🔍 Document Q&A
- Upload a document and ask questions directly.  
- Uses TF-IDF retrieval to find the most relevant chunks.  
- The selected context is passed to your model for grounded answers.

---

## 🧩 Requirements

Create a `requirements.txt` file with:

```text
streamlit
ollama
PyPDF2
python-docx
scikit-learn
numpy
````

Then install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have **Ollama** installed and running locally.
Download from [https://ollama.com](https://ollama.com).

---

## ⚙️ Setup & Run

1. **Pull your preferred model** (e.g., Phi-3):

   ```bash
   ollama pull phi3
   ```

   or for Llama 3:

   ```bash
   ollama pull llama3
   ```

2. **Run the app:**

   ```bash
   streamlit run genai_lab2.py
   ```

3. **Open your browser**
   The app will launch at:
   👉 `http://localhost:8501`

---

## 🧠 Model Options

In the sidebar, you can choose:

* `phi3` – small, fast, good for summarization and chat.
* `llama3` – larger, smarter, but slower.
* `phi3-mini` – ultra-light model for very low-end systems.

All responses are processed locally via the Ollama runtime.

---

## ⚡ Tips for Better Performance

* **Enable GPU acceleration** if available (Ollama detects CUDA automatically).
* Use smaller models for faster replies.
* Lower `num_predict` (token limit) to shorten responses.
* Disable streaming if you face connection issues.

---

## 🧱 Project Structure

```
GenAI-Lab/
│
├── genai_lab2.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Documentation (this file)
```

---

## 🔒 Privacy & Locality

All computation, file processing, and inference occur **locally**.
No data is sent to external servers — your documents and questions stay on your device.

---

## 🧑‍💻 Author

**Satya Srinath**
📧 [satyasrinath6@gmail.com](mailto:satyasrinath6@gmail.com)

Built for local GenAI experimentation, research, and personal productivity.

---

## 📜 License

This project is released under the **MIT License**.
You’re free to modify and use it for any personal or research purpose.

---
