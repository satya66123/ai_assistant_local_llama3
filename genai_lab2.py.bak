import streamlit as st
import ollama
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

st.set_page_config(page_title="🦙 GenAI Lab 2 — Local Llama3 / Phi3", layout="wide")

# ----------------------
# Helper functions
# ----------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages.append(txt)
    return "\n".join(pages)

def extract_text_from_docx(uploaded_file):
    d = docx.Document(uploaded_file)
    return "\n".join([p.text for p in d.paragraphs])

def chunk_text(text, chunk_size_words=500, overlap_words=80):
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size_words])
        chunks.append(chunk)
        i += chunk_size_words - overlap_words
    return chunks

def summarize_with_chunks(text, bullets=6, model="llama3", temperature=0.15, num_predict=512):
    chunks = chunk_text(text, chunk_size_words=700, overlap_words=120)
    partial_summaries = []
    for idx, ch in enumerate(chunks):
        prompt = (
            f"Summarize the following text into {max(3, math.ceil(bullets/2))} concise bullet points. "
            f"Keep bullets short and actionable.\n\n"
            f"--- BEGIN CHUNK {idx+1} ---\n{ch}\n--- END CHUNK ---"
        )
        resp = ollama.chat(model=model, messages=[{"role":"user","content": prompt}],
                           options={"temperature": temperature, "num_predict": num_predict})
        out = resp["message"]["content"]
        partial_summaries.append(out)
    combined = "\n".join(partial_summaries)
    final_prompt = f"Combine and shorten these partial summaries into {bullets} clear bullet points:\n\n{combined}"
    final = ollama.chat(model=model, messages=[{"role":"user","content": final_prompt}],
                       options={"temperature": 0.12, "num_predict": 384})
    return final["message"]["content"], partial_summaries

def compute_tfidf_keywords(text, top_n=15):
    tf = TfidfVectorizer(max_df=0.8, min_df=1, stop_words="english", ngram_range=(1,2))
    try:
        X = tf.fit_transform([text])
    except Exception:
        return []
    feature_array = np.array(tf.get_feature_names_out())
    # approximate idf sorting is not readily available; use feature names fallback
    return feature_array[:top_n].tolist()

def retrieve_top_chunks(query, chunks, top_k=3):
    docs = chunks + [query]
    vect = TfidfVectorizer(stop_words="english").fit_transform(docs)
    sims = cosine_similarity(vect[-1], vect[:-1])[0]
    idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in idx], sims[idx].tolist()

# ----------------------
# UI: Sidebar settings
# ----------------------
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select model", options=["phi3", "llama3", "phi3-mini"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
num_predict = st.sidebar.number_input("Max tokens (num_predict)", min_value=64, max_value=2048, value=512, step=64)
streaming_checkbox = st.sidebar.checkbox("Enable streaming in Chat", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Make sure the model is pulled locally (e.g. `ollama pull phi3`) and Ollama desktop/service is running.")

# ----------------------
# Tabs: chat, summarizer, semantic analyzer, doc qa
# ----------------------
st.title("🦙 GenAI Lab — Chat · Summarize · Analyze · Doc Q&A (Local Models)")
tabs = st.tabs(["💬 Chat Assistant", "📄 Summarizer", "🧠 Semantic Analyzer", "🔍 Document Q&A"])

# ----------------------
# 1) Chat Assistant
# ----------------------
with tabs[0]:
    st.header("💬 Chat Assistant (Streaming supported)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_text = st.text_area("You:", height=140, key="chat_input")

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("Send", key="send_chat"):
            if user_text.strip():
                st.session_state.chat_history.append(("You", user_text))
                # streaming or non-streaming
                if streaming_checkbox:
                    placeholder = st.empty()
                    st.session_state.chat_history.append(("Llama", ""))  # reserve slot
                    # stream and update last message
                    try:
                        full = ""
                        for chunk in ollama.chat(
                            model=model_choice,
                            messages=[{"role":"user","content": user_text}],
                            stream=True,
                            options={"temperature": temperature, "num_predict": num_predict}
                        ):
                            # chunk is a dict; update text
                            part = chunk.get("message", {}).get("content", "")
                            if part:
                                full += part
                                # update reserved last message
                                st.session_state.chat_history[-1] = ("Llama", full)
                                # render full history to placeholder
                                with placeholder.container():
                                    for role, txt in st.session_state.chat_history:
                                        if role == "You":
                                            st.markdown(f"**You:** {txt}")
                                        else:
                                            st.markdown(f"**{model_choice}:** {txt}")
                        # done streaming
                    except Exception as e:
                        st.error(f"Streaming error: {e}")
                else:
                    with st.spinner("Generating..."):
                        try:
                            resp = ollama.chat(model=model_choice,
                                               messages=[{"role":"user","content": user_text}],
                                               options={"temperature": temperature, "num_predict": num_predict})
                            reply = resp["message"]["content"]
                            st.session_state.chat_history.append((model_choice, reply))
                        except Exception as e:
                            st.error(f"Ollama error: {e}")

    with col3:
        st.markdown("**Conversation**")
        for role, txt in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**You:** {txt}")
            else:
                st.markdown(f"**{role}:** {txt}")

# ----------------------
# 2) Summarizer
# ----------------------
with tabs[1]:
    st.header("📄 Summarizer (PDF / DOCX / Text)")
    uploaded_file = st.file_uploader("Upload PDF or DOCX (optional)", type=["pdf", "docx"])
    pasted_text = st.text_area("Or paste text here (overrides upload)", height=200)
    bullets = st.number_input("Number of bullets (final)", min_value=3, max_value=20, value=6)
    summarize_btn = st.button("Summarize Document", key="summ_btn")

    doc_text = ""
    if uploaded_file and not pasted_text.strip():
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                doc_text = extract_text_from_pdf(uploaded_file)
            elif ext == "docx":
                doc_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            doc_text = ""
    final_text = pasted_text.strip() if pasted_text.strip() else doc_text

    if summarize_btn:
        if not final_text:
            st.warning("Upload a document or paste text first.")
        else:
            with st.spinner("Summarizing (chunking + LLM)..."):
                try:
                    final_summary, partials = summarize_with_chunks(final_text, bullets=bullets,
                                                                    model=model_choice, temperature=temperature,
                                                                    num_predict=num_predict)
                    st.subheader("Final Summary")
                    st.write(final_summary)
                    st.subheader("Partial Summaries")
                    for i, p in enumerate(partials):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(p)
                except Exception as e:
                    st.error(f"Summarization error: {e}")

# ----------------------
# 3) Semantic Analyzer
# ----------------------
with tabs[2]:
    st.header("🧠 Semantic Analyzer")
    uploaded_sem = st.file_uploader("Upload PDF or DOCX (optional)", type=["pdf", "docx"], key="sem_up")
    pasted_sem = st.text_area("Or paste text here (overrides upload)", height=200, key="sem_paste")
    query = st.text_input("Optional: query to retrieve relevant chunks (e.g., 'key findings')", key="sem_query")
    run_sem = st.button("Run Semantic Analysis", key="run_sem")

    sem_text = ""
    if uploaded_sem and not pasted_sem.strip():
        ext = uploaded_sem.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                sem_text = extract_text_from_pdf(uploaded_sem)
            elif ext == "docx":
                sem_text = extract_text_from_docx(uploaded_sem)
        except Exception as e:
            st.error(f"Read error: {e}")
    final_sem_text = pasted_sem.strip() if pasted_sem.strip() else sem_text

    if run_sem:
        if not final_sem_text:
            st.warning("Please provide text or upload a document.")
        else:
            with st.spinner("Computing TF-IDF keywords and retrieving chunks..."):
                try:
                    keywords = compute_tfidf_keywords(final_sem_text, top_n=20)
                    chunks = chunk_text(final_sem_text, chunk_size_words=400, overlap_words=80)
                    topic_sentences = []
                    sentences = [s.strip() for s in final_sem_text.replace("\n", " ").split(".") if len(s.strip()) > 20]
                    if sentences:
                        try:
                            vect = TfidfVectorizer(stop_words="english").fit_transform(sentences)
                            scores = vect.sum(axis=1).A1
                            top_idx = np.argsort(scores)[::-1][:5]
                            topic_sentences = [sentences[i].strip() for i in top_idx]
                        except Exception:
                            topic_sentences = sentences[:5]
                    retrieved = []
                    similarities = []
                    if query.strip():
                        retrieved, similarities = retrieve_top_chunks(query, chunks, top_k=3)

                    st.subheader("Top Keywords")
                    st.write(keywords[:30])
                    st.subheader("Top Topic Sentences")
                    for i, s in enumerate(topic_sentences):
                        st.write(f"{i+1}. {s}")
                    if query.strip():
                        st.subheader(f"Top retrieved chunks for: {query}")
                        for i, (c, sim) in enumerate(zip(retrieved, similarities)):
                            st.markdown(f"**Chunk {i+1} — sim {sim:.3f}**")
                            st.write(c[:1000] + ("..." if len(c) > 1000 else ""))

                    if st.checkbox("Ask model to interpret keywords & topics", key="interp"):
                        compose = (
                            "I have the following keywords and topic sentences extracted from a document.\n\n"
                            f"Keywords: {', '.join(keywords[:20])}\n\n"
                            "Top sentences:\n" + "\n".join(f"- {s}" for s in topic_sentences[:6]) +
                            "\n\nPlease provide a clear, concise explanation of the main themes and one short title."
                        )
                        with st.spinner("Asking model..."):
                            resp = ollama.chat(model=model_choice, messages=[{"role":"user", "content": compose}],
                                               options={"temperature": temperature, "num_predict": num_predict})
                            st.subheader("Model Interpretation")
                            st.write(resp["message"]["content"])
                except Exception as e:
                    st.error(f"Semantic analysis error: {e}")

# ----------------------
# 4) Document Q&A
# ----------------------
with tabs[3]:
    st.header("🔍 Document Q&A (Ask your document)")
    uploaded_qa = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key="qa_up")
    question = st.text_input("Enter your question about the uploaded doc", key="qa_q")
    ask_btn = st.button("Ask Document", key="ask_doc")

    doc_text = ""
    if uploaded_qa:
        ext = uploaded_qa.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                doc_text = extract_text_from_pdf(uploaded_qa)
            elif ext == "docx":
                doc_text = extract_text_from_docx(uploaded_qa)
        except Exception as e:
            st.error(f"Read error: {e}")

    if ask_btn:
        if not doc_text.strip() or not question.strip():
            st.warning("Please upload a document and enter a question.")
        else:
            # retrieve top chunks then query model with those contexts
            chunks = chunk_text(doc_text, chunk_size_words=500, overlap_words=80)
            top_chunks, sims = retrieve_top_chunks(question, chunks, top_k=3)
            context = "\n\n---\n\n".join(top_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely with references to the context."
            with st.spinner("Querying model..."):
                try:
                    resp = ollama.chat(model=model_choice, messages=[{"role":"user","content": prompt}],
                                       options={"temperature": temperature, "num_predict": num_predict})
                    st.subheader("Answer")
                    st.write(resp["message"]["content"])
                    st.subheader("Used context (top chunks)")
                    for i, (c, s) in enumerate(zip(top_chunks, sims)):
                        st.markdown(f"**Chunk {i+1} — sim {s:.3f}**")
                        st.write(c[:1000] + ("..." if len(c) > 1000 else ""))
                except Exception as e:
                    st.error(f"Ollama error: {e}")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.write("Run locally — make sure Ollama Desktop/Service is running and model pulled (e.g., `ollama pull phi3`).")
