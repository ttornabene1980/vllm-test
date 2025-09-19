import os
import io
import streamlit as st
import openai
import pdfplumber, docx
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate

# ========== CONFIG ==========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")

if not OPENAI_API_KEY or not WEAVIATE_URL:
    st.error("Missing env vars: OPENAI_API_KEY and WEAVIATE_URL required")
    st.stop()

openai.api_key = OPENAI_API_KEY

INDEX_NAME = "RAGDocs"
TEXT_KEY = "text"

# ========== HELPERS ==========
def connect_weaviate():
    if WEAVIATE_API_KEY:
        return weaviate.Client(url=WEAVIATE_URL, auth_client_secret=WEAVIATE_API_KEY)
    return weaviate.Client(url=WEAVIATE_URL)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    content = uploaded_file.read()
    if suffix == ".pdf":
        return extract_text_from_pdf(content)
    elif suffix in [".docx", ".doc"]:
        return extract_text_from_docx(content)
    else:  # txt / md
        return content.decode("utf-8", errors="ignore")

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def ingest_files(uploaded_files):
    docs = []
    for f in uploaded_files:
        try:
            txt = extract_text(f)
            if txt.strip():
                docs.append(Document(page_content=txt, metadata={"source": f.name}))
        except Exception as e:
            st.warning(f"Errore lettura {f.name}: {e}")

    if not docs:
        st.warning("Nessun documento valido")
        return 0

    chunks = chunk_documents(docs)
    embeddings = OpenAIEmbeddings()
    client = connect_weaviate()

    Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key=TEXT_KEY
    )
    return len(chunks)

def get_vectorstore():
    client = connect_weaviate()
    embeddings = OpenAIEmbeddings()
    return Weaviate(client=client, index_name=INDEX_NAME, text_key=TEXT_KEY, embedding=embeddings)

# ========== UI ==========
st.set_page_config(page_title="RAG + Weaviate + Streamlit", layout="wide")
st.title("📚 RAG Demo con Weaviate + Streamlit")

tab1, tab2 = st.tabs(["📥 Ingestione", "💬 Q&A"])

with tab1:
    st.header("Carica documenti")
    uploaded_files = st.file_uploader("Scegli file (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Ingesta in Weaviate") and uploaded_files:
        with st.spinner("Processo documenti..."):
            n_chunks = ingest_files(uploaded_files)
        st.success(f"Ingestione completata ✅ ({n_chunks} chunks salvati in {INDEX_NAME})")

with tab2:
    st.header("Domande sui documenti")
    top_k = st.slider("Top K", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max tokens", 128, 2048, 512)

    query = st.text_input("Scrivi una domanda")
    if st.button("Interroga") and query.strip():
        store = get_vectorstore()
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        hits = retriever.get_relevant_documents(query)

        if not hits:
            st.warning("Nessun documento trovato.")
        else:
            context = "\n\n".join([f"Fonte: {h.metadata.get('source')} (chunk {h.metadata.get('chunk','?')})\n{h.page_content}" for h in hits])

            st.write("### Documenti recuperati")
            for h in hits:
                st.write(f"- {h.metadata.get('source')} (chunk {h.metadata.get('chunk','?')})")

            messages = [
                {"role": "system", "content": "Sei un assistente che usa SOLO le fonti recuperate. Cita sempre le fonti."},
                {"role": "user", "content": f"Domanda: {query}\n\nContesto:\n{context}\n\nRisposta:"}
            ]

            st.write("### Risposta LLM (streaming)")
            output_area = st.empty()
            partial = ""

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                for chunk in response:
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            partial += token
                            output_area.text_area("Risposta (in arrivo)...", value=partial, height=300)
                output_area.text_area("Risposta (completa)", value=partial, height=300)
            except Exception as e:
                st.error(f"Errore LLM: {e}")
