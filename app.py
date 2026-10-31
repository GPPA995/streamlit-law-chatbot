import os
import time
import requests
import streamlit as st
from pathlib import Path
import torch

# Prevent PyTorch multi-threading deadlocks on cloud environments
torch.set_num_threads(1)

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(page_title="Philippine Digital Bank & Cyber Scam Laws Assistant", layout="centered")

st.title("🛡️ Philippine Digital Bank & Cyber Scam Laws Assistant")
st.write("A local RAG application powered by **sentence-transformers** embeddings, **Chroma**, and **Qwen (via Ollama)**.")

# --- Sidebar Configuration for Ollama & RAG Parameters ---
st.sidebar.title("⚙️ RAG & Ollama Settings")

# Ngrok tunnel URL input
ollama_url = st.sidebar.text_input(
    "Ollama Ngrok Base URL",
    value="https://cabdriver-handcraft-schedule.ngrok-free.dev",
    type="default"
).rstrip("/")

# Updated model tag to match your exact local qwen3:8b-q4_K_M tag[cite: 6]
ollama_model = st.sidebar.selectbox(
    "Ollama Model Tag",
    ["qwen2.5:3b", "qwen3:8b-q4_K_M"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Indexing Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 300, 2000, 800, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 100, 25)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3, 1)

DOMAIN_NAME = "Philippine Digital Bank & Cyber Scam Laws"
SOURCE_DIR = "Knowledge_base"
CHROMA_DIR = "chatbot_chroma_db"
COLLECTION_NAME = "student_chatbot"

# --- 1. Load Documents & Build Vector Store (Cached) ---
@st.cache_resource(show_spinner=True)
def initialize_vectorstore(c_size, c_overlap, t_k):
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR, exist_ok=True)
        
    supported = {".txt", ".md", ".pdf", ".docx"}
    documents = []
    
    for path in sorted(Path(SOURCE_DIR).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

            for doc in loaded:
                doc.metadata["source"] = path.name
            documents.extend(loaded)
        except Exception as exc:
            print(f"Could not load {path.name}: {exc}")

    if not documents:
        return None, None

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    # Local Embeddings (CPU-safe)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Chroma Database Setup
    active_chroma_dir = f"{CHROMA_DIR}_cs{c_size}_ov{c_overlap}_k{t_k}"
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=active_chroma_dir
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": t_k})
    return retriever, len(chunks)

with st.spinner("Initializing Chroma vector database and embedding documents..."):
    retriever, chunk_count = initialize_vectorstore(chunk_size, chunk_overlap, top_k)

if not retriever:
    st.error(f"⚠️ No readable documents found in the '{SOURCE_DIR}' folder! Please upload PDFs via GitHub.")
    st.stop()
else:
    st.sidebar.success(f"Database active! Total chunks indexed: {chunk_count}")

# --- 2. Ollama Request Handler ---
def call_ollama(messages, model_tag, base_url):
    formatted_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    payload = {
        "model": model_tag,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "num_predict": 256,
            "seed": 42
        }
    }

    bypass_headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "69420",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    request_start = time.perf_counter()

    try:
        # Attempt Chat API Endpoint
        response = requests.post(
            f"{base_url}/api/chat",
            headers=bypass_headers,
            json=payload,
            timeout=300
        )

        # Fallback to Generate API Endpoint if chat route fails
        if response.status_code in [403, 404]:
            generate_payload = {
                "model": model_tag,
                "prompt": formatted_prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": payload["options"]
            }
            response = requests.post(
                f"{base_url}/api/generate",
                headers=bypass_headers,
                json=generate_payload,
                timeout=300
            )

        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to local Ollama via Ngrok. Ensure ngrok and Ollama are active on your PC. Error: {str(e)}")

    generation_wall_seconds = time.perf_counter() - request_start

    if "message" in data:
        answer_text = (data["message"].get("content") or "").strip()
    else:
        answer_text = (data.get("response") or "").strip()

    eval_count = int(data.get("eval_count") or 0)
    eval_duration_seconds = (data.get("eval_duration") or 0) / 1_000_000_000
    tokens_per_sec = eval_count / eval_duration_seconds if eval_duration_seconds > 0 else 0.0

    return {
        "answer": answer_text,
        "generation_wall_seconds": generation_wall_seconds,
        "tokens_per_second": tokens_per_sec,
        "generated_token_count": eval_count
    }

# --- 3. Chat Interface Loop ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about Philippine digital scam laws..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving local documents and querying local Qwen model..."):
            retrieval_start = time.perf_counter()
            retrieved_docs = retriever.invoke(user_query)
            retrieval_seconds = time.perf_counter() - retrieval_start

            context_parts = []
            sources = set()
            for i, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page")
                location = f" {source}" + (f", page {page + 1}" if isinstance(page, int) else "")
                context_parts.append(f"[Source {i}:{location}]\n{doc.page_content}")
                sources.add(f"{source}" + (f" (page {page + 1})" if isinstance(page, int) else ""))

            context = "\n\n".join(context_parts) or "No relevant passages were retrieved."

            system_message = (
                f"You are a helpful assistant specializing in {DOMAIN_NAME}. "
                "Answer using the supplied context as the primary evidence. "
                "Do not invent facts or citations. If the context does not contain "
                "enough information, clearly say that the available documents do "
                "not provide a sufficient answer."
            )
            user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"
            
            messages_payload = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            try:
                res = call_ollama(messages_payload, ollama_model, ollama_url)
                answer = res["answer"]
                
                # Append telemetry & sources metrics
                answer += "\n\n**Sources:**\n* " + "\n* ".join(sources)
                answer += f"\n\n---\n*⏱️ Retrieval: {retrieval_seconds:.2f}s | Gen Time: {res['generation_wall_seconds']:.2f}s | Speed: {res['tokens_per_second']:.2f} tok/s*"
                
            except Exception as exc:
                answer = f"❌ **Ollama Connection Error:** {str(exc)}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
