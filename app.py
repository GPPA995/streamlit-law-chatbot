import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

st.set_page_config(page_title="Philippine Cyber Law RAG Assistant", layout="centered")

# Retrieve and validate API Key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = str(st.secrets["GOOGLE_API_KEY"]).strip().strip('"').strip("'")
else:
    st.error("Error: GOOGLE_API_KEY not found in Streamlit Secrets!")
    st.stop()

# Initialize the Google GenAI client cleanly with the key
client = genai.Client(api_key=api_key)

st.title("🛡️ Philippine Digital Bank & Cyber Scam Laws Assistant")
st.write("An interactive RAG chatbot powered by Google Gemini and your legal reference compendium.")

@st.cache_resource
def initialize_vectorstore():
    loader = PyPDFDirectoryLoader("Knowledge_base")
    docs = loader.load()
    
    if not docs:
        st.error("Error: No PDF documents found in the 'Knowledge_base' folder!")
        st.stop()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

with st.spinner("Initializing legal knowledge base..."):
    vectorstore = initialize_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about Philippine digital scam laws..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""You are an expert legal assistant specializing in Philippine digital banking laws, 
cyber scams, and financial consumer protection (such as R.A. 12010 AFASA Act). 
Use the following retrieved context to answer the user's question accurately. 
If you don't know the answer, say that you don't know. Keep your answers professional, 
structured, and cite specific pages or sections from the context.

Context:
{context_text}

Question: {user_query}
"""
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                answer = response.text
            except Exception as e:
                answer = f"API Error encountered: {str(e)}"
            
            sources = set()
            for doc in retrieved_docs:
                source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", 0) + 1
                sources.add(f"{source_name}, page {page}")
                
            if sources and "API Error" not in answer:
                answer += "\n\n**Sources:**\n* " + "\n* ".join(sources)
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
