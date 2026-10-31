import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Philippine Cyber Law RAG Assistant", layout="centered")

# Load API Key seamlessly from Streamlit Secrets without showing any sidebar
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Error: GOOGLE_API_KEY not found in Streamlit Secrets!")
    st.stop()

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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

template = """You are an expert legal assistant specializing in Philippine digital banking laws, 
cyber scams, and financial consumer protection (such as R.A. 12010 AFASA Act). 
Use the following retrieved context to answer the user's question accurately. 
If you don't know the answer, say that you don't know. Keep your answers professional, 
structured, and cite specific pages or sections from the context.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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
            answer = rag_chain.invoke(user_query)
            
            sources = set()
            for doc in retrieved_docs:
                source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", 0) + 1
                sources.add(f"{source_name}, page {page}")
                
            if sources:
                answer += "\n\n**Sources:**\n* " + "\n* ".join(sources)
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
