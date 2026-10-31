import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Philippine Cyber Law RAG Assistant", layout="centered")

# Sidebar configuration
st.sidebar.title("🔐 Configuration")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Google Gemini API Key in the sidebar to start.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

st.title("🛡️ Philippine Digital Bank & Cyber Scam Laws Assistant")
st.write("An interactive RAG chatbot powered by Google Gemini and your legal reference compendium.")

@st.cache_resource
def initialize_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    loader = PyPDFDirectoryLoader("knowledge_base")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

with st.spinner("Initializing legal knowledge base..."):
    vectorstore = initialize_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

system_prompt = (
    "You are an expert legal assistant specializing in Philippine digital banking laws, "
    "cyber scams, and financial consumer protection (such as R.A. 12010 AFASA Act). "
    "Use the following retrieved context to answer the user's question accurately. "
    "If you don't know the answer, say that you don't know. Keep your answers professional, "
    "structured, and cite specific pages or sections from the context.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            sources = set()
            for doc in response.get("context", []):
                source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", 0) + 1
                sources.add(f"{source_name}, page {page}")
                
            if sources:
                answer += "\n\n**Sources:**\n* " + "\n* ".join(sources)
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
