## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

## RAG Q&A Conversation With PDF Including Chat History
# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ✅ CORRECT embedding model name
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("Conversational RAG With PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if not api_key:
    st.warning("Please enter the Groq API key")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)

# --------------------------------------------------
# Process PDFs
# --------------------------------------------------
if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    # ❗ Guard: no text extracted
    documents = [d for d in documents if d.page_content.strip()]

    if not documents:
        st.error("No extractable text found in uploaded PDFs.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(documents)

    if not splits:
        st.error("No chunks created from documents.")
        st.stop()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # --------------------------------------------------
    # History-aware retriever
    # --------------------------------------------------
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question, "
            "rewrite the question so it is standalone."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # --------------------------------------------------
    # QA Chain
    # --------------------------------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Use the following context to answer the question. "
            "If you don't know, say you don't know.\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # --------------------------------------------------
    # Chat history
    # --------------------------------------------------
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # --------------------------------------------------
    # Chat UI
    # --------------------------------------------------
    user_input = st.text_input("Your question:")

    if user_input:
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.write("### Assistant")
        st.write(response["answer"])

        st.write("### Chat History")
        st.write(st.session_state.store[session_id].messages)










