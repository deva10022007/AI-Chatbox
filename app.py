streamlit run app.py
pip install streamlit langchain langchain-openai langchain-community chromadb pypdf
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Page Configuration ---
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Document Assistant (RAG Chat)")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )
    clear_chat = st.button("Clear Chat History")

# --- Session State Initialization ---
if "messages" not in st.session_state or clear_chat:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# --- Helper: Build Vectorstore & RAG Chain ---
def initialize_rag(files, api_key):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs.extend(loader.load())
            os.remove(tmp_file.name)

    # 1. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Embed and store in in-memory Chroma DB
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. LLM Setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

    # 4. Contextualize question prompt (Handles conversational memory)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 5. Answer prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. Keep answers concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Final Retrieval Chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- Process Uploaded Documents ---
if uploaded_files and openai_api_key:
    if st.session_state.rag_chain is None:
        with st.spinner("Processing documents and creating vector index..."):
            st.session_state.rag_chain = initialize_rag(uploaded_files, openai_api_key)
            st.success("Documents indexed successfully!")
elif not openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & Response Generation ---
if user_input := st.chat_input("Ask a question about your documents..."):
    if not openai_api_key:
        st.warning("Please provide an OpenAI API key.")
    elif st.session_state.rag_chain is None:
        st.warning("Please upload at least one PDF document first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Format history for LangChain
        chat_history = [
            (
                ("human", m["content"])
                if m["role"] == "user"
                else ("assistant", m["content"])
            )
            for m in st.session_state.messages[:-1]
        ]

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input, "chat_history": chat_history}
                )
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
