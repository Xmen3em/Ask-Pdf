import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import time
import tempfile

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up embeddings using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit UI
st.title("AskPdf")
st.write("Upload PDFs and ask questions about their content.")

# Manage session state for chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Prompt the user to upload PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs if available
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Save the uploaded PDF file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(uploaded_file.getvalue())
            temppdf = temp_pdf_file.name

        # Load the PDF content
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Initialize FAISS as the vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Define the contextualized question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Initialize the LLM model (replace with your chosen LLM if needed)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Create retriever and question-answer chain
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to retrieve session history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Set up the conversational chain with message history handling
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Capture user question input
    user_input = st.text_input("Ask a question about the uploaded PDFs:")
    if user_input:
        # Start timer for response time
        start_time = time.time()

        # Fetch session history and invoke the RAG chain
        session_id = "default_session"  # Fixed session ID for simplicity
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # Calculate response time
        response_time = time.time() - start_time

        # Display chat responses in a structured format
        with st.container():
            st.write("### Chat History")
            for message in session_history.messages:
                if isinstance(message, HumanMessage):
                    st.markdown(f"<span style='color:red; font-weight:bold;'>You:</span> {message.content}", unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    st.markdown(f"<span style='color:green; font-weight:bold;'>AskPdf:</span> {message.content}", unsafe_allow_html=True)

else:
    st.warning("Please upload at least one PDF file to start the conversation.")
