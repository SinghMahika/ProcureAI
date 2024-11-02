import os
from dotenv import load_dotenv
import openai
import logging
import streamlit as st
from langchain_chroma import Chroma
from pypdf.errors import PdfReadError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
folder_path = os.getenv('PDF_FOLDER_PATH')

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

openai.api_key = openai_api_key

# Initialize Language Model
llm = ChatOpenAI(model="gpt-4o-mini")

@st.cache_resource
def load_and_split_pdfs(folder_path):
    """Load and split all PDF documents in the specified folder."""
    all_splits = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                logging.info(f"Loaded {len(docs)} documents from {file_path}.")
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                doc_splits = text_splitter.split_documents(docs)
                all_splits.extend(doc_splits)
                
                if not doc_splits:
                    logging.warning(f"No splits created for {file_path}. Check the document content.")
                    
            except PdfReadError:
                logging.error(f"Skipping corrupted or unreadable PDF file: {file_path}")
            except Exception as e:
                logging.error(f"Unexpected error processing {file_path}: {e}")
        else:
            logging.warning(f"{file_path} is not a file.")
    
    if not all_splits:
        raise ValueError("No document splits to add to vector store. Check the input files or parsing logic.")
    
    return all_splits

@st.cache_resource
# Function to Create Vector Store
def create_vectorstore(_splits, _embedding_function, batch_size=166):
    if _splits is None or len(_splits) == 0:
        logging.error("No document splits available for vector store creation.")
        raise ValueError("No document splits to add to vector store.")
    vectorstore = Chroma(embedding_function=_embedding_function, persist_directory= folder_path) 
    chunks = [_splits[i:i + batch_size] for i in range(0, len(_splits), batch_size)]
    # Process each chunk
    for chunk in chunks:
        logging.info(f"Added {len(chunk)} documents to vector store.")
        vectorstore.add_documents(documents=chunk)
    return vectorstore.as_retriever()

@st.cache_resource
# Define Prompts and Retrieval Chains
def initialize_prompt_and_chain(_retriever):
    system_prompt = (
    "You are a helpful and polite assistant for question-answering tasks. "
    "You respond to greetings."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(_retriever, question_answer_chain)

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, _retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# @st.cache_resource
# Initialize Embeddings and Vectorstore
def initialize_chatbot():
    embedding_function = OpenAIEmbeddings()
    splits = load_and_split_pdfs(folder_path)
    retriever = create_vectorstore(splits, embedding_function)
    return initialize_prompt_and_chain(retriever)

rag_chain = initialize_chatbot() # Initialize only once, or reuse the chain if possible

def format_chat_history(chat_history):
    formatted_history = []
    for message in chat_history:
        if 'user' in message:
            formatted_history.append({"role": "user", "content": message['user']})
        elif 'ProcureAI' in message:
            formatted_history.append({"role": "assistant", "content": message['ProcureAI']})
    return formatted_history

def qna(user_input, chat_history):
    # chat_history = []
    """Process a question and return the answer."""
    try:
        logging.info(f"User Input: {user_input}") 
        response = rag_chain.invoke({"input": user_input, "chat_history": format_chat_history(chat_history)})
        return response.get("answer", "No answer generated.")
    except Exception as e:
        logging.error(f"Error in question-answering: {e}", exc_info=True)
        return "Sorry, I couldn't process that question."