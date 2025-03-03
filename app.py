# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import os
import tempfile
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PAGES = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def validate_pdf(file) -> bool:
    """Validate PDF file size and content."""
    if file.size > MAX_FILE_SIZE:
        st.error(f"File {file.name} is too large. Maximum size is 10MB.")
        return False
    return True

# extracting text from pdf
def get_pdf_text(docs: List[st.UploadedFile]) -> str:
    text = ""
    total_pages = 0
    
    for pdf in docs:
        if not validate_pdf(pdf):
            continue
            
        try:
            # Create a temporary file to handle the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf.getvalue())
                tmp_file_path = tmp_file.name
            
            pdf_reader = PdfReader(tmp_file_path)
            total_pages += len(pdf_reader.pages)
            
            if total_pages > MAX_PAGES:
                st.warning(f"Maximum page limit ({MAX_PAGES}) reached. Processing first {MAX_PAGES} pages.")
                break
                
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            logger.error(f"Error processing PDF {pdf.name}: {str(e)}")
            continue
            
    return text

# converting text to chunks
def get_chunks(raw_text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )   
    chunks = text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks: List[str]):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        logger.error(f"Error creating vector store: {str(e)}")
        return None

# generating conversation chain  
def get_conversationchain(vectorstore):
    try:
        llm = ChatOpenAI(temperature=0.2)
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        logger.error(f"Error creating conversation chain: {str(e)}")
        return None

# generating response from user queries and displaying them accordingly
def handle_question(question: str):
    try:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response["chat_history"]
        
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        logger.error(f"Error processing question: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply CSS
    st.write(css, unsafe_allow_html=True)
    
    # Main header
    st.header("Chat with multiple PDFs :books:")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        st.markdown("""
        ### Instructions
        1. Upload your PDF files (max 10MB each)
        2. Click 'Process' to analyze the documents
        3. Ask questions about your documents
        """)
        
        docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process", type="primary"):
            if not docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        # Get the pdf text
                        raw_text = get_pdf_text(docs)
                        
                        if not raw_text:
                            st.error("No text could be extracted from the PDFs.")
                            return
                        
                        # Get the text chunks
                        text_chunks = get_chunks(raw_text)
                        
                        # Create vectorstore
                        vectorstore = get_vectorstore(text_chunks)
                        
                        if vectorstore:
                            # Create conversation chain
                            st.session_state.conversation = get_conversationchain(vectorstore)
                            if st.session_state.conversation:
                                st.success("Documents processed successfully! You can now ask questions.")
                                st.session_state.processed_files = [doc.name for doc in docs]
                            else:
                                st.error("Failed to create conversation chain.")
                        else:
                            st.error("Failed to create vector store.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"Error in main processing: {str(e)}")
    
    # Main chat interface
    if st.session_state.conversation:
        st.markdown("### Ask questions about your documents")
        question = st.text_input("Your question:", key="question_input")
        
        if question:
            handle_question(question)
            
        # Display processed files
        if st.session_state.processed_files:
            st.markdown("### Processed Documents")
            for file in st.session_state.processed_files:
                st.markdown(f"- {file}")
    else:
        st.info("👈 Upload and process your PDF documents to start chatting!")

if __name__ == '__main__':
    main()