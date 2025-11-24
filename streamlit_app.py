"""
Document Q&A Chatbot
====================
A multi-document question-answering system using LangChain, OpenAI, and FAISS.

Features:
- Upload and process multiple PDF documents
- Ask questions across all documents with source citations
- AI-generated question suggestions
- Analytics dashboard for usage tracking
- Export conversations as markdown
- Customizable settings for document processing and answer generation

Author: Yash Jalan
Created: November 2024
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pandas as pd
from datetime import datetime
from collections import Counter
import time

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Application constants
MAX_FILE_SIZE_MB = 50  # Maximum PDF file size in megabytes
MAX_TOTAL_FILES = 10   # Maximum number of documents allowed

# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate_api_keys():
    """
    Validate that required API keys are present and properly formatted.
    Stops execution if keys are missing or invalid.
    """
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        st.error("⚠️ **Missing API Key!** Please add OPENAI_API_KEY to your .env file")
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
        st.stop()
    
    if not openai_key.startswith('sk-'):
        st.error("⚠️ **Invalid API Key Format!** OpenAI keys should start with 'sk-'")
        st.stop()

# Run validation before app starts
validate_api_keys()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

def add_custom_css():
    """
    Apply custom CSS styling for a professional, modern UI.
    Includes colors, spacing, animations, and responsive design.
    """
    st.markdown("""
        <style>
        /* Main container background */
        .main {
            background-color: #f5f7fa;
        }
        
        /* Header styling with blue theme */
        h1 {
            color: #1e3a8a;
            font-weight: 700;
            padding-bottom: 10px;
            border-bottom: 3px solid #3b82f6;
        }
        
        /* Sidebar with white background */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 2px solid #e5e7eb;
        }
        
        /* Metrics display styling */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 600;
            color: #1e40af;
        }
        
        /* Chat message containers */
        .stChatMessage {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button styling with hover effects */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Download button with green accent */
        .stDownloadButton > button {
            background-color: #10b981;
            color: white;
            border: none;
        }
        
        .stDownloadButton > button:hover {
            background-color: #059669;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f3f4f6;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #3b82f6;
        }
        
        /* Dividers */
        hr {
            margin: 20px 0;
            border: none;
            border-top: 2px solid #e5e7eb;
        }
        
        /* File uploader area */
        [data-testid="stFileUploader"] {
            background-color: #f9fafb;
            border: 2px dashed #d1d5db;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #3b82f6 !important;
        }
        
        /* Source preview text */
        .stExpander pre {
            background-color: #f9fafb;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #3b82f6;
        }
        
        /* Success, warning, and error message styling */
        .stSuccess {
            background-color: #d1fae5;
            color: #065f46;
            border-radius: 8px;
        }
        
        .stWarning {
            background-color: #fef3c7;
            color: #92400e;
            border-radius: 8px;
        }
        
        .stError {
            background-color: #fee2e2;
            color: #991b1b;
            border-radius: 8px;
        }
        
        /* Analytics card styling */
        .analytics-card {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 10px 0;
        }
        
        /* Progress bar color */
        .stProgress > div > div {
            background-color: #3b82f6;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply styling
add_custom_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Vector store for document embeddings
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Chat conversation history (list of tuples: (question, answer, sources))
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Dictionary of processed files: filename -> statistics
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

# All document chunks from all processed PDFs
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []

# Analytics tracking data
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_questions': 0,
        'response_times': [],
        'documents_referenced': Counter(),
        'topics_keywords': [],
        'question_history': []
    }

# User-configurable settings
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'chunk_size': 1000,        # Size of text chunks
        'chunk_overlap': 200,       # Overlap between chunks
        'num_sources': 3,           # Number of sources to retrieve
        'temperature': 0.0          # LLM temperature (0=focused, 1=creative)
    }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_embeddings():
    """
    Initialize OpenAI embeddings model.
    Cached to avoid re-initialization on every rerun.
    
    Returns:
        OpenAIEmbeddings: Embedding model for document vectorization
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return embeddings
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {str(e)}")
        st.info("💡 Check your OPENAI_API_KEY in the .env file")
        st.stop()

# Initialize embeddings (runs once)
embeddings = initialize_embeddings()

def get_llm(temperature):
    """
    Get OpenAI language model with specified temperature.
    
    Args:
        temperature (float): Controls randomness (0.0 to 1.0)
    
    Returns:
        ChatOpenAI: Configured language model or None on error
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=temperature,
            max_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None

# ============================================================================
# AI QUESTION GENERATION
# ============================================================================
@st.cache_data(ttl=3600)
def generate_suggested_questions(_vectorstore, num_questions=5, current_documents=None):
    """
    Generate questions ONLY from currently uploaded documents.
    
    Args:
        _vectorstore: FAISS vector store
        num_questions (int): Number of questions to generate
        current_documents (list): List of document names uploaded in current session
    """
    try:
        # Get documents from vectorstore
        docs = _vectorstore.similarity_search("main topics concepts key ideas", k=30)
        
        # ⭐ FILTER: Only keep chunks from currently uploaded documents
        if current_documents:
            docs = [doc for doc in docs if doc.metadata.get('document') in current_documents]
        
        # If no docs found, try broader search
        if len(docs) < 5 and current_documents:
            all_docs = _vectorstore.similarity_search("main topics concepts key ideas", k=100)
            docs = [doc for doc in all_docs if doc.metadata.get('document') in current_documents]
        
        # If still no docs, return generic questions
        if not docs:
            return [
                "What are the main topics covered in your uploaded documents?",
                "Explain the key concepts discussed",
                "What are the important points mentioned?",
            ]
        
        # Extract unique content samples
        content_samples = []
        seen_pages = set()
        
        for doc in docs:
            page = doc.metadata.get('page', 0)
            doc_name = doc.metadata.get('document', 'Unknown')
            
            if (doc_name, page) not in seen_pages:
                content_samples.append(doc.page_content[:500])
                seen_pages.add((doc_name, page))
        
        combined_content = "\n\n".join(content_samples[:5])
        
        llm = get_llm(0.7)
        if not llm:
            raise Exception("Failed to get LLM")
        
        prompt = f"""Based on the following document excerpts, generate {num_questions} interesting and diverse questions.

Make the questions:
- Specific and clear
- Cover different topics
- Range from basic to advanced
- Naturally worded

Document excerpts:
{combined_content}

Generate exactly {num_questions} questions, one per line, without numbering:"""

        response = llm.invoke(prompt)
        
        questions = [q.strip() for q in response.content.strip().split('\n') 
                    if q.strip() and not q.strip().startswith(('#', '-', '*', '1', '2', '3', '4', '5'))]
        
        cleaned_questions = []
        for q in questions:
            q = q.lstrip('0123456789.)')
            q = q.strip()
            if q and len(q) > 10:
                cleaned_questions.append(q)
        
        return cleaned_questions[:num_questions]
        
    except Exception as e:
        return [
            "What are the main topics covered in your documents?",
            "Explain the key concepts discussed",
            "What are the important points mentioned?"
        ]

# ============================================================================
# ANALYTICS TRACKING
# ============================================================================

def track_analytics(question, response_time, sources):
    """
    Track analytics data for each question asked.
    
    Args:
        question (str): The question text
        response_time (float): Time taken to generate response
        sources (list): List of source documents referenced
    """
    # Increment question counter
    st.session_state.analytics['total_questions'] += 1
    
    # Track response times for average calculation
    st.session_state.analytics['response_times'].append(response_time)
    
    # Store question history with timestamp
    st.session_state.analytics['question_history'].append({
        'question': question,
        'timestamp': datetime.now(),
        'response_time': response_time
    })
    
    # Count document references
    for source in sources:
        doc_name = source.get('document', 'Unknown')
        st.session_state.analytics['documents_referenced'][doc_name] += 1
    
    # Extract keywords (simple word-based extraction)
    keywords = [word.lower() for word in question.split() if len(word) > 4]
    st.session_state.analytics['topics_keywords'].extend(keywords)

# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def export_chat_history():
    """
    Export the entire chat history as a formatted markdown file.
    Includes questions, answers, sources, and metadata.
    
    Returns:
        str: Markdown-formatted chat history or None if empty
    """
    if not st.session_state.chat_history:
        return None
    
    # Get list of documents
    document_list = ", ".join(st.session_state.processed_files.keys())
    
    # Build markdown content
    markdown_content = f"""# Document Q&A Chat History

**Documents:** {document_list}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Questions:** {st.session_state.analytics['total_questions']}

---

"""
    
    # Add each Q&A pair
    for i, chat_item in enumerate(st.session_state.chat_history):
        # Handle both old (q, a) and new (q, a, sources) formats
        if len(chat_item) == 2:
            q, a = chat_item
            sources = []
        else:
            q, a, sources = chat_item
        
        markdown_content += f"""## Question {i+1}

**Q:** {q}

**A:** {a}

"""
        
        # Add source citations
        if sources:
            markdown_content += "**Sources:**\n\n"
            for j, source in enumerate(sources):
                doc_name = source.get('document', 'Unknown')
                markdown_content += f"- Source {j+1} - {doc_name} (Page {source['page']})\n"
            markdown_content += "\n"
        
        markdown_content += "---\n\n"
    
    return markdown_content

# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_file(pdf_file):
    """
    Validate uploaded PDF file for size and type.
    
    Args:
        pdf_file: Streamlit UploadedFile object
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # Check file size
    file_size_mb = pdf_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB"
    
    # Check file extension
    if not pdf_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, "Valid"

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_pdf(pdf_file, chunk_size, chunk_overlap):
    """
    Process a single PDF file: load, split into chunks, and extract metadata.
    
    Args:
        pdf_file: Streamlit UploadedFile object
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        tuple: (chunks, stats, error) where error is None on success
    """
    try:
        start_time = time.time()
        
        # Validate file
        is_valid, message = validate_file(pdf_file)
        if not is_valid:
            raise ValueError(message)
        
        # Save uploaded file temporarily
        temp_path = f"temp_{pdf_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
        except Exception as e:
            raise Exception(f"Failed to save file: {str(e)}")
        
        # Get file size for statistics
        file_size = os.path.getsize(temp_path) / (1024 * 1024)
        
        # Load PDF using LangChain's PyPDFLoader
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("PDF appears to be empty or corrupted")
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"Failed to load PDF: {str(e)}")
        
        # Add source document name to metadata for tracking
        for doc in documents:
            doc.metadata['document'] = pdf_file.name
        
        # Split documents into smaller chunks for better retrieval
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No text content found in PDF")
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"Failed to split document: {str(e)}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Compile statistics
        stats = {
            'pages': len(documents),
            'chunks': len(chunks),
            'file_size': file_size,
            'processing_time': processing_time,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return chunks, stats, None
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None, str(e)

def add_documents(uploaded_files, chunk_size, chunk_overlap):
    """
    Process and add multiple documents to the vector store.
    Shows progress bar and handles errors gracefully.
    
    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
    """
    # Check total file limit
    if len(st.session_state.processed_files) + len(uploaded_files) > MAX_TOTAL_FILES:
        st.error(f"❌ Maximum {MAX_TOTAL_FILES} documents allowed. Remove some documents first.")
        return
    
    success_count = 0
    error_count = 0
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each file
    for idx, pdf_file in enumerate(uploaded_files):
        # Update progress bar
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {pdf_file.name}")
        
        # Skip if already processed
        if pdf_file.name in st.session_state.processed_files:
            st.warning(f"⚠️ {pdf_file.name} already loaded. Skipping.")
            continue
        
        # Process with error handling
        chunks, stats, error = process_pdf(pdf_file, chunk_size, chunk_overlap)
        
        if error:
            st.error(f"❌ Failed: {pdf_file.name} - {error}")
            error_count += 1
        else:
            # Add to document collection
            st.session_state.all_documents.extend(chunks)
            st.session_state.processed_files[pdf_file.name] = stats
            st.success(f"✅ {pdf_file.name}: {stats['pages']} pages, {stats['chunks']} chunks")
            success_count += 1
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show summary
    if success_count > 0:
        st.success(f"🎉 Successfully processed {success_count} document(s)!")
        rebuild_vectorstore()
    
    if error_count > 0:
        st.warning(f"⚠️ {error_count} document(s) failed to process.")

def rebuild_vectorstore():
    """
    Rebuild the FAISS vector store from all loaded document chunks.
    Required after adding or removing documents.
    """
    try:
        if st.session_state.all_documents:
            with st.spinner("🔄 Rebuilding vector database..."):
                st.session_state.vectorstore = FAISS.from_documents(
                    documents=st.session_state.all_documents,
                    embedding=embeddings
                )
        else:
            st.session_state.vectorstore = None
    except Exception as e:
        st.error(f"❌ Failed to rebuild vector store: {str(e)}")
        st.session_state.vectorstore = None

def remove_document(filename):
    """
    Remove a specific document from the collection and rebuild vector store.
    
    Args:
        filename (str): Name of the document to remove
    """
    try:
        # Remove from processed files dictionary
        if filename in st.session_state.processed_files:
            del st.session_state.processed_files[filename]
        
        # Filter out chunks from this document
        st.session_state.all_documents = [
            doc for doc in st.session_state.all_documents 
            if doc.metadata.get('document') != filename
        ]
        
        # Rebuild vector store without this document
        rebuild_vectorstore()
        
        # Clear chat history (since context changed)
        st.session_state.chat_history = []
        
        st.success(f"✅ Removed {filename}")
        
    except Exception as e:
        st.error(f"❌ Failed to remove document: {str(e)}")

# ============================================================================
# QUESTION ANSWERING
# ============================================================================

def ask_question(vectorstore, question, num_sources, temperature):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    Process:
    1. Search vector store for relevant document chunks
    2. Build context from retrieved chunks
    3. Generate answer using GPT-4 with context
    4. Track analytics
    
    Args:
        vectorstore: FAISS vector store
        question (str): User's question
        num_sources (int): Number of source chunks to retrieve
        temperature (float): LLM temperature setting
    
    Returns:
        tuple: (answer, sources, response_time, error)
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not question or len(question.strip()) < 3:
            raise ValueError("Please enter a valid question (at least 3 characters)")
        
        # Retrieve relevant document chunks using similarity search
        docs = vectorstore.similarity_search(question, k=num_sources)
        
        if not docs:
            raise ValueError("No relevant information found in documents")
        
        # Build context and source list
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get('page', 'Unknown')
            doc_name = doc.metadata.get('document', 'Unknown')
            
            # Add to context with source info
            context_parts.append(
                f"Source {i+1} from {doc_name} (Page {page_num + 1}):\n{doc.page_content}"
            )
            
            # Store source metadata for display
            sources.append({
                'page': page_num + 1,  # Convert to 1-indexed
                'content': doc.page_content[:200] + "...",  # Preview only
                'document': doc_name
            })
        
        # Combine all context
        context = "\n\n".join(context_parts)
        
        # Build prompt for GPT-4
        prompt = f"""Based on the following context from the documents, please answer the question.
If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        # Get LLM instance
        llm = get_llm(temperature)
        if not llm:
            raise Exception("Failed to initialize language model")
        
        # Generate answer
        response = llm.invoke(prompt)
        
        # Calculate total response time
        response_time = time.time() - start_time
        
        # Track analytics
        track_analytics(question, response_time, sources)
        
        return response.content, sources, response_time, None
        
    except Exception as e:
        response_time = time.time() - start_time
        return None, [], response_time, str(e)

# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

def show_analytics_dashboard():
    """
    Display comprehensive analytics dashboard showing usage patterns.
    Includes metrics, charts, and activity timeline.
    """
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Track your usage patterns and insights")
    
    analytics = st.session_state.analytics
    
    # Empty state
    if analytics['total_questions'] == 0:
        st.info("📭 No questions asked yet. Start chatting to see analytics!")
        
        # Preview of what will be shown
        st.markdown("##### 📈 Analytics Preview:")
        st.caption("- Total questions asked")
        st.caption("- Average response time")
        st.caption("- Most referenced documents")
        st.caption("- Popular topics and keywords")
        st.caption("- Recent activity timeline")
        return
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Total Questions",
            analytics['total_questions']
        )
    
    with col2:
        if analytics['response_times']:
            avg_time = sum(analytics['response_times']) / len(analytics['response_times'])
            st.metric(
                "⏱️ Avg Response",
                f"{avg_time:.2f}s"
            )
        else:
            st.metric("⏱️ Avg Response", "N/A")
    
    with col3:
        if analytics['documents_referenced']:
            most_used = analytics['documents_referenced'].most_common(1)[0]
            display_name = most_used[0][:20] + "..." if len(most_used[0]) > 20 else most_used[0]
            st.metric(
                "📄 Most Referenced",
                display_name,
                f"{most_used[1]} refs"
            )
        else:
            st.metric("📄 Most Referenced", "N/A")
    
    with col4:
        total_docs = len(st.session_state.processed_files)
        st.metric(
            "📚 Total Documents",
            total_docs
        )
    
    st.divider()
    
    # Document usage chart
    if analytics['documents_referenced']:
        st.markdown("#### 📑 Document Reference Frequency")
        
        try:
            doc_data = pd.DataFrame(
                analytics['documents_referenced'].most_common(),
                columns=['Document', 'References']
            )
            
            st.bar_chart(doc_data.set_index('Document'))
        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
    
    st.divider()
    
    # Top keywords display
    if analytics['topics_keywords']:
        st.markdown("#### 🔑 Top Keywords")
        
        try:
            keyword_counts = Counter(analytics['topics_keywords'])
            top_keywords = keyword_counts.most_common(10)
            
            # Display top 5 keywords in cards
            cols = st.columns(5)
            for idx, (keyword, count) in enumerate(top_keywords[:5]):
                with cols[idx]:
                    st.markdown(f"""
                        <div class="analytics-card" style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: #1e40af;">{count}</div>
                            <div style="font-size: 12px; color: #6b7280;">{keyword}</div>
                        </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying keywords: {str(e)}")
    
    st.divider()
    
    # Recent activity timeline
    st.markdown("#### 🕐 Recent Activity")
    
    try:
        recent = analytics['question_history'][-5:]  # Last 5 questions
        for item in reversed(recent):
            st.caption(
                f"🕐 {item['timestamp'].strftime('%H:%M:%S')} - "
                f"{item['question'][:60]}... ({item['response_time']:.2f}s)"
            )
    except Exception as e:
        st.error(f"Error displaying activity: {str(e)}")

# ============================================================================
# USER INTERFACE
# ============================================================================

# Header section
st.title("📚 Document Q&A Chatbot")
st.markdown("""
    <p style='font-size: 18px; color: #6b7280; margin-top: -10px;'>
        Upload multiple PDFs and get AI-powered answers with source citations
    </p>
""", unsafe_allow_html=True)

# Main navigation tabs
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

# ============================================================================
# ANALYTICS TAB
# ============================================================================

with tab2:
    show_analytics_dashboard()

# ============================================================================
# CHAT TAB
# ============================================================================

with tab1:
    st.divider()
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### 📁 Upload Documents")
        st.markdown("---")
        
        # File uploader with multiple file support
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help=f"Max {MAX_FILE_SIZE_MB}MB per file, {MAX_TOTAL_FILES} files total"
        )
        
        # Process button
        if uploaded_files:

            st.session_state.current_uploaded_docs = [file.name for file in uploaded_files]

            if st.button("🚀 Process Documents", type="primary"):
                add_documents(
                    uploaded_files,
                    st.session_state.settings['chunk_size'],
                    st.session_state.settings['chunk_overlap']
                )
                st.rerun()
        
        st.divider()
        
        # Document management section
        if st.session_state.processed_files:
            st.markdown("### 📚 Loaded Documents")
            
            # Summary statistics
            total_pages = sum(s['pages'] for s in st.session_state.processed_files.values())
            total_chunks = sum(s['chunks'] for s in st.session_state.processed_files.values())
            
            st.info(f"📊 **Total:** {len(st.session_state.processed_files)} docs, {total_pages} pages, {total_chunks} chunks")
        
            st.divider()
            
            # List each document with details and delete option
            for filename, stats in st.session_state.processed_files.items():
                with st.expander(f"📄 {filename}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.caption(f"**Pages:** {stats['pages']}")
                        st.caption(f"**Chunks:** {stats['chunks']}")
                        st.caption(f"**Size:** {stats['file_size']:.2f} MB")
                        st.caption(f"**Uploaded:** {stats['upload_time']}")
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{filename}", help="Delete this document"):
                            remove_document(filename)
                        # ⭐ Update current_uploaded_docs after deletion
                        if hasattr(st.session_state, 'current_uploaded_docs'):
                            if filename in st.session_state.current_uploaded_docs:
                                st.session_state.current_uploaded_docs.remove(filename)
                        st.rerun()
            
            st.divider()
            
            # Action buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🗑️ Clear All", use_container_width=True, help="Remove all documents"):
                    st.session_state.processed_files = {}
                    st.session_state.all_documents = []
                    st.session_state.vectorstore = None
                    st.session_state.chat_history = []
                    st.session_state.current_uploaded_docs = []
                    st.success("✅ All documents cleared")
                    st.rerun()
            
            with col_b:
                if st.button("💬 Clear Chat", use_container_width=True, help="Clear conversation history"):
                    st.session_state.chat_history = []
                    st.success("✅ Chat cleared")
                    st.rerun()
            
            # Export functionality
            if st.session_state.chat_history:
                st.divider()
                markdown_export = export_chat_history()
                if markdown_export:
                    st.download_button(
                        label="📥 Export Chat",
                        data=markdown_export,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        help="Download conversation as markdown"
                    )
        else:
            st.warning("⚠️ No documents loaded")
            st.caption("👆 Upload PDFs above to get started")
        
        st.markdown("---")
        
        # Settings panel
        st.markdown("### ⚙️ Settings")
        
        with st.expander("🔧 Advanced Settings", expanded=False):
            st.markdown("**Document Processing:**")
            st.caption("⚠️ Apply before uploading documents")
            
            st.session_state.settings['chunk_size'] = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.settings['chunk_size'],
                step=100,
                help="Larger = more context, smaller = more precise"
            )
            
            st.session_state.settings['chunk_overlap'] = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.settings['chunk_overlap'],
                step=50,
                help="Higher overlap = better continuity"
            )
            
            st.divider()
            st.markdown("**Answer Generation:**")
            st.caption("⚡ Apply instantly")
            
            st.session_state.settings['num_sources'] = st.slider(
                "Number of Sources",
                min_value=1,
                max_value=10,
                value=st.session_state.settings['num_sources'],
                step=1,
                help="More sources = comprehensive answers"
            )
            
            st.session_state.settings['temperature'] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings['temperature'],
                step=0.1,
                help="0 = factual, 1 = creative"
            )
        
        st.markdown("---")
        
        # Quick guide
        st.markdown("""
        ### 📖 Quick Guide
        
        1️⃣ Upload PDF files  
        2️⃣ Click "Process Documents"  
        3️⃣ Ask questions or use suggestions  
        4️⃣ Check analytics tab!
        
        💡 *Searches across all documents*
        """)

    # ========================================================================
    # MAIN CHAT AREA
    # ========================================================================
    
    if st.session_state.vectorstore is None:
        # Empty state with helpful information
        st.markdown("""
            <div style='text-align: center; padding: 40px; background-color: white; 
                        border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <h2 style='color: #6b7280; margin-bottom: 20px;'>👈 Get Started</h2>
                <p style='font-size: 16px; color: #9ca3af;'>
                    Upload one or more PDF documents in the sidebar to begin!
                </p>
                <br>
                <p style='font-size: 14px; color: #d1d5db;'>
                    ✨ Supported: Research papers, textbooks, reports, documentation
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Display conversation history
        for i, chat_item in enumerate(st.session_state.chat_history):
            # Handle different chat item formats
            if len(chat_item) == 2:
                q, a = chat_item
                sources = []
            else:
                q, a, sources = chat_item
            
            # User message
            with st.chat_message("user"):
                st.write(q)
            
            # Assistant response
            with st.chat_message("assistant"):
                st.write(a)
                
                # Display sources if available
                if sources:
                    st.divider()
                    st.caption("📚 **Sources:**")
                    for j, source in enumerate(sources):
                        doc_name = source.get('document', 'Unknown')
                        with st.expander(f"📄 Source {j+1} - {doc_name} (Page {source['page']})"):
                            st.text(source['content'])
        
        # Suggested questions (only when chat is empty)
        if not st.session_state.chat_history:
            st.markdown("### 💡 Suggested Questions")
            st.caption("Click a question to ask it, or type your own below:")
            
            with st.spinner("🤖 Generating intelligent questions..."):
                try:
                    # ⭐ Pass the current documents list
                    current_docs = getattr(st.session_state, 'current_uploaded_docs', None)
                    suggested_questions = generate_suggested_questions(
                        st.session_state.vectorstore,
                        num_questions=5,
                        current_documents=current_docs  # ⭐ THIS IS THE KEY FIX
                    )
                except Exception as e:
                    st.error(f"Failed to generate suggestions: {str(e)}")
                    suggested_questions = []
            
            # Display as clickable buttons
            for idx, question_text in enumerate(suggested_questions):
                if st.button(
                    f"💬 {question_text}", 
                    key=f"suggested_{idx}",
                    use_container_width=True
                ):
                    # Store pending question to ask on rerun
                    st.session_state.pending_question = question_text
                    st.rerun()
            
            st.divider()
        
        # Chat input box
        question = st.chat_input("Ask a question about your documents...")
        
        # Handle question from suggestion button click
        if 'pending_question' in st.session_state:
            question = st.session_state.pending_question
            del st.session_state.pending_question
        
        # Process question
        if question:
            # Display user question
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and display answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer, sources, response_time, error = ask_question(
                        st.session_state.vectorstore,
                        question,
                        st.session_state.settings['num_sources'],
                        st.session_state.settings['temperature']
                    )
                    
                    if error:
                        # Display error with helpful message
                        st.error(f"❌ Error: {error}")
                        st.caption("💡 Try rephrasing your question or adjusting settings in the sidebar")
                        answer = f"Error: {error}"
                        sources = []
                    else:
                        # Display successful answer
                        st.write(answer)
                        
                        # Display sources with metadata
                        st.divider()
                        st.caption(f"📚 **Sources:** {len(sources)} | ⏱️ Response: {response_time:.2f}s")
                        
                        for i, source in enumerate(sources):
                            doc_name = source.get('document', 'Unknown')
                            with st.expander(f"📄 Source {i+1} - {doc_name} (Page {source['page']})"):
                                st.text(source['content'])
            
            # Save to chat history
            st.session_state.chat_history.append((question, answer, sources))