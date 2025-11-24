"""
Document Q&A Chatbot
====================
A multi-document question-answering system using LangChain, OpenAI, and FAISS.

Features:
- Upload and process multiple PDF documents
- Ask questions across all documents with source citations
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

load_dotenv()

MAX_FILE_SIZE_MB = 50
MAX_TOTAL_FILES = 10

# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate_api_keys():
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        st.error("⚠️ **Missing API Key!** Please add OPENAI_API_KEY to your .env file")
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
        st.stop()
    
    if not openai_key.startswith('sk-'):
        st.error("⚠️ **Invalid API Key Format!** OpenAI keys should start with 'sk-'")
        st.stop()

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
    st.markdown("""
        <style>
        .main { background-color: #f5f7fa; }
        h1 { color: #1e3a8a; font-weight: 700; padding-bottom: 10px; border-bottom: 3px solid #3b82f6; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 2px solid #e5e7eb; }
        [data-testid="stMetricValue"] { font-size: 24px; font-weight: 600; color: #1e40af; }
        .stChatMessage { background-color: #ffffff; border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.3s; }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stDownloadButton > button { background-color: #10b981; color: white; border: none; }
        .stDownloadButton > button:hover { background-color: #059669; }
        .streamlit-expanderHeader { background-color: #f3f4f6; border-radius: 8px; font-weight: 600; }
        .stAlert { border-radius: 10px; border-left: 4px solid #3b82f6; }
        hr { margin: 20px 0; border: none; border-top: 2px solid #e5e7eb; }
        [data-testid="stFileUploader"] { background-color: #f9fafb; border: 2px dashed #d1d5db; border-radius: 10px; padding: 20px; }
        .stSuccess { background-color: #d1fae5; color: #065f46; border-radius: 8px; }
        .stWarning { background-color: #fef3c7; color: #92400e; border-radius: 8px; }
        .stError { background-color: #fee2e2; color: #991b1b; border-radius: 8px; }
        .analytics-card { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin: 10px 0; }
        .stProgress > div > div { background-color: #3b82f6; }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_questions': 0,
        'response_times': [],
        'documents_referenced': Counter(),
        'topics_keywords': [],
        'question_history': []
    }
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'num_sources': 3,
        'temperature': 0.0
    }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_embeddings():
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return embeddings
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {str(e)}")
        st.stop()

embeddings = initialize_embeddings()

def get_llm(temperature):
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=temperature, max_tokens=1024)
        return llm
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None

# ============================================================================
# ANALYTICS TRACKING
# ============================================================================

def track_analytics(question, response_time, sources):
    st.session_state.analytics['total_questions'] += 1
    st.session_state.analytics['response_times'].append(response_time)
    st.session_state.analytics['question_history'].append({
        'question': question,
        'timestamp': datetime.now(),
        'response_time': response_time
    })
    
    for source in sources:
        doc_name = source.get('document', 'Unknown')
        st.session_state.analytics['documents_referenced'][doc_name] += 1
    
    keywords = [word.lower() for word in question.split() if len(word) > 4]
    st.session_state.analytics['topics_keywords'].extend(keywords)

# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def export_chat_history():
    if not st.session_state.chat_history:
        return None
    
    document_list = ", ".join(st.session_state.processed_files.keys())
    
    markdown_content = f"""# Document Q&A Chat History

**Documents:** {document_list}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Questions:** {st.session_state.analytics['total_questions']}

---

"""
    
    for i, chat_item in enumerate(st.session_state.chat_history):
        if len(chat_item) == 2:
            q, a = chat_item
            sources = []
        else:
            q, a, sources = chat_item
        
        markdown_content += f"""## Question {i+1}

**Q:** {q}

**A:** {a}

"""
        
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
    file_size_mb = pdf_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB"
    
    if not pdf_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, "Valid"

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_pdf(pdf_file, chunk_size, chunk_overlap):
    try:
        start_time = time.time()
        
        is_valid, message = validate_file(pdf_file)
        if not is_valid:
            raise ValueError(message)
        
        temp_path = f"temp_{pdf_file.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        file_size = os.path.getsize(temp_path) / (1024 * 1024)
        
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("PDF appears to be empty or corrupted")
        
        for doc in documents:
            doc.metadata['document'] = pdf_file.name
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No text content found in PDF")
        
        processing_time = time.time() - start_time
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        stats = {
            'pages': len(documents),
            'chunks': len(chunks),
            'file_size': file_size,
            'processing_time': processing_time,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return chunks, stats, None
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None, str(e)

def add_documents(uploaded_files, chunk_size, chunk_overlap):
    if len(st.session_state.processed_files) + len(uploaded_files) > MAX_TOTAL_FILES:
        st.error(f"❌ Maximum {MAX_TOTAL_FILES} documents allowed. Remove some documents first.")
        return
    
    success_count = 0
    error_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(uploaded_files):
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {pdf_file.name}")
        
        if pdf_file.name in st.session_state.processed_files:
            st.warning(f"⚠️ {pdf_file.name} already loaded. Skipping.")
            continue
        
        chunks, stats, error = process_pdf(pdf_file, chunk_size, chunk_overlap)
        
        if error:
            st.error(f"❌ Failed: {pdf_file.name} - {error}")
            error_count += 1
        else:
            st.session_state.all_documents.extend(chunks)
            st.session_state.processed_files[pdf_file.name] = stats
            st.success(f"✅ {pdf_file.name}: {stats['pages']} pages, {stats['chunks']} chunks")
            success_count += 1
    
    progress_bar.empty()
    status_text.empty()
    
    if success_count > 0:
        st.success(f"🎉 Successfully processed {success_count} document(s)!")
        rebuild_vectorstore()
    
    if error_count > 0:
        st.warning(f"⚠️ {error_count} document(s) failed to process.")

def rebuild_vectorstore():
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
    try:
        if filename in st.session_state.processed_files:
            del st.session_state.processed_files[filename]
        
        st.session_state.all_documents = [
            doc for doc in st.session_state.all_documents 
            if doc.metadata.get('document') != filename
        ]
        
        rebuild_vectorstore()
        st.session_state.chat_history = []
        st.success(f"✅ Removed {filename}")
        
    except Exception as e:
        st.error(f"❌ Failed to remove document: {str(e)}")

# ============================================================================
# QUESTION ANSWERING
# ============================================================================

def ask_question(vectorstore, question, num_sources, temperature):
    start_time = time.time()
    
    try:
        if not question or len(question.strip()) < 3:
            raise ValueError("Please enter a valid question (at least 3 characters)")
        
        docs = vectorstore.similarity_search(question, k=num_sources)
        
        if not docs:
            raise ValueError("No relevant information found in documents")
        
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get('page', 'Unknown')
            doc_name = doc.metadata.get('document', 'Unknown')
            
            context_parts.append(
                f"Source {i+1} from {doc_name} (Page {page_num + 1}):\n{doc.page_content}"
            )
            
            sources.append({
                'page': page_num + 1,
                'content': doc.page_content[:200] + "...",
                'document': doc_name
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context from the documents, please answer the question.
If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        llm = get_llm(temperature)
        if not llm:
            raise Exception("Failed to initialize language model")
        
        response = llm.invoke(prompt)
        response_time = time.time() - start_time
        
        track_analytics(question, response_time, sources)
        
        return response.content, sources, response_time, None
        
    except Exception as e:
        response_time = time.time() - start_time
        return None, [], response_time, str(e)

# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

def show_analytics_dashboard():
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Track your usage patterns and insights")
    
    analytics = st.session_state.analytics
    
    if analytics['total_questions'] == 0:
        st.info("📭 No questions asked yet. Start chatting to see analytics!")
        st.markdown("##### 📈 Analytics Preview:")
        st.caption("- Total questions asked")
        st.caption("- Average response time")
        st.caption("- Most referenced documents")
        st.caption("- Popular topics and keywords")
        st.caption("- Recent activity timeline")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Questions", analytics['total_questions'])
    
    with col2:
        if analytics['response_times']:
            avg_time = sum(analytics['response_times']) / len(analytics['response_times'])
            st.metric("⏱️ Avg Response", f"{avg_time:.2f}s")
        else:
            st.metric("⏱️ Avg Response", "N/A")
    
    with col3:
        if analytics['documents_referenced']:
            most_used = analytics['documents_referenced'].most_common(1)[0]
            display_name = most_used[0][:20] + "..." if len(most_used[0]) > 20 else most_used[0]
            st.metric("📄 Most Referenced", display_name, f"{most_used[1]} refs")
        else:
            st.metric("📄 Most Referenced", "N/A")
    
    with col4:
        st.metric("📚 Total Documents", len(st.session_state.processed_files))
    
    st.divider()
    
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
    
    if analytics['topics_keywords']:
        st.markdown("#### 🔑 Top Keywords")
        try:
            keyword_counts = Counter(analytics['topics_keywords'])
            top_keywords = keyword_counts.most_common(10)
            
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
    
    st.markdown("#### 🕐 Recent Activity")
    try:
        recent = analytics['question_history'][-5:]
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

st.title("📚 Document Q&A Chatbot")
st.markdown("""
    <p style='font-size: 18px; color: #6b7280; margin-top: -10px;'>
        Upload multiple PDFs and get AI-powered answers with source citations
    </p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

with tab2:
    show_analytics_dashboard()

with tab1:
    st.divider()
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### 📁 Upload Documents")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help=f"Max {MAX_FILE_SIZE_MB}MB per file, {MAX_TOTAL_FILES} files total"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                add_documents(
                    uploaded_files,
                    st.session_state.settings['chunk_size'],
                    st.session_state.settings['chunk_overlap']
                )
        
        st.divider()
        
        if st.session_state.processed_files:
            st.markdown("### 📚 Loaded Documents")
            
            total_pages = sum(s['pages'] for s in st.session_state.processed_files.values())
            total_chunks = sum(s['chunks'] for s in st.session_state.processed_files.values())
            
            st.info(f"📊 **Total:** {len(st.session_state.processed_files)} docs, {total_pages} pages, {total_chunks} chunks")
            
            st.divider()
            
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
                            st.rerun()
            
            st.divider()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🗑️ Clear All", use_container_width=True, help="Remove all documents"):
                    st.session_state.processed_files = {}
                    st.session_state.all_documents = []
                    st.session_state.vectorstore = None
                    st.session_state.chat_history = []
                    st.success("✅ All documents cleared")
                    st.rerun()
            
            with col_b:
                if st.button("💬 Clear Chat", use_container_width=True, help="Clear conversation history"):
                    st.session_state.chat_history = []
                    st.success("✅ Chat cleared")
                    st.rerun()
            
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
        
        st.markdown("### ⚙️ Settings")
        
        with st.expander("🔧 Advanced Settings", expanded=False):
            st.markdown("**Document Processing:**")
            st.caption("⚠️ Apply before uploading documents")
            
            st.session_state.settings['chunk_size'] = st.slider(
                "Chunk Size", 500, 2000, st.session_state.settings['chunk_size'], 100,
                help="Larger = more context, smaller = more precise"
            )
            
            st.session_state.settings['chunk_overlap'] = st.slider(
                "Chunk Overlap", 0, 500, st.session_state.settings['chunk_overlap'], 50,
                help="Higher overlap = better continuity"
            )
            
            st.divider()
            st.markdown("**Answer Generation:**")
            st.caption("⚡ Apply instantly")
            
            st.session_state.settings['num_sources'] = st.slider(
                "Number of Sources", 1, 10, st.session_state.settings['num_sources'], 1,
                help="More sources = comprehensive answers"
            )
            
            st.session_state.settings['temperature'] = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.settings['temperature'], 0.1,
                help="0 = factual, 1 = creative"
            )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📖 Quick Guide
        
        1️⃣ Upload PDF files  
        2️⃣ Click "Process Documents"  
        3️⃣ Ask questions in the chat  
        4️⃣ Check analytics tab!
        
        💡 *Searches across all documents*
        """)

    # ========================================================================
    # MAIN CHAT AREA
    # ========================================================================
    
    if st.session_state.vectorstore is None:
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
            if len(chat_item) == 2:
                q, a = chat_item
                sources = []
            else:
                q, a, sources = chat_item
            
            with st.chat_message("user"):
                st.write(q)
            
            with st.chat_message("assistant"):
                st.write(a)
                
                if sources:
                    st.divider()
                    st.caption("📚 **Sources:**")
                    for j, source in enumerate(sources):
                        doc_name = source.get('document', 'Unknown')
                        with st.expander(f"📄 Source {j+1} - {doc_name} (Page {source['page']})"):
                            st.text(source['content'])
        
        # Chat input box
        question = st.chat_input("Ask a question about your documents...")
        
        # Process question
        if question:
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer, sources, response_time, error = ask_question(
                        st.session_state.vectorstore,
                        question,
                        st.session_state.settings['num_sources'],
                        st.session_state.settings['temperature']
                    )
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                        st.caption("💡 Try rephrasing your question or adjusting settings in the sidebar")
                        answer = f"Error: {error}"
                        sources = []
                    else:
                        st.write(answer)
                        st.divider()
                        st.caption(f"📚 **Sources:** {len(sources)} | ⏱️ Response: {response_time:.2f}s")
                        
                        for i, source in enumerate(sources):
                            doc_name = source.get('document', 'Unknown')
                            with st.expander(f"📄 Source {i+1} - {doc_name} (Page {source['page']})"):
                                st.text(source['content'])
            
            st.session_state.chat_history.append((question, answer, sources))