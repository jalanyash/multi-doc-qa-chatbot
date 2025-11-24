import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Load environment variables
load_dotenv()

print("🚀 Initializing Document Q&A System...")

# Initialize Claude
# llm = ChatAnthropic(
#     model="claude-sonnet-4-20250514",
#     temperature=0,
#     max_tokens=1024
# )

# Initialize GPT-4
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=1024
)

# Initialize embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_documents(pdf_path):
    """Load and split PDF into chunks"""
    print(f"📄 Loading {pdf_path}...")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} pages")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create vector database from chunks"""
    print("🔍 Creating vector store (this may take a minute)...")
    
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save it locally
    vectorstore.save_local("./data/faiss_index")
    
    print("✅ Vector store created and saved!")
    return vectorstore

def load_existing_vectorstore():
    """Load previously created vector store"""
    try:
        vectorstore = FAISS.load_local(
            "./data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Loaded existing vector store!")
        return vectorstore
    except:
        return None

def ask_question(vectorstore, question):
    """Ask a question and get answer"""
    # Search for relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    
    # Combine document content
    context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    # Create prompt
    prompt = f"""Based on the following context from the document, please answer the question.
If the answer cannot be found in the context, say "I cannot answer this based on the provided document."

Context:
{context}

Question: {question}

Answer:"""
    
    # Get answer from Claude
    response = llm.invoke(prompt)
    
    return response.content

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 DOCUMENT Q&A SYSTEM")
    print("="*60 + "\n")
    
    # Check if vector store already exists
    vectorstore = load_existing_vectorstore()
    
    if vectorstore is None:
        # First time setup - load PDF
        pdf_path = "documents/test.pdf"  # ⚠️ CHANGE THIS to your PDF name
        
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF not found at {pdf_path}")
            print("Please add a PDF to the 'documents/' folder and update the filename in the code.")
            exit(1)
        
        chunks = load_documents(pdf_path)
        vectorstore = create_vector_store(chunks)
    
    print("\n" + "="*60)
    print("✅ System Ready! Ask questions about your document.")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    # Question loop
    while True:
        question = input("❓ Your Question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        print("\n🤔 Thinking...\n")
        
        try:
            answer = ask_question(vectorstore, question)
            
            print("💡 Answer:")
            print("-" * 60)
            print(answer)
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")