# 📚 Document Q&A Chatbot

An intelligent multi-document question-answering system powered by LangChain, OpenAI GPT-4, and FAISS vector search. Upload multiple PDFs and get AI-powered answers with precise source citations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### Core Functionality
- 📄 **Multi-Document Support** - Upload and query across multiple PDF documents simultaneously
- 🔍 **Semantic Search** - Uses FAISS vector database for fast, accurate information retrieval
- 💬 **Source Citations** - Every answer includes exact page numbers and document references
- 🤖 **AI Question Suggestions** - Automatically generates relevant questions based on document content

### Analytics & Insights
- 📊 **Usage Analytics** - Track questions asked, response times, and document usage patterns
- 🔑 **Keyword Analysis** - Identify frequently queried topics
- 📈 **Visual Dashboards** - Interactive charts and metrics

### User Experience
- 🎨 **Modern UI** - Clean, professional Streamlit interface with custom styling
- ⚙️ **Customizable Settings** - Adjust chunk size, number of sources, and AI temperature
- 📥 **Export Conversations** - Download chat history as formatted markdown
- 🚀 **Real-time Processing** - Progress bars and status updates

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Streamlit |
| **LLM** | OpenAI GPT-4 |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Orchestration** | LangChain |
| **PDF Processing** | PyPDF |
| **Language** | Python 3.8+ |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Setup Instructions

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/doc-qa-chatbot.git
   cd doc-qa-chatbot
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Configure API keys**
   
   Create a `.env` file in the project root:
```bash
   OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the application**
```bash
   streamlit run streamlit_app.py
```

5. **Open your browser**
   
   Navigate to `http://localhost:8501`

---

## 📖 Usage Guide

### Quick Start

1. **Upload Documents**
   - Click "Choose PDF files" in the sidebar
   - Select one or more PDF documents
   - Click "🚀 Process Documents"

2. **Ask Questions**
   - Use the suggested questions, or
   - Type your own question in the chat input
   - Get AI-powered answers with source citations

3. **Explore Features**
   - Click sources to see exact text used
   - Adjust settings for different behaviors
   - View analytics to track your usage
   - Export conversations for later reference

### Advanced Features

#### Customizable Settings
- **Chunk Size** (500-2000): Control how documents are split
- **Chunk Overlap** (0-500): Adjust continuity between chunks
- **Number of Sources** (1-10): How many sources to use per answer
- **Temperature** (0.0-1.0): Control answer creativity vs. precision

#### Analytics Dashboard
- Track total questions asked
- Monitor average response times
- See which documents are referenced most
- Identify trending topics and keywords
- Review recent activity timeline

---

## 📂 Project Structure
```
doc-qa-chatbot/
├── streamlit_app.py        # Main application code
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── documents/             # Place PDFs here (optional)
└── data/                  # Vector store data (auto-generated)
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Application Settings

Modify these in the sidebar's Advanced Settings:

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| Chunk Size | 1000 | 500-2000 | Text chunk size in characters |
| Chunk Overlap | 200 | 0-500 | Overlap between consecutive chunks |
| Number of Sources | 3 | 1-10 | Sources retrieved per query |
| Temperature | 0.0 | 0.0-1.0 | LLM creativity (0=factual, 1=creative) |

---

## 💡 How It Works

### RAG (Retrieval-Augmented Generation) Pipeline

1. **Document Processing**
   - PDFs are loaded and split into chunks
   - Each chunk is converted to embeddings
   - Embeddings are stored in FAISS vector database

2. **Question Answering**
   - User question is converted to embedding
   - Similarity search finds relevant chunks
   - Retrieved chunks are sent to GPT-4 as context
   - GPT-4 generates answer based on context
   - Sources are displayed with page numbers

3. **Analytics Tracking**
   - Every query is logged with metadata
   - Response times are measured
   - Document usage is tracked
   - Keywords are extracted and counted

---

## 📊 Example Use Cases

- 📚 **Academic Research** - Query multiple research papers simultaneously
- 📖 **Study Aid** - Ask questions about textbook chapters
- 📄 **Document Analysis** - Analyze reports, whitepapers, or documentation
- 🔬 **Literature Review** - Compare findings across multiple papers
- 📝 **Legal/Compliance** - Search through policy documents

---

## 🎯 Key Achievements

- ✅ **Multi-document search** across unlimited PDFs
- ✅ **Source transparency** with exact page citations
- ✅ **Production-quality error handling**
- ✅ **Analytics dashboard** for usage insights
- ✅ **Intelligent question generation** using AI
- ✅ **Professional UI/UX** with modern design
- ✅ **Export functionality** for saving conversations
- ✅ **Fully customizable** processing parameters

---

## 🔮 Future Enhancements

Potential improvements for future versions:

- [ ] Support for additional file formats (DOCX, TXT, HTML)
- [ ] Conversation memory for follow-up questions
- [ ] Multi-language support
- [ ] OCR for scanned PDFs
- [ ] Integration with cloud storage (Google Drive, Dropbox)
- [ ] User authentication and session management
- [ ] Batch question processing
- [ ] Advanced search filters (by document, date, topic)

---

## 🐛 Troubleshooting

### Common Issues

**"Missing API Key" Error**
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY` is set correctly
- Check key starts with `sk-`

**"Failed to load PDF" Error**
- Ensure PDF is not corrupted
- Check file size is under 50MB
- Try re-uploading the document

**Slow Response Times**
- Reduce number of sources in settings
- Use smaller chunk sizes
- Check your internet connection

**Out of API Credits**
- Add credits at https://platform.openai.com/account/billing
- Monitor usage in OpenAI dashboard

---

## 💰 Cost Estimation

Approximate costs per usage:

| Operation | Cost |
|-----------|------|
| Document Processing (embeddings) | ~$0.01 per 50 pages |
| Question Answering (GPT-4) | ~$0.03 per 10 questions |
| **Typical Session** | **~$0.50** |

💡 **Tip**: Use temperature=0 and fewer sources to minimize costs during development.

---

## 📜 License

This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2024 Yash Jalan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Yash Jalan**
- Teaching Assistant @ Northeastern University
- Computer Science Student

📧 Contact: [Your Email]
💼 LinkedIn: [Your LinkedIn]
🐙 GitHub: [Your GitHub]

---

## 🙏 Acknowledgments

- **LangChain** - For the RAG framework
- **OpenAI** - For GPT-4 and embeddings
- **Streamlit** - For the web framework
- **Northeastern University** - For the inspiration from coursework

---

## 📸 Screenshots

### Main Interface
*Upload documents and start chatting with your PDFs*

### Source Citations
*Every answer includes exact page numbers and document references*

### Analytics Dashboard
*Track your usage patterns and document references*

### Question Suggestions
*AI-generated questions to help you explore your documents*

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `OPENAI_API_KEY` to secrets
5. Deploy!

Your app will be live at: `https://your-app.streamlit.app`

---

## 📝 Development Notes

**Built in ~15 hours** as a portfolio project demonstrating:
- Modern AI/ML application development
- Production-quality error handling
- User-centric design principles
- Full-stack development skills
- LangChain and vector database implementation

**Technologies mastered:**
- Retrieval-Augmented Generation (RAG)
- Vector embeddings and similarity search
- Large Language Model (LLM) integration
- Web application development with Streamlit
- API integration and management

---

## ⭐ Star This Repo!

If you found this project helpful, please consider giving it a star! ⭐

**Questions? Issues? Contributions?**
Feel free to open an issue or submit a pull request!

---

*Built with ❤️ and ☕ by Yash Jalan*
