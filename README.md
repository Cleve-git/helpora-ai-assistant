# Helpora: Enterprise Hybrid RAG Assistant

Helpora is a professional, production-ready Hybrid Retrieval-Augmented Generation (RAG) assistant designed for document-based interaction. Developed specifically for operational and structural analysis, Helpora bridges classical software logic with modern Generative AI to minimize hallucinations and deliver precision-driven context fetching.

## 🚀 Key Architecture & Features

This project implements a **Tri-Layer Hybrid Routing Architecture** to optimize compute resources and response times:

1. **Deterministic Rule-Based Engine**: Captures predefined, high-frequency queries instantly, saving API costs and compute overhead.
2. **Local Fine-Tuned Model (T5)**: Handles specialized, domain-specific generation completely offline, ensuring data privacy for sensitive queries.
3. **Cloud Foundation Model (Groq + LLaMA 3.1)**: Acts as the heavy-duty engine for complex reasoning and Document RAG. Powered by the Groq LPU inference engine for ultra-low latency responses.

### Additional Features:
- **Context Retrieval Filtering**: Employs structural threshold configurations (FAISS L2 distance) to discard low-relevance documents dynamically, eliminating AI hallucinations.
- **Persistent Session Management**: Complete chat history tracking stored locally via JSON, functioning independently for each user session.
- **SaaS-Inspired Dashboard UI**: A clean, distraction-free corporate dashboard crafted with Inter typography, persistent message tracking, and full dynamic session controls (Rename and Delete chat tabs).

## 🛠️ Tech Stack

- **Backend & API**: Python, Flask
- **LLM Orchestration**: LangChain, Groq API (LLaMA-3.1-8b-instant)
- **Local AI & Embeddings**: Hugging Face `transformers` (T5-small fine-tuned), `all-MiniLM-L6-v2`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Frontend Core**: Bootstrap 5, FontAwesome, Native Async JavaScript

## 📊 System Workflow

```text
  [ User Query ] ──► [ Hybrid Intent Router ] ──► (Matches Rule?) ──► Yes ──► [ Instant Precise Rule Response ]
                            │
                            ▼ No
                   (Model Selection)
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   [ Local T5 Engine ]           [ Groq Cloud Engine ] ◄── (Semantic Vector Search via FAISS)
             │                             │
             └──────────────┬──────────────┘
                            ▼
                     [ Clean Chat UI ]
 ```
## ⚙️ Installation & Local Setup
Clone the Repository

Bash
git clone [https://github.com/Cleve-git/helpora-ai-assistant.git](https://github.com/Cleve-git/helpora-ai-assistant.git)
cd helpora-ai-assistant
Configure Virtual Environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Core Requirements
Ensure you have generated the requirements.txt file, then run:

Bash
pip install -r requirements.txt
Environment Variables Setup
Create a .env file in the root directory and add your Groq API Key:

Plaintext
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_flask_secret_key
Load Data
Place your reference PDF documents inside the data/ folder. The system will automatically build the FAISS vector index upon startup.

Execute Backend Server

Bash
python app.py
The application will be accessible at http://localhost:5000.
