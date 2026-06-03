import os
import re
import glob
import uuid
import json
import torch
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import T5Tokenizer, T5ForConditionalGeneration

# ENVIRONMENT & APP INITIALIZATION
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "wisebot-secret-key-2024")

DATA_FOLDER = "data"
T5_MODEL_PATH = "t5_finetuned_os"
HISTORY_FOLDER = "chat_history"

# Create history directory if it doesn't exist
os.makedirs(HISTORY_FOLDER, exist_ok=True)

vectorstore = None

# JSON HISTORY HELPERS
def get_history_path(sid: str) -> str:
    """Return the file path to the session's history JSON."""
    return os.path.join(HISTORY_FOLDER, f"{sid}.json")

def load_history(sid: str) -> list:
    """Load chat history from JSON. Return an empty list if it doesn't exist."""
    path = get_history_path(sid)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(sid: str, conversations: list):
    """Save the entire conversation list to a JSON file."""
    path = get_history_path(sid)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Failed to save history: {e}")

def get_or_create_session():
    """Retrieve existing session_id or generate a new one."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True  # Keep session active after browser closes
    return session["session_id"]

def get_or_create_conversation(sid: str, conv_id: str = None) -> tuple:
    """
    Return (conversations_list, conversation_dict).
    Loads from file, searches for conv_id, or creates a new conversation.
    """
    conversations = load_history(sid)

    if conv_id:
        for conv in conversations:
            if conv["id"] == conv_id:
                return conversations, conv

    # Initialize a new conversation
    new_conv = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": []
    }
    conversations.append(new_conv)
    save_history(sid, conversations)
    return conversations, new_conv

def auto_title(text: str, max_len=40) -> str:
    """Generate a dynamic title based on the user's first prompt."""
    title = text.strip()
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0] + "…"
    return title

# LOAD FINE-TUNED T5 MODEL
print("=" * 60)
print("🔄 Loading Fine-tuned T5 Engine")
print("=" * 60)

t5_tokenizer = None
t5_model = None

try:
    print(f"📂 Loading weights from: {T5_MODEL_PATH}")
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH, legacy=False)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model = t5_model.to(device)
    t5_model.eval()

    print(f"✅ T5 engine initialized successfully on {device}")
except Exception as e:
    print(f"❌ Error loading T5 engine: {e}")

print("=" * 60 + "\n")

# LOCAL T5 GENERATION FUNCTION
def ask_t5(question):
    """Generate response using the local fine-tuned T5 model."""
    if t5_tokenizer is None or t5_model is None:
        return "❌ T5 model unavailable. Please ensure weights are loaded."

    try:
        prompt = f"question: {question}"
        inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(t5_model.device)

        with torch.no_grad():
            outputs = t5_model.generate(
                input_ids=input_ids,
                max_length=40,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.0,
                length_penalty=1.0,
                early_stopping=True
            )

        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer
    except Exception as e:
        return f"Error during generation: {str(e)}"

# FAISS VECTORSTORE INITIALIZATION
def init_vectorstore():
    """Load PDFs, split text, and initialize the FAISS vector database."""
    global vectorstore
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files detected in the data directory.")
        return

    print(f"📚 Found {len(pdf_files)} PDF(s). Initializing embeddings...")
    docs = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ FAISS Vectorstore operational")

if os.path.exists(DATA_FOLDER):
    init_vectorstore()
else:
    os.makedirs(DATA_FOLDER, exist_ok=True)

# RULE-BASED LOGIC ENGINE
def get_rule_based_response(text):
    """Catch deterministic inputs before routing to generative models."""
    rules = {
        "hello": "Hello! Ask me anything about Operating Systems.",
        "hi": "Hi! How can I help you?",
        "thanks": "You're welcome! Happy to help. 😊",
        "bye": "Goodbye! Have a great day!"
    }
    text_lower = text.lower()
    for key, response in rules.items():
        if re.search(rf"\b{key}\b", text_lower):
            return response
    return None

# GROQ LLM CONFIGURATION
def get_groq():
    """Initialize Groq cloud inference engine."""
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# CITATION & CONTEXT HELPERS
def build_citations(retrieved_docs_with_scores, score_threshold=0.8) -> list[dict]:
    """
    Filter context documents based on FAISS L2 distance.
    Lower score = higher relevance. Discard chunks exceeding the threshold.
    """
    seen = set()
    citations = []

    for doc, score in retrieved_docs_with_scores:
        if score > score_threshold:
            print(f"⚠️  Skipped low-relevance chunk (score={score:.3f}): {doc.metadata.get('source','?')}")
            continue

        meta = doc.metadata or {}
        source = os.path.basename(meta.get("source", "Unknown document"))
        page = meta.get("page", None)
        page_display = int(page) + 1 if page is not None else None

        key = (source, page_display)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page_display,
                "score": round(float(score), 3)
            })

    return citations

OUT_OF_CONTEXT_PHRASES = [
    "cannot find information",
    "not in the context",
    "not mentioned in",
    "no information",
    "i don't have information",
    "outside the scope",
    "not provided in",
    "does not contain",
    "not covered in",
]

def is_out_of_context(answer: str) -> bool:
    """Check if the LLM admitted it couldn't find the answer in the context."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in OUT_OF_CONTEXT_PHRASES)

# FLASK ROUTING
@app.route("/")
def home():
    return render_template("index.html")

# History API 

@app.route("/history/list", methods=["GET"])
def history_list():
    sid = get_or_create_session()
    conversations = load_history(sid)
    convs = [{"id": c["id"], "title": c["title"]} for c in reversed(conversations)]
    return jsonify({"conversations": convs})

@app.route("/history/<conv_id>", methods=["GET"])
def history_get(conv_id):
    sid = get_or_create_session()
    conversations = load_history(sid)
    for conv in conversations:
        if conv["id"] == conv_id:
            return jsonify({"id": conv["id"], "title": conv["title"], "messages": conv["messages"]})
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/history/<conv_id>", methods=["PUT"])
def history_rename(conv_id):
    """Handle frontend request to rename a specific chat tab."""
    sid = get_or_create_session()
    data = request.json
    new_title = data.get("title", "").strip()
    
    if not new_title:
        return jsonify({"error": "Title cannot be empty"}), 400

    conversations = load_history(sid)
    for conv in conversations:
        if conv["id"] == conv_id:
            conv["title"] = new_title
            save_history(sid, conversations)
            return jsonify({"ok": True, "title": new_title})
            
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/history/<conv_id>", methods=["DELETE"])
def history_delete(conv_id):
    sid = get_or_create_session()
    conversations = load_history(sid)
    conversations = [c for c in conversations if c["id"] != conv_id]
    save_history(sid, conversations)
    return jsonify({"ok": True})

@app.route("/history/new", methods=["POST"])
def history_new():
    sid = get_or_create_session()
    conversations, conv = get_or_create_conversation(sid)
    return jsonify({"id": conv["id"], "title": conv["title"]})

# Chat API 

@app.route("/chat", methods=["POST"])
def chat():
    try:
        sid = get_or_create_session()
        data = request.json
        user_msg = data.get("message", "").strip()
        model_type = data.get("model_type", "pretrain")
        conv_id = data.get("conv_id")

        if not user_msg:
            return jsonify({"response": "Empty message.", "used_model": model_type})

        conversations, conv = get_or_create_conversation(sid, conv_id)

        if not conv["messages"]:
            conv["title"] = auto_title(user_msg)

        conv["messages"].append({"role": "user", "content": user_msg, "citations": []})
        save_history(sid, conversations)

        rule = get_rule_based_response(user_msg)
        if rule:
            conv["messages"].append({
                "role": "bot",
                "content": rule,
                "citations": [],
                "used_model": "rule-based"
            })
            save_history(sid, conversations)
            return jsonify({
                "response": rule,
                "used_model": "rule-based",
                "citations": [],
                "conv_id": conv["id"],
                "conv_title": conv["title"]
            })

        CLOSING_TEXT = (
            "\n\n✨ I hope this explanation helps! "
            "If you have any other questions about Operating Systems, "
            "feel free to ask anytime. Happy learning! 😊"
        )

        if model_type == "finetune":
            answer = ask_t5(user_msg)
            final_response = answer + CLOSING_TEXT
            conv["messages"].append({
                "role": "bot",
                "content": final_response,
                "citations": [],
                "used_model": "finetune"
            })
            save_history(sid, conversations)
            return jsonify({
                "response": final_response,
                "used_model": "finetune",
                "citations": [],
                "conv_id": conv["id"],
                "conv_title": conv["title"]
            })

        if not vectorstore:
            msg = "No PDF loaded. Please add PDF to 'data' folder."
            conv["messages"].append({
                "role": "bot",
                "content": msg,
                "citations": [],
                "used_model": "pretrain"
            })
            save_history(sid, conversations)
            return jsonify({
                "response": msg,
                "used_model": "pretrain",
                "citations": [],
                "conv_id": conv["id"]
            })

        llm = get_groq()

        prompt = ChatPromptTemplate.from_template("""
You are an expert academic assistant for Operating Systems.
Use the following pieces of context to answer the question at the end.

Follow these rules strictly:
1. Answer ONLY based on the provided context below.
2. Provide a detailed and comprehensive explanation if available.
3. Maintain a formal and educational tone.
4. If the answer is not in the context, state clearly: "I cannot find information about that in the provided documents."

Context:
{context}

Question: {question}

Answer:
""")

        chain = prompt | llm | StrOutputParser()

        retrieved_docs_with_scores = vectorstore.similarity_search_with_score(user_msg, k=4)

        citations = build_citations(retrieved_docs_with_scores, score_threshold=0.8)

        filtered_docs = [doc for doc, score in retrieved_docs_with_scores if score <= 0.8]
        context_text = "\n\n".join(d.page_content for d in filtered_docs) if filtered_docs else ""

        result = chain.invoke({"context": context_text, "question": user_msg})
        final_response = result + CLOSING_TEXT

        if is_out_of_context(result):
            citations = []

        conv["messages"].append({
            "role": "bot",
            "content": final_response,
            "citations": citations,
            "used_model": "pretrain"
        })
        save_history(sid, conversations)

        return jsonify({
            "response": final_response,
            "used_model": "pretrain",
            "citations": citations,
            "conv_id": conv["id"],
            "conv_title": conv["title"]
        })

    except Exception as e:
        return jsonify({
            "response": f"System Error: {str(e)}",
            "used_model": "error",
            "citations": []
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)