from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import sqlite3
import tempfile
import os
import json
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://rag-assistant-llm-python-react-1.onrender.com"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant. You have access to the following documents: {sources_list}

Answer ONLY based on the provided context chunks below.
If the answer is not in the context, say: "There is no information on this topic in the documents."
If the user asks which documents you have access to, list them from the context metadata.
Answer in the same language as the question.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# SQLite для метаданных и истории
def get_db():
    conn = sqlite3.connect("metadata.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            pages INTEGER,
            chunks INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

class QuestionRequest(BaseModel):
    question: str
    history: list = []

def format_history(raw_history: list):
    messages = []
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["text"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["text"]))
    return messages

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def hybrid_search(question: str) -> list:
    if vectordb._collection.count() == 0:
        return []

    # Векторный поиск
    vector_docs = vectordb.similarity_search(question, k=5)

    # BM25
    all_docs_result = vectordb.get()
    corpus = all_docs_result["documents"]
    metadatas = all_docs_result["metadatas"]

    if corpus:
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(question.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_docs = [
            type('Doc', (), {
                'page_content': corpus[i],
                'metadata': metadatas[i]
            })()
            for i in top_indices
        ]

        seen = set()
        relevant_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                relevant_docs.append(doc)
        return relevant_docs

    return vector_docs

@app.post("/ask")
async def ask(request: QuestionRequest):
    relevant_docs = hybrid_search(request.question)
    if not relevant_docs:
        raise HTTPException(status_code=400, detail="Database is empty. Please upload documents first.")

    sources = list(set(doc.metadata.get("source", "unknown") for doc in relevant_docs))
    context = format_docs(relevant_docs)
    history = format_history(request.history)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "history": history,
        "question": request.question,
        "sources_list": ", ".join(sources)
    })

    return {"answer": answer, "sources": sources}

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    relevant_docs = hybrid_search(request.question)
    if not relevant_docs:
        raise HTTPException(status_code=400, detail="Database is empty. Please upload documents first.")

    sources = list(set(doc.metadata.get("source", "unknown") for doc in relevant_docs))
    context = format_docs(relevant_docs)
    history = format_history(request.history)

    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        chain = prompt | llm
        async for chunk in chain.astream({
            "context": context,
            "history": history,
            "question": request.question,
            "sources_list": ", ".join(sources)
        }):
            if chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/upload")
async def upload(file: UploadFile):
    allowed = [".pdf", ".txt", ".docx", ".csv"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(allowed)}")

    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM documents WHERE filename = ?", (file.filename,)
    ).fetchone()
    conn.close()

    if existing:
        raise HTTPException(status_code=400, detail="File already uploaded")

    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif ext == ".csv":
        loader = CSVLoader(tmp_path, encoding="utf-8")

    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    for chunk in chunks:
        chunk.metadata["source"] = file.filename

    vectordb.add_documents(chunks)

    conn = get_db()
    conn.execute(
        "INSERT INTO documents (filename, pages, chunks) VALUES (?, ?, ?)",
        (file.filename, len(pages), len(chunks))
    )
    conn.commit()
    conn.close()

    return {"message": f"Uploaded {len(pages)} pages, {len(chunks)} chunks"}

@app.get("/documents")
async def get_documents():
    conn = get_db()
    rows = conn.execute(
        "SELECT filename, pages, chunks, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return {"documents": [dict(row) for row in rows]}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM documents WHERE filename = ?", (filename,)
    ).fetchone()

    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    result = vectordb.get(where={"source": filename})
    if result["ids"]:
        vectordb.delete(ids=result["ids"])

    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

    return {"message": f"Deleted: {filename}"}

@app.get("/history")
async def get_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT role, text, sources, created_at FROM history ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return {"history": [dict(row) for row in rows]}

@app.post("/history")
async def save_message(message: dict):
    conn = get_db()
    conn.execute(
        "INSERT INTO history (role, text, sources) VALUES (?, ?, ?)",
        (message["role"], message["text"], json.dumps(message.get("sources", [])))
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/history")
async def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents_indexed": vectordb._collection.count()
    }