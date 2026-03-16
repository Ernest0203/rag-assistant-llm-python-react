from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from dotenv import load_dotenv
import tempfile
import os
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone с Inference API — эмбеддинги в облаке, без torch
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

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

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Генерируем эмбеддинги через Pinecone Inference API"""
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={"input_type": "passage"}
    )
    return [item.values for item in result]

def embed_query(query: str) -> list[float]:
    """Эмбеддинг для поискового запроса"""
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    return result[0].values

def hybrid_search(question: str) -> list:
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        return []

    # Векторный поиск
    query_vector = embed_query(question)
    vector_results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )
    vector_docs = [
        Document(
            page_content=match.metadata.get("text", ""),
            metadata={"source": match.metadata.get("source", "unknown")}
        )
        for match in vector_results.matches
    ]

    # BM25 — берём все документы для keyword поиска
    all_results = index.query(
        vector=query_vector,
        top_k=100,
        include_metadata=True
    )
    all_docs = [
        Document(
            page_content=match.metadata.get("text", ""),
            metadata={"source": match.metadata.get("source", "unknown")}
        )
        for match in all_results.matches
    ]

    if all_docs:
        corpus = [doc.page_content for doc in all_docs]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(question.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_docs = [all_docs[i] for i in top_indices]

        seen = set()
        relevant_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                relevant_docs.append(doc)
        return relevant_docs

    return vector_docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    # Проверяем дубли
    existing = index.query(
        vector=[0.0] * 1024,
        top_k=1,
        include_metadata=True,
        filter={"source": file.filename}
    )
    if existing.matches:
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

    # Генерируем эмбеддинги батчами по 50
    texts = [chunk.page_content for chunk in chunks]
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(embed_texts(batch))

    # Сохраняем в Pinecone
    vectors = [
        {
            "id": f"{file.filename}-{i}",
            "values": all_embeddings[i],
            "metadata": {
                "text": chunks[i].page_content,
                "source": file.filename
            }
        }
        for i in range(len(chunks))
    ]
    index.upsert(vectors=vectors)

    return {"message": f"Uploaded {len(pages)} pages, {len(chunks)} chunks"}

@app.get("/documents")
async def get_documents():
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        return {"documents": []}

    # Получаем уникальные источники
    result = index.query(
        vector=[0.0] * 1024,
        top_k=100,
        include_metadata=True
    )
    sources = list(set(
        match.metadata.get("source", "unknown")
        for match in result.matches
    ))
    return {"documents": sources}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    existing = index.query(
        vector=[0.0] * 1024,
        top_k=1,
        include_metadata=True,
        filter={"source": filename}
    )
    if not existing.matches:
        raise HTTPException(status_code=404, detail="Document not found")

    index.delete(filter={"source": filename})
    return {"message": f"Deleted: {filename}"}

@app.get("/health")
async def health():
    stats = index.describe_index_stats()
    return {
        "status": "ok",
        "documents_indexed": stats.total_vector_count
    }