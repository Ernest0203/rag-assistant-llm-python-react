from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import tempfile
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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

@app.post("/ask")
async def ask(request: QuestionRequest):
    if vectordb._collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Database is empty. Please upload documents first."
        )

    # Получаем релевантные чанки через векторный поиск
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_retriever.invoke(request.question)
    
    # BM25 поиск вручную
    all_docs_result = vectordb.get()
    corpus = all_docs_result["documents"]
    metadatas = all_docs_result["metadatas"]
    
    if corpus:
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = request.question.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Берём топ-5 по BM25
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        
        bm25_docs = []
        for i in top_indices:
            bm25_docs.append(
                type('Doc', (), {
                    'page_content': corpus[i],
                    'metadata': metadatas[i]
                })()
            )
        
        # Объединяем результаты, убираем дубли
        seen = set()
        relevant_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                relevant_docs.append(doc)
    else:
        relevant_docs = vector_docs

    # Собираем citations — уникальные источники
    sources = list(set(
        doc.metadata.get("source", "unknown")
        for doc in relevant_docs
    ))

    # Формируем контекст
    context = format_docs(relevant_docs)
    history = format_history(request.history)

    # Запускаем LLM
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "history": history,
        "question": request.question,
        "sources_list": ", ".join(sources)
    })

    return {
        "answer": answer,
        "sources": sources  # Citations
    }

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    if vectordb._collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Database is empty. Please upload documents first."
        )

    # hybrid search
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_retriever.invoke(request.question)

    all_docs_result = vectordb.get()
    corpus = all_docs_result["documents"]
    metadatas = all_docs_result["metadatas"]

    if corpus:
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = request.question.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_docs = [
            type('Doc', (), {'page_content': corpus[i], 'metadata': metadatas[i]})()
            for i in top_indices
        ]
        seen = set()
        relevant_docs = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                relevant_docs.append(doc)
    else:
        relevant_docs = vector_docs

    sources = list(set(
        doc.metadata.get("source", "unknown")
        for doc in relevant_docs
    ))
    context = format_docs(relevant_docs)
    history = format_history(request.history)

    async def generate():
        # Сначала отправляем sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Стримим ответ по чанкам
        chain = prompt | llm
        async for chunk in chain.astream({
            "context": context,
            "history": history,
            "question": request.question,
            "sources_list": ", ".join(sources)
        }):
            if chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

        # Сигнал что стрим закончен
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

    existing = vectordb.get(where={"source": file.filename})
    if existing["ids"]:
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

    for chunk in chunks:
        chunk.metadata["source"] = file.filename

    vectordb.add_documents(chunks)
    os.unlink(tmp_path)

    return {"message": f"Uploaded {len(pages)} pages, {len(chunks)} chunks"}

@app.get("/documents")
async def get_documents():
    result = vectordb.get()
    sources = list(set(
        meta.get("source", "unknown")
        for meta in result["metadatas"]
    ))
    return {"documents": sources}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    result = vectordb.get(where={"source": filename})
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Document not found")

    vectordb.delete(ids=result["ids"])
    return {"message": f"Deleted: {filename}"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents_indexed": vectordb._collection.count()
    }