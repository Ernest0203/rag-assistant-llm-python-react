# RAG Assistant

AI-powered document assistant. Upload PDF, DOCX, TXT, or CSV files and ask questions about their content using LLM.

## Stack

- **Backend**: Python, FastAPI, LangChain, ChromaDB, Groq API
- **Frontend**: React, Vite, Axios
- **Automation**: n8n

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-assistant
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows (Git Bash)
source venv/Scripts/activate

# Mac / Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file inside the `backend` folder:

```
GROQ_API_KEY=your_api_key_here
```

Get your free API key at: https://console.groq.com

Start the server:

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

### 4. n8n (optional, for automation)

```bash
# Windows
N8N_RESTRICT_FILE_ACCESS_TO=C:\Users\<you>\watch n8n start

# Mac / Linux
N8N_RESTRICT_FILE_ACCESS_TO=/home/<you>/watch n8n start
```

n8n runs at `http://localhost:5678`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status and indexed document count |
| POST | `/ask` | Ask a question about uploaded documents |
| POST | `/upload` | Upload and index a document |

### POST /ask

```json
{
  "question": "What is the vacation policy?"
}
```

### POST /upload

Multipart form-data with a `file` field.

---

## Supported File Formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| CSV | `.csv` |

---

## How It Works

1. Uploaded documents are split into chunks and converted into vector embeddings
2. Embeddings are stored in ChromaDB (local vector database)
3. When a question is asked, the most relevant chunks are retrieved
4. Retrieved context is passed to the LLM along with the question
5. LLM generates an answer based strictly on the provided context
