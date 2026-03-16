import { useState, useEffect, useRef } from "react"
import axios from "axios"

const API_URL = import.meta.env.VITE_API_URL

export default function App() {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [documents, setDocuments] = useState([])
  const [showDocs, setShowDocs] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const fetchDocuments = async () => {
    try {
      const { data } = await axios.get(`${API_URL}/documents`)
      setDocuments(data.documents)
    } catch (e) {
      console.error(e)
    }
  }

  const handleAsk = async () => {
    if (!question.trim() || loading) return

    const userMessage = { role: "user", text: question }
    setMessages(prev => [...prev, userMessage])
    setQuestion("")
    setLoading(true)

    const history = messages.filter(m => m.role === "user" || m.role === "assistant")

    try {
      const response = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history })
      })

      // Добавляем пустое сообщение ассистента — будем наполнять по мере стриминга
      setMessages(prev => [...prev, { role: "assistant", text: "", sources: [] }])

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split("\n").filter(line => line.startsWith("data: "))

        for (const line of lines) {
          const data = JSON.parse(line.replace("data: ", ""))

          if (data.type === "sources") {
            // Обновляем sources последнего сообщения
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1].sources = data.sources
              return updated
            })
          }

          if (data.type === "token") {
            // Добавляем токен к последнему сообщению
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                text: updated[updated.length - 1].text + data.token
              }
              return updated
            })
          }

          if (data.type === "done") {
            setLoading(false)
          }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: "error", text: "Stream error" }])
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.ctrlKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)
    setUploading(true)

    try {
      const { data } = await axios.post(`${API_URL}/upload`, formData)
      setMessages(prev => [...prev, { role: "system", text: `✅ ${data.message}` }])
      fetchDocuments()
    } catch (err) {
      const detail = err.response?.data?.detail || "Upload error"
      setMessages(prev => [...prev, { role: "error", text: `❌ ${detail}` }])
    } finally {
      setUploading(false)
      e.target.value = ""
    }
  }

  const handleDelete = async (filename) => {
    try {
      await axios.delete(`${API_URL}/documents/${encodeURIComponent(filename)}`)
      setMessages(prev => [...prev, { role: "system", text: `🗑️ Deleted: ${filename}` }])
      fetchDocuments()
    } catch (err) {
      const detail = err.response?.data?.detail || "Delete error"
      setMessages(prev => [...prev, { role: "error", text: `❌ ${detail}` }])
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>📄 RAG Assistant</h1>
        <div style={styles.headerActions}>
          <button style={styles.docsBtn} onClick={() => setShowDocs(!showDocs)}>
            📚 Docs ({documents.length})
          </button>
          <label style={styles.uploadBtn}>
            {uploading ? "Uploading..." : "📎 Upload"}
            <input
              type="file"
              accept=".pdf,.txt,.docx,.csv"
              onChange={handleUpload}
              style={{ display: "none" }}
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      {showDocs && (
        <div style={styles.docsList}>
          {documents.length === 0 && (
            <div style={styles.noDocuments}>No documents uploaded yet</div>
          )}
          {documents.map((doc, i) => (
            <div key={i} style={styles.docItem}>
              <span style={styles.docName}>📄 {doc}</span>
              <button style={styles.deleteBtn} onClick={() => handleDelete(doc)}>🗑️</button>
            </div>
          ))}
        </div>
      )}

      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>Upload a document and ask a question</div>
        )}
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.message,
            ...(msg.role === "user" ? styles.userMsg : {}),
            ...(msg.role === "assistant" ? styles.assistantMsg : {}),
            ...(msg.role === "error" ? styles.errorMsg : {}),
            ...(msg.role === "system" ? styles.systemMsg : {}),
          }}>
            {msg.role === "user" && <span style={styles.label}>You</span>}
            {msg.role === "assistant" && <span style={styles.label}>Assistant</span>}
            <p style={styles.msgText}>{msg.text}</p>
            {msg.sources && msg.sources.length > 0 && (
              <div style={styles.sources}>
                📎 {msg.sources.join(", ")}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ ...styles.message, ...styles.assistantMsg }}>
            <span style={styles.label}>Assistant</span>
            <p style={styles.msgText}>Thinking...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputArea}>
        <textarea
          style={styles.textarea}
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question... (Enter to send, Ctrl+Enter for new line)"
          rows={3}
          disabled={loading}
        />
        <button
          style={{ ...styles.sendBtn, opacity: loading ? 0.6 : 1 }}
          onClick={handleAsk}
          disabled={loading}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    fontFamily: "sans-serif",
    padding: "16px",
    boxSizing: "border-box",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "12px",
  },
  title: { margin: 0, fontSize: "20px" },
  headerActions: { display: "flex", gap: "8px" },
  uploadBtn: {
    background: "#4f46e5",
    color: "white",
    padding: "8px 16px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
  },
  docsBtn: {
    background: "#f3f4f6",
    border: "1px solid #d1d5db",
    padding: "8px 16px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
  },
  docsList: {
    background: "#f9fafb",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    padding: "8px",
    marginBottom: "12px",
    maxHeight: "160px",
    overflowY: "auto",
  },
  noDocuments: { color: "#9ca3af", fontSize: "13px", padding: "4px 8px" },
  docItem: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "6px 8px",
    borderRadius: "6px",
    background: "white",
    marginBottom: "4px",
    border: "1px solid #e5e7eb",
  },
  docName: { fontSize: "13px", color: "#374151" },
  deleteBtn: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "14px",
    padding: "2px 6px",
    borderRadius: "4px",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    marginBottom: "16px",
  },
  empty: { textAlign: "center", color: "#9ca3af", marginTop: "40px" },
  message: {
    padding: "12px 16px",
    borderRadius: "12px",
    maxWidth: "80%",
  },
  userMsg: { background: "#4f46e5", color: "white", alignSelf: "flex-end" },
  assistantMsg: { background: "#f3f4f6", color: "#111", alignSelf: "flex-start" },
  errorMsg: { background: "#fee2e2", color: "#991b1b", alignSelf: "flex-start" },
  systemMsg: { background: "#d1fae5", color: "#065f46", alignSelf: "center", fontSize: "13px" },
  label: { fontSize: "11px", fontWeight: "bold", opacity: 0.6, display: "block", marginBottom: "4px" },
  msgText: { margin: 0, lineHeight: 1.5 },
  sources: {
    marginTop: "8px",
    fontSize: "11px",
    opacity: 0.6,
    borderTop: "1px solid rgba(0,0,0,0.1)",
    paddingTop: "6px",
  },
  inputArea: { display: "flex", gap: "8px", alignItems: "flex-end" },
  textarea: {
    flex: 1,
    padding: "10px",
    borderRadius: "8px",
    border: "1px solid #d1d5db",
    fontSize: "14px",
    resize: "none",
    outline: "none",
  },
  sendBtn: {
    background: "#4f46e5",
    color: "white",
    border: "none",
    padding: "10px 20px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
    height: "42px",
  },
}