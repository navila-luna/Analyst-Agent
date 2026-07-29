import { useState, type FormEvent } from 'react'
import { askQuestion, type Reference } from '../api'

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  references?: Reference[]
  isError?: boolean
}

let nextId = 0

export function Chat() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed) return

    setMessages((prev) => [...prev, { id: nextId++, role: 'user', text: trimmed }])
    setQuestion('')
    setLoading(true)

    try {
      const result = await askQuestion(trimmed)
      setMessages((prev) => [
        ...prev,
        { id: nextId++, role: 'assistant', text: result.answer, references: result.references },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId++,
          role: 'assistant',
          text: 'Something went wrong asking the backend. Is it running on port 8000?',
          isError: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((message) => (
          <div key={message.id} className={`bubble-row ${message.role}`}>
            <div className={`bubble ${message.role} ${message.isError ? 'error-bubble' : ''}`}>
              <p>{message.text}</p>

              {message.references && message.references.length > 0 && (
                <div className="references">
                  <h3>References</h3>
                  <ul>
                    {message.references.map((ref) => (
                      <li key={ref.index}>
                        [{ref.index}] {ref.doc_name} (chunk {ref.chunk_index}, similarity={ref.similarity})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="bubble-row assistant">
            <div className="bubble assistant typing">Thinking...</div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your docs..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  )
}
