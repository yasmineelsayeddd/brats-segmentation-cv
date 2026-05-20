import { useState, useRef, useEffect, useCallback } from 'react'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'

interface UploadedFile {
  file_path: string
  filename: string
  shape: number[]
  dtype: string
  size_kb: number
}

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  tool_calls?: { tool_name: string; result_summary: string }[]
  files?: { file_path: string; filename: string; shape: number[] }[]
  suggestions?: string[]
}

const API_BASE = '/api'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [sessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const uploadFile = useCallback(async (file: File): Promise<UploadedFile> => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new Error('Upload failed')
    return res.json()
  }, [])

  const sendMessage = useCallback(async (text: string, files: UploadedFile[]) => {
    const userMsg: Message = {
      role: 'user',
      content: text,
      files: files.map(f => ({ file_path: f.file_path, filename: f.filename, shape: f.shape })),
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    const allMessages = [...messages, userMsg]

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMessages.map(m => ({ role: m.role, content: m.content })),
          files: userMsg.files,
          session_id: sessionId,
        }),
      })

      const data = await res.json()

      const assistantMsg: Message = {
        role: 'assistant',
        content: data.content,
        tool_calls: data.tool_calls || [],
        suggestions: data.suggestions || [],
      }

      setMessages(prev => [...prev, assistantMsg])
    } catch {
      const errorMsg: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error connecting to the analysis server. Please try again.',
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }, [messages, sessionId])

  return (
    <div className="h-screen flex flex-col bg-bg-primary">
      <header className="flex items-center justify-between px-6 py-3 border-b border-surface-border bg-bg-secondary shadow-soft">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent-muted flex items-center justify-center">
            <svg className="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.2 2.688a2.25 2.25 0 01-2.063 1.352H7.463a2.25 2.25 0 01-2.063-1.352L4.2 15.3m15.6 0l-1.2-2.688M4.2 15.3l1.2-2.688" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary">BraTS Tumor Analysis</h1>
            <p className="text-xs text-text-secondary">AI-powered brain tumor segmentation assistant</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-accent-muted px-3 py-1.5 rounded-full">
          <span className="w-1.5 h-1.5 rounded-full bg-success" />
          <span className="text-xs font-medium text-accent">Online</span>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <ChatWindow messages={messages} isLoading={isLoading} onSuggestionClick={t => sendMessage(t, [])} />
        <div ref={messagesEndRef} />
      </main>

      <footer className="border-t border-surface-border bg-bg-secondary">
        <ChatInput onSend={sendMessage} onUpload={uploadFile} disabled={isLoading} />
      </footer>
    </div>
  )
}
