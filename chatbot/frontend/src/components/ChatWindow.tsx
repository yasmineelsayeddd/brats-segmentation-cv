import { useRef, useEffect, type ReactNode } from 'react'
import ChatMessage from './ChatMessage'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  tool_calls?: { tool_name: string; result_summary: string }[]
  files?: { file_path: string; filename: string; shape: number[] }[]
  suggestions?: string[]
}

interface Props {
  messages: Message[]
  isLoading: boolean
  onSuggestionClick: (text: string) => void
}

const SUGGESTIONS = [
  {
    icon: 'segment',
    label: 'Segment a scan',
    prompt: 'How do I segment a brain MRI scan?',
  },
  {
    icon: 'uncertainty',
    label: 'Analyze uncertainty',
    prompt: 'What does uncertainty analysis tell us about tumor boundaries?',
  },
  {
    icon: 'metrics',
    label: 'Evaluate metrics',
    prompt: 'What metrics are used to evaluate segmentation quality?',
  },
  {
    icon: 'cascade',
    label: 'Cascade detection',
    prompt: 'Explain the cascade detection approach',
  },
]

const ICONS: Record<string, ReactNode> = {
  segment: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.2 2.688a2.25 2.25 0 01-2.063 1.352H7.463a2.25 2.25 0 01-2.063-1.352L4.2 15.3m15.6 0l-1.2-2.688M4.2 15.3l1.2-2.688" />
    </svg>
  ),
  uncertainty: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  ),
  metrics: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
    </svg>
  ),
  cascade: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m0 0l-6.75-6.75M12 19.5l6.75-6.75" />
    </svg>
  ),
}

export default function ChatWindow({ messages, isLoading, onSuggestionClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [messages])

  const isEmpty = messages.length === 0

  return (
    <div ref={containerRef} className="h-full overflow-y-auto px-4 py-6">
      <div className="max-w-3xl mx-auto">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
            <div className="w-16 h-16 rounded-2xl bg-accent-muted flex items-center justify-center mb-6 shadow-soft">
              <svg className="w-8 h-8 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.2 2.688a2.25 2.25 0 01-2.063 1.352H7.463a2.25 2.25 0 01-2.063-1.352L4.2 15.3m15.6 0l-1.2-2.688M4.2 15.3l1.2-2.688" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold text-text-primary mb-1.5">Brain Tumor Analysis Assistant</h2>
            <p className="text-text-secondary text-sm max-w-md mb-8 leading-relaxed">
              Ask questions about brain tumor segmentation, run analysis tools, and get clinical insights from the BraTS model.
            </p>
            <div className="grid grid-cols-2 gap-3 w-full max-w-lg">
              {SUGGESTIONS.map(s => (
                <button
                  key={s.label}
                  onClick={() => onSuggestionClick(s.prompt)}
                  className="group px-4 py-3.5 rounded-xl border border-surface-border bg-bg-secondary hover:bg-bg-tertiary hover:border-accent/30 hover:shadow-card transition-all text-left"
                >
                  <div className="text-text-muted group-hover:text-accent mb-2 transition-colors">
                    {ICONS[s.icon]}
                  </div>
                  <span className="text-sm font-medium text-text-primary">{s.label}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} onSuggestionClick={onSuggestionClick} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-2.5 px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-accent/40 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 rounded-full bg-accent/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 rounded-full bg-accent/80 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm text-text-secondary">Analyzing...</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
