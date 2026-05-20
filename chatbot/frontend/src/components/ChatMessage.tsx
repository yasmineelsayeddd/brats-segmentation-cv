import ReactMarkdown from 'react-markdown'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  tool_calls?: { tool_name: string; result_summary: string }[]
  files?: { file_path: string; filename: string; shape: number[] }[]
  suggestions?: string[]
}

interface Props {
  message: Message
  onSuggestionClick?: (text: string) => void
}

const TOOL_LABELS: Record<string, string> = {
  segment_scan: 'Segment MRI',
  analyze_uncertainty: 'Uncertainty Analysis',
  compute_metrics: 'Compute Metrics',
  cascade_detect: 'Cascade Detection',
  explain_findings: 'Explain Findings',
}

export default function ChatMessage({ message, onSuggestionClick }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className="max-w-[80%]">
        {isUser && message.files && message.files.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2 justify-end">
            {message.files.map((f, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-accent-muted text-xs font-medium text-accent"
              >
                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>{f.filename}</span>
                {f.shape && <span className="text-text-muted">[{f.shape.join(', ')}]</span>}
              </span>
            ))}
          </div>
        )}
        {!isUser && message.tool_calls && message.tool_calls.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {message.tool_calls.map((tc, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-accent-muted text-xs font-medium text-accent"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
                {TOOL_LABELS[tc.tool_name] || tc.tool_name}
              </span>
            ))}
          </div>
        )}
        <div
          className={`px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-accent text-white rounded-2xl rounded-br-sm shadow-elevated'
              : 'bg-bg-secondary border border-surface-border rounded-2xl rounded-bl-sm shadow-card text-text-primary'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content || 'Run analysis on uploaded scan'}</p>
          ) : (
            <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0.5 prose-code:text-accent prose-code:bg-accent-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-pre:bg-bg-tertiary prose-pre:border prose-pre:border-surface-border">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-3">
            <p className="text-xs font-medium text-text-muted mb-2">Follow-up questions</p>
            <div className="flex flex-wrap gap-2">
              {message.suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => onSuggestionClick?.(s)}
                  className="px-3 py-1.5 rounded-xl border border-surface-border bg-bg-secondary hover:bg-bg-tertiary hover:border-accent/30 transition-all text-xs text-text-secondary hover:text-text-primary text-left max-w-full"
                >
                  {i === 0 ? <span className="text-accent font-medium mr-1">1.</span> : null}
                  {i === 1 ? <span className="text-accent font-medium mr-1">2.</span> : null}
                  {i === 2 ? <span className="text-accent font-medium mr-1">3.</span> : null}
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
