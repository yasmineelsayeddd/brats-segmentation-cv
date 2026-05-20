import { useState, useRef, KeyboardEvent, ChangeEvent } from 'react'

interface UploadedFile {
  file_path: string
  filename: string
  shape: number[]
  dtype: string
  size_kb: number
}

interface Props {
  onSend: (text: string, files: UploadedFile[]) => void
  onUpload: (file: File) => Promise<UploadedFile>
  disabled: boolean
}

export default function ChatInput({ onSend, onUpload, disabled }: Props) {
  const [value, setValue] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = () => {
    const trimmed = value.trim()
    if ((!trimmed && uploadedFiles.length === 0) || disabled) return
    onSend(trimmed, uploadedFiles)
    setValue('')
    setUploadedFiles([])
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 120) + 'px'
    }
  }

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploading(true)
    setUploadError(null)
    try {
      const info = await onUpload(file)
      setUploadedFiles(prev => [...prev, info])
    } catch {
      setUploadError('Upload failed')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const removeFile = (idx: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== idx))
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-3">
      {uploadedFiles.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {uploadedFiles.map((f, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-accent-muted text-xs font-medium text-accent border border-accent/20"
            >
              <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>{f.filename}</span>
              {f.shape && <span className="text-text-muted">[{f.shape.join(', ')}]</span>}
              <button
                onClick={() => removeFile(i)}
                className="ml-0.5 hover:text-danger transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="flex items-center gap-2 bg-bg-primary border border-surface-border rounded-2xl px-3 py-2.5 focus-within:border-accent/50 focus-within:shadow-card transition-all">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isUploading}
          className="flex-shrink-0 w-8 h-8 rounded-xl hover:bg-bg-tertiary disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center justify-center text-text-muted hover:text-accent"
          title="Upload scan (.npy, .jpg, .png)"
        >
          {isUploading ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-8m0 0l-3 3m3-3l3 3M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
            </svg>
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".npy,.jpg,.jpeg,.png"
          onChange={handleFileSelect}
          className="hidden"
        />
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask about tumor analysis..."
          rows={1}
          disabled={disabled}
          className="flex-1 bg-transparent text-text-primary text-sm placeholder:text-text-muted resize-none outline-none max-h-[120px] leading-6"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || (!value.trim() && uploadedFiles.length === 0)}
          className="flex-shrink-0 w-8 h-8 rounded-xl bg-accent hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center justify-center active:scale-95"
        >
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
          </svg>
        </button>
      </div>
      {uploadError && (
        <p className="text-xs text-danger text-center mt-1.5">{uploadError}</p>
      )}
      <p className="text-center text-xs text-text-muted mt-1.5">
        Upload MRI scans or type a question
      </p>
    </div>
  )
}
