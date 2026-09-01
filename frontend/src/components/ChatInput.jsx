import { useEffect, useRef, useState } from 'react'

/**
 * Text input with a Send button. Submits on Enter (without Shift) or on click.
 * Disabled while a request is in flight.
 */
export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  // Auto-grow the textarea up to a few lines.
  useEffect(() => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 120)}px`
    }
  }, [text])

  const handleSubmit = () => {
    const value = text.trim()
    if (!value || disabled) return
    onSend(value)
    setText('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        placeholder="Ask a question about the RTI Act, case law, or procedures…"
        className="max-h-[120px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed text-slate-800 outline-none placeholder:text-slate-400"
      />
      <button
        type="button"
        onClick={handleSubmit}
        disabled={disabled || !text.trim()}
        className="inline-flex h-10 shrink-0 items-center gap-2 rounded-xl bg-[#1e4b5e] px-4 text-sm font-semibold text-white transition hover:bg-[#163a49] disabled:cursor-not-allowed disabled:opacity-40"
      >
        Send
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      </button>
    </div>
  )
}
