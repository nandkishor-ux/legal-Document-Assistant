import { useEffect, useRef, useState } from 'react'
import { ask } from './api.js'
import MessageBubble from './components/MessageBubble.jsx'
import TypingIndicator from './components/TypingIndicator.jsx'
import ChatInput from './components/ChatInput.jsx'

function App() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef(null)

  // Keep the conversation scrolled to the latest message.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, loading])

  const handleSend = async (question) => {
    const userMsg = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    try {
      const data = await ask(question)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', ...data },
      ])
    } catch (err) {
      const timedOut = err && err.name === 'AbortError'
      const reason = timedOut
        ? 'The request timed out after a couple of minutes. Please try again.'
        : err && err.message
          ? err.message
          : 'Something went wrong while reaching the assistant.'
      setMessages((prev) => [
        ...prev,
        {
          role: 'error',
          content: `I couldn’t get an answer. ${reason}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const hasConversation = messages.length > 0

  return (
    <div className="flex h-full flex-col bg-slate-50">
      {/* Header */}
      <header className="shrink-0 border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur sm:px-6">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#1e4b5e] text-lg text-white shadow-sm">
            ⚖️
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-[17px] font-semibold tracking-tight text-slate-900">
              RTI Legal Document Assistant
            </h1>
            <p className="truncate text-[13px] text-slate-500">
              Cited, verified answers from the Indian Right to Information corpus.
            </p>
          </div>
        </div>
      </header>

      {/* Conversation */}
      <main ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {!hasConversation && !loading && (
            <div className="mx-auto mt-8 max-w-md rounded-2xl border border-slate-200 bg-white px-6 py-8 text-center shadow-sm">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-emerald-50 text-2xl">
                ⚖️
              </div>
              <h2 className="text-lg font-semibold text-slate-900">
                Ask about the Right to Information
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-500">
                Try a question about the RTI Act 2005, Delhi RTI Act 2001, or
                related case law. Every answer is grounded in cited sources and
                automatically checked for accuracy.
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {[
                  'What does Section 8(1)(j) protect?',
                  'How long must a CPIO respond?',
                  'What fees apply to an RTI request?',
                ].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSend(s)}
                    className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-600 transition hover:border-emerald-300 hover:bg-emerald-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => {
            if (msg.role === 'error') {
              return (
                <div key={i} className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {msg.content}
                </div>
              )
            }
            return <MessageBubble key={i} role={msg.role} message={msg} />
          })}

          {loading && <TypingIndicator />}
        </div>
      </main>

      {/* Input */}
      <footer className="shrink-0 border-t border-slate-200 bg-white/90 px-4 py-3 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-3xl">
          <ChatInput onSend={handleSend} disabled={loading} />
          <p className="mt-1.5 text-center text-[11px] text-slate-400">
            Answers are auto-grounded in the indexed corpus and may take a few
            seconds. Not legal advice.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
