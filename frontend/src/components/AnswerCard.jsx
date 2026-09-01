import SourceCard from './SourceCard'
import { VerifiedBadge, GraphBadge } from './StatusBadges'

/**
 * Renders an assistant answer. Handles the refused (informational) case
 * separately from a normal verified/unverified answer with source cards.
 */
export default function AnswerCard({ message }) {
  const { answer, sources, verified, unsupported, graph_expansion_triggered, refused } = message

  // Refused: neutral informational message, NOT an error.
  if (refused) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
        <div className="flex items-start gap-2">
          <svg
            className="mt-0.5 h-4 w-4 shrink-0 text-slate-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p>{answer}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <VerifiedBadge verified={verified} unsupported={unsupported} />
        {graph_expansion_triggered && <GraphBadge />}
      </div>

      <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">
        {answer}
      </p>

      {Array.isArray(sources) && sources.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Sources ({sources.length})
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {sources.map((src, i) => (
              <SourceCard key={i} source={src} index={i + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
