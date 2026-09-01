/**
 * Small status badges shown alongside an answer: verification result,
 * unsupported claims, and graph-expansion (related case law) notice.
 */

export function VerifiedBadge({ verified, unsupported }) {
  if (verified === true) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-800">
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        Verified
      </span>
    )
  }
  if (verified === false || (unsupported && unsupported.length > 0)) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800">
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />
        </svg>
        Review needed
      </span>
    )
  }
  return null
}

export function GraphBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2.5 py-0.5 text-xs font-medium text-sky-800">
      <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 010 5.656l-2 2a4 4 0 01-5.656-5.657l1.1-1.1m8.485-2.485l1.1-1.1a4 4 0 00-5.657-5.657l-2 2a4 4 0 000 5.657" />
      </svg>
      Related case law found
    </span>
  )
}
