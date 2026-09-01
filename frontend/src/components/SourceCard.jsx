import { useState } from 'react'

/**
 * A small, clickable source citation card. Clicking expands it to show the
 * full section / subsection / clause breakdown when more detail is available.
 */
export default function SourceCard({ source, index }) {
  const [open, setOpen] = useState(false)

  const hasDetail =
    source.section != null && source.section !== '' &&
    (source.subsection != null && source.subsection !== '' ||
      source.clause != null && source.clause !== '')
  const isExpandable = hasDetail

  return (
    <button
      type="button"
      onClick={() => setOpen((o) => !o)}
      className={`group w-full text-left rounded-lg border text-sm transition
        ${
          open
            ? 'border-emerald-300 bg-emerald-50'
            : 'border-slate-200 bg-white hover:border-emerald-300 hover:bg-emerald-50/40'
        }`}
    >
      <div className="flex items-start gap-2.5 px-3 py-2.5">
        <span className="mt-0.5 inline-flex h-5 w-8 shrink-0 items-center justify-center rounded bg-slate-800 text-[11px] font-semibold text-white">
          {index}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-medium leading-tight text-slate-800">
            {source.label || source.document}
          </div>
          {source.document && source.label !== source.document && (
            <div className="mt-0.5 truncate text-xs text-slate-500">
              {source.document}
            </div>
          )}
        </div>
        {isExpandable && (
          <svg
            className={`mt-1 h-4 w-4 shrink-0 text-slate-400 transition-transform ${
              open ? 'rotate-180' : ''
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </div>

      {open && isExpandable && (
        <dl className="grid grid-cols-3 gap-2 border-t border-emerald-200/70 px-3 py-2.5 text-xs">
          <div>
            <dt className="font-medium text-slate-400">Section</dt>
            <dd className="mt-0.5 text-slate-700">
              {source.section || '—'}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-400">Subsection</dt>
            <dd className="mt-0.5 text-slate-700">
              {source.subsection || '—'}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-400">Clause</dt>
            <dd className="mt-0.5 text-slate-700">
              {source.clause || '—'}
            </dd>
          </div>
        </dl>
      )}
    </button>
  )
}
