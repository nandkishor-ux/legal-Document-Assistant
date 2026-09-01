import AnswerCard from './AnswerCard'

/**
 * A chat message bubble. User questions are right-aligned in a filled accent
 * bubble; assistant answers are left-aligned in a light bubble and may render
 * the full AnswerCard (sources, badges).
 */
export default function MessageBubble({ role, message }) {
  const isUser = role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-[#1e4b5e] px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-sm sm:max-w-[70%]">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[95%] rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3.5 shadow-sm sm:max-w-[85%]">
        <AnswerCard message={message} />
      </div>
    </div>
  )
}
