// src/components/ChatMessage.tsx
import { Message } from '@/lib/api'; // <-- Import from real API
import { Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import Visualization from './charts/Visualization'; // <-- IMPORT NEW COMPONENT

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === 'user';
  const timestamp = new Date(message.timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div
      className={`flex gap-2.5 animate-fade-in ${
        isUser ? 'flex-row-reverse' : 'flex-row'
      }`}
    >
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
          isUser ? 'bg-primary' : 'bg-muted'
        }`}
      >
        {isUser ? (
          <User className="h-3.5 w-3.5 text-primary-foreground" />
        ) : (
          <Bot className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </div>

      <div className={`flex max-w-[85%] flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-lg px-3 py-2 text-sm ${
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-sm'
              : 'bg-muted text-foreground rounded-tl-sm'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          ) : (
            // --- MODIFIED ASSISTANT VIEW ---
            <div className="prose prose-sm max-w-none dark:prose-invert [&>*]:text-sm [&_strong]:font-semibold">
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-1.5 last:mb-0">{children}</p>,
                  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                  ul: ({ children }) => <ul className="mb-1.5 list-disc pl-4">{children}</ul>,
                  li: ({ children }) => <li className="mb-0.5">{children}</li>,
                  // Disable default markdown tables, we use our own component
                  table: () => null,
                  thead: () => null,
                  tbody: () => null,
                  tr: () => null,
                  td: () => null,
                  th: () => null,
                }}
              >
                {message.content}
              </ReactMarkdown>

              {/* --- ADD VISUALIZATION RENDERER --- */}
              {message.visualization && (
                <div className="not-prose mt-2">
                  <Visualization visData={message.visualization} />
                </div>
              )}
            </div>
          )}
        </div>
        <span className="mt-0.5 text-xs text-muted-foreground">{timestamp}</span>
      </div>
    </div>
  );
};

export default ChatMessage;