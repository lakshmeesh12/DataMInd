import { useEffect, useState } from 'react';
import { Bot } from 'lucide-react';
import { getLoadingMessage } from '@/lib/mockApi';

const TypingIndicator = () => {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => prev + 1);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const currentMessage = getLoadingMessage(messageIndex);

  return (
    <div className="flex gap-2.5 animate-fade-in">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted">
        <Bot className="h-3.5 w-3.5 text-muted-foreground" />
      </div>

      <div className="flex items-center gap-2 rounded-lg rounded-tl-sm bg-muted px-3 py-2">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground animate-pulse">{currentMessage}</span>
          <div className="flex gap-0.5">
            <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-typing" style={{ animationDelay: '0ms' }} />
            <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-typing" style={{ animationDelay: '200ms' }} />
            <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-typing" style={{ animationDelay: '400ms' }} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
