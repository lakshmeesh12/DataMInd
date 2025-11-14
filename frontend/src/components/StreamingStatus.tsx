// src/components/StreamingStatus.tsx
import { Bot } from 'lucide-react';
import React from 'react';

interface StreamingStatusProps {
  message: string;
}

const StreamingStatus = ({ message }: StreamingStatusProps) => {
  return (
    <div className="flex gap-2.5 animate-fade-in">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted">
        <Bot className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <div className="flex max-w-[85%] flex-col items-start">
        <div className="rounded-lg bg-muted px-3 py-2 text-sm text-foreground rounded-tl-sm">
          <p className="italic text-muted-foreground">{message}</p>
        </div>
      </div>
    </div>
  );
};

export default StreamingStatus;