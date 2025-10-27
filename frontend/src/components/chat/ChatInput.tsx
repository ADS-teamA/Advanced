/**
 * ChatInput Component - Message input textarea with send button
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn(
      'p-4',
      'border-t border-accent-200 dark:border-accent-800'
    )}>
      <div className="max-w-4xl mx-auto flex gap-2">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={disabled}
          className={cn(
            'flex-1 resize-none rounded-lg px-4 py-3 text-sm',
            'border border-accent-300 dark:border-accent-600',
            'bg-white dark:bg-accent-900',
            'focus:outline-none focus:ring-2 focus:ring-accent-600 dark:focus:ring-accent-400',
            'focus:scale-[1.01] transition-all duration-200',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'placeholder:text-accent-400 dark:placeholder:text-accent-500'
          )}
          rows={1}
        />
        <button
          type="submit"
          disabled={disabled || !message.trim()}
          className={cn(
            'px-4 py-3 rounded-lg',
            'bg-accent-700 dark:bg-accent-300',
            'text-white dark:text-accent-900',
            'hover:bg-accent-800 dark:hover:bg-accent-400',
            'hover:scale-105 active:scale-95',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-all duration-150',
            'flex items-center justify-center'
          )}
        >
          {disabled ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
        </button>
      </div>
      <div className={cn(
        'max-w-4xl mx-auto mt-2 text-xs',
        'text-accent-500 dark:text-accent-400'
      )}>
        Press Enter to send, Shift+Enter for new line
      </div>
    </form>
  );
}
