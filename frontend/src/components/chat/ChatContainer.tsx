/**
 * ChatContainer Component - Main chat area with messages and input
 */

import { useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { FileText } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useFadeIn } from '../../design-system/animations/hooks/useFadeIn';
import type { Conversation } from '../../types';

interface ChatContainerProps {
  conversation: Conversation | undefined;
  isStreaming: boolean;
  onSendMessage: (message: string) => void;
  onEditMessage: (messageId: string, newContent: string) => void;
  onRegenerateMessage: (messageId: string) => void;
}

export function ChatContainer({
  conversation,
  isStreaming,
  onSendMessage,
  onEditMessage,
  onRegenerateMessage,
}: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  const { style: fadeStyle } = useFadeIn({ duration: 'slow' });

  if (!conversation) {
    return (
      <div className={cn(
        'flex-1 flex items-center justify-center',
        'bg-white dark:bg-accent-950'
      )}>
        <div className="text-center max-w-md px-4" style={fadeStyle}>
          <FileText size={64} className={cn(
            'mx-auto mb-4',
            'text-accent-300 dark:text-accent-700'
          )} />
          <h2 className={cn(
            'text-2xl font-bold mb-2',
            'text-accent-800 dark:text-accent-200'
          )}>
            Welcome to SUJBOT2
          </h2>
          <p className={cn(
            'mb-6',
            'text-accent-600 dark:text-accent-400'
          )}>
            Start a new conversation by typing a message below.
          </p>
          <p className={cn(
            'text-sm',
            'text-accent-500 dark:text-accent-500'
          )}>
            This is a RAG-powered assistant for legal and technical documents.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto">
          {conversation.messages.length === 0 ? (
            <div className="flex items-center justify-center h-full py-12">
              <div className="text-center max-w-md px-4" style={fadeStyle}>
                <FileText size={48} className={cn(
                  'mx-auto mb-3',
                  'text-accent-300 dark:text-accent-700'
                )} />
                <h3 className={cn(
                  'text-lg font-semibold mb-2',
                  'text-accent-800 dark:text-accent-200'
                )}>
                  {conversation.title}
                </h3>
                <p className={cn(
                  'text-accent-600 dark:text-accent-400'
                )}>
                  Ask me anything about your documents!
                </p>
              </div>
            </div>
          ) : (
            <div className={cn(
              'divide-y',
              'divide-accent-200 dark:divide-accent-800'
            )}>
              {conversation.messages
                .filter((message) => {
                  // Show user messages always
                  if (message.role === 'user') return true;

                  // Show assistant messages with:
                  // 1. Non-empty content (after trimming), OR
                  // 2. Tool calls (even if content is empty/whitespace)
                  const hasContent = message.content && message.content.trim().length > 0;
                  const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;

                  return hasContent || hasToolCalls;
                })
                .map((message, index, filteredMessages) => {
                  // Calculate response duration for assistant messages
                  let responseDurationMs: number | undefined;

                  if (message.role === 'assistant' && index > 0) {
                    // Find the previous user message
                    const prevMessage = filteredMessages[index - 1];
                    if (prevMessage && prevMessage.role === 'user') {
                      const userTime = new Date(prevMessage.timestamp).getTime();
                      const assistantTime = new Date(message.timestamp).getTime();
                      const duration = assistantTime - userTime;

                      // Only set duration if it's positive and reasonable (> 50ms, < 5 min)
                      if (duration > 50 && duration < 300000) {
                        responseDurationMs = duration;
                      }
                    }
                  }

                  return (
                    <ChatMessage
                      key={message.id}
                      message={message}
                      animationDelay={index * 50}
                      onEdit={onEditMessage}
                      onRegenerate={onRegenerateMessage}
                      disabled={isStreaming}
                      responseDurationMs={responseDurationMs}
                    />
                  );
                })}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <ChatInput onSend={onSendMessage} disabled={isStreaming} />
    </div>
  );
}
