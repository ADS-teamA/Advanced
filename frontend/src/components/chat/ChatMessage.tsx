/**
 * ChatMessage Component - Displays a single message (user or assistant)
 */

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { User, Bot, Clock, DollarSign, Edit2, RotateCw, Check, X } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useSlideIn } from '../../design-system/animations/hooks/useSlideIn';
import type { Message } from '../../types';
import { ToolCallDisplay } from './ToolCallDisplay';

interface ChatMessageProps {
  message: Message;
  animationDelay?: number;
  onEdit: (messageId: string, newContent: string) => void;
  onRegenerate: (messageId: string) => void;
  disabled?: boolean;
  responseDurationMs?: number; // Duration in milliseconds for assistant responses
}

export function ChatMessage({
  message,
  animationDelay = 0,
  onEdit,
  onRegenerate,
  disabled = false,
  responseDurationMs,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

  const { style: slideStyle } = useSlideIn({
    direction: 'up',
    delay: animationDelay,
    duration: 'normal',
  });

  const handleEdit = () => {
    setIsEditing(true);
    setEditedContent(message.content);
  };

  const handleSaveEdit = () => {
    if (editedContent.trim() && editedContent !== message.content) {
      onEdit(message.id, editedContent.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedContent(message.content);
  };

  const handleRegenerate = () => {
    console.log('🔄 Regenerate clicked for message:', message.id);
    console.log('🔄 onRegenerate function:', onRegenerate);
    console.log('🔄 Calling onRegenerate...');
    onRegenerate(message.id);
    console.log('🔄 onRegenerate called successfully');
  };

  return (
    <div
      style={slideStyle}
      className={cn(
        'flex gap-4 p-4',
        'transition-shadow duration-300',
        'hover:shadow-md',
        isUser
          ? 'bg-accent-50 dark:bg-accent-900/50'
          : 'bg-white dark:bg-accent-950'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full',
          'flex items-center justify-center',
          'transition-transform duration-200',
          'hover:scale-110',
          isUser
            ? 'bg-accent-700 dark:bg-accent-300 text-white dark:text-accent-900'
            : 'bg-accent-800 dark:bg-accent-200 text-accent-100 dark:text-accent-900'
        )}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Role label and actions */}
        <div className="flex items-center justify-between mb-2">
          <div className={cn(
            'text-sm font-medium',
            'text-accent-700 dark:text-accent-300'
          )}>
            {isUser ? 'You' : 'Assistant'}
          </div>

          {/* Action buttons */}
          {!disabled && !isEditing && (
            <div className="flex gap-1">
              {isUser && (
                <button
                  onClick={handleEdit}
                  className={cn(
                    'p-1.5 rounded',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title="Edit message"
                >
                  <Edit2 size={14} />
                </button>
              )}
              {!isUser && (
                <button
                  onClick={handleRegenerate}
                  className={cn(
                    'p-1.5 rounded',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title="Regenerate response"
                >
                  <RotateCw size={14} />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Message content or editor */}
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editedContent}
              onChange={(e) => setEditedContent(e.target.value)}
              className={cn(
                'w-full p-3 rounded-lg border',
                'border-accent-300 dark:border-accent-700',
                'bg-white dark:bg-accent-900',
                'text-accent-900 dark:text-accent-100',
                'focus:outline-none focus:ring-2',
                'focus:ring-accent-500 dark:focus:ring-accent-400',
                'resize-none'
              )}
              rows={4}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={handleSaveEdit}
                disabled={!editedContent.trim()}
                className={cn(
                  'px-3 py-1.5 rounded flex items-center gap-1.5',
                  'bg-accent-700 hover:bg-accent-800',
                  'dark:bg-accent-600 dark:hover:bg-accent-700',
                  'text-white text-sm font-medium',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
              >
                <Check size={14} />
                Save & Send
              </button>
              <button
                onClick={handleCancelEdit}
                className={cn(
                  'px-3 py-1.5 rounded flex items-center gap-1.5',
                  'bg-accent-200 hover:bg-accent-300',
                  'dark:bg-accent-800 dark:hover:bg-accent-700',
                  'text-accent-900 dark:text-accent-100 text-sm font-medium',
                  'transition-colors'
                )}
              >
                <X size={14} />
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="prose dark:prose-invert prose-sm max-w-none">
            {(() => {
              // Parse content and insert tool call displays inline
              const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;

              // Check if content has substance (after removing [Using ...] markers)
              const contentWithoutMarkers = message.content.replace(/\[Using [^\]]+\.\.\.\]\n*/g, '');
              const hasContent = contentWithoutMarkers.trim().length > 0;

              // Show error if empty response from assistant
              if (!isUser && !hasContent && !hasToolCalls) {
                return (
                  <div className={cn(
                    'px-3 py-2 rounded',
                    'bg-accent-100 dark:bg-accent-800',
                    'text-accent-700 dark:text-accent-300',
                    'text-sm italic'
                  )}>
                    ⚠️ Model didn't return any content. Try regenerating the response.
                  </div>
                );
              }

              // Split content by [Using ...] markers and render inline
              const toolMarkerRegex = /\[Using ([^\]]+)\.\.\.\]\n*/g;
              const parts: JSX.Element[] = [];
              let lastIndex = 0;
              let match;

              // Track which tool calls we've already used (by index in array)
              const usedToolCallIndices = new Set<number>();

              while ((match = toolMarkerRegex.exec(message.content)) !== null) {
                // Add text before the marker
                if (match.index > lastIndex) {
                  const textBefore = message.content.substring(lastIndex, match.index);
                  parts.push(
                    <ReactMarkdown
                      key={`text-${lastIndex}`}
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ node, inline, className, children, ...props }: any) {
                          return inline ? (
                            <code
                              className={cn(
                                'px-1 py-0.5 rounded text-sm',
                                'bg-accent-100 dark:bg-accent-800'
                              )}
                              {...props}
                            >
                              {children}
                            </code>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {textBefore}
                    </ReactMarkdown>
                  );
                }

                // Add tool call display - find first unused tool call with matching name
                const toolName = match[1];
                const toolCallIndex = message.toolCalls?.findIndex(
                  (tc, idx) => tc.name === toolName && !usedToolCallIndices.has(idx)
                );

                if (toolCallIndex !== undefined && toolCallIndex >= 0 && message.toolCalls) {
                  const toolCall = message.toolCalls[toolCallIndex];
                  usedToolCallIndices.add(toolCallIndex);

                  parts.push(
                    <div key={`tool-${toolCall.id}`} className="my-3">
                      <ToolCallDisplay toolCall={toolCall} />
                    </div>
                  );
                }

                lastIndex = match.index + match[0].length;
              }

              // Add remaining text after last marker
              if (lastIndex < message.content.length) {
                const textAfter = message.content.substring(lastIndex);
                parts.push(
                  <ReactMarkdown
                    key={`text-${lastIndex}`}
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                    components={{
                      code({ node, inline, className, children, ...props }: any) {
                        return inline ? (
                          <code
                            className={cn(
                              'px-1 py-0.5 rounded text-sm',
                              'bg-accent-100 dark:bg-accent-800'
                            )}
                            {...props}
                          >
                            {children}
                          </code>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {textAfter}
                  </ReactMarkdown>
                );
              }

              // If no markers found, render as before
              if (parts.length === 0) {
                return (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                    components={{
                      code({ node, inline, className, children, ...props }: any) {
                        return inline ? (
                          <code
                            className={cn(
                              'px-1 py-0.5 rounded text-sm',
                              'bg-accent-100 dark:bg-accent-800'
                            )}
                            {...props}
                          >
                            {children}
                          </code>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                );
              }

              return <>{parts}</>;
            })()}
          </div>
        )}

        {/* Cost information */}
        {message.cost && (
          <div className={cn(
            'mt-3 flex items-center gap-4 text-xs',
            'text-accent-500 dark:text-accent-400'
          )}>
            <span className="flex items-center gap-1">
              <DollarSign size={12} />
              ${message.cost.totalCost.toFixed(4)}
            </span>
            <span>
              {message.cost.inputTokens.toLocaleString()} in /{' '}
              {message.cost.outputTokens.toLocaleString()} out
            </span>
            {message.cost.cachedTokens > 0 && (
              <span className={cn(
                'text-accent-600 dark:text-accent-400'
              )}>
                {message.cost.cachedTokens.toLocaleString()} cached
              </span>
            )}
            {/* Show tool usage count */}
            {message.toolCalls && message.toolCalls.length > 0 && (
              <span className={cn(
                'text-accent-600 dark:text-accent-400'
              )}>
                tools used: {message.toolCalls.length}
              </span>
            )}
          </div>
        )}

        {/* Timestamp or Response Duration */}
        <div className={cn(
          'mt-2 flex items-center gap-1 text-xs',
          'text-accent-400 dark:text-accent-500'
        )}>
          <Clock size={12} />
          {!isUser && responseDurationMs !== undefined ? (
            // Show response duration for assistant messages
            `${(responseDurationMs / 1000).toFixed(2)}s`
          ) : (
            // Show timestamp for user messages
            (() => {
              const date = message.timestamp ? new Date(message.timestamp) : new Date();
              return !isNaN(date.getTime()) ? date.toLocaleTimeString() : 'Just now';
            })()
          )}
        </div>
      </div>
    </div>
  );
}
