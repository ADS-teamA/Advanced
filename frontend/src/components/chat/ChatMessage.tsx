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
}

export function ChatMessage({
  message,
  animationDelay = 0,
  onEdit,
  onRegenerate,
  disabled = false,
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
    onRegenerate(message.id);
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
          </div>
        )}

        {/* Tool calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-4 space-y-2">
            {message.toolCalls.map((toolCall) => (
              <ToolCallDisplay key={toolCall.id} toolCall={toolCall} />
            ))}
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
          </div>
        )}

        {/* Timestamp */}
        <div className={cn(
          'mt-2 flex items-center gap-1 text-xs',
          'text-accent-400 dark:text-accent-500'
        )}>
          <Clock size={12} />
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
