/**
 * ToolCallDisplay Component - Shows tool execution details
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import type { ToolCall } from '../../types';

interface ToolCallDisplayProps {
  toolCall: ToolCall;
}

export function ToolCallDisplay({ toolCall }: ToolCallDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const headerHover = useHover({ lift: true, shadow: true });

  const statusIcon = {
    running: <Loader2 size={14} className={cn('animate-spin', 'text-accent-600 dark:text-accent-400')} />,
    completed: <CheckCircle2 size={14} className={cn('text-accent-700 dark:text-accent-300')} />,
    failed: <XCircle size={14} className={cn('text-accent-800 dark:text-accent-200')} />,
  }[toolCall.status];

  return (
    <div className={cn(
      'border rounded-lg overflow-hidden',
      'border-accent-200 dark:border-accent-800',
      'transition-all duration-300',
      'hover:shadow-md'
    )}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        {...headerHover.hoverProps}
        style={headerHover.style}
        className={cn(
          'w-full flex items-center gap-2 p-3',
          'bg-accent-50 dark:bg-accent-900',
          'hover:bg-accent-100 dark:hover:bg-accent-800',
          'transition-colors duration-150'
        )}
      >
        {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        {statusIcon}
        <span className="font-mono text-sm font-medium">{toolCall.name}</span>
        {toolCall.executionTimeMs !== undefined && (
          <span className={cn(
            'ml-auto text-xs',
            'text-accent-500 dark:text-accent-400'
          )}>
            {toolCall.executionTimeMs.toFixed(0)}ms
          </span>
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className={cn(
          'p-3 space-y-3',
          'bg-white dark:bg-accent-950',
          'animate-scale-in'
        )}>
          {/* Input */}
          <div>
            <div className={cn(
              'text-xs font-medium mb-1',
              'text-accent-500 dark:text-accent-400'
            )}>
              Input:
            </div>
            <div className={cn(
              'text-xs p-2 rounded overflow-x-auto',
              'bg-accent-50 dark:bg-accent-900',
              'text-accent-900 dark:text-accent-100',
              'font-mono'
            )}>
              {Object.entries(toolCall.input).map(([key, value]) => {
                const isExplicit = toolCall.explicitParams?.includes(key);
                return (
                  <div key={key} className="mb-1 last:mb-0">
                    <span className={cn(isExplicit && 'font-semibold')}>
                      {key}:
                    </span>{' '}
                    <span className={cn(isExplicit && 'font-semibold')}>
                      {typeof value === 'object'
                        ? JSON.stringify(value)
                        : String(value)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Result */}
          {toolCall.result !== undefined && (
            <div>
              <div className={cn(
                'text-xs font-medium mb-1',
                'text-accent-500 dark:text-accent-400'
              )}>
                Result:
              </div>
              <pre className={cn(
                'text-xs p-2 rounded overflow-x-auto max-h-64',
                'bg-accent-50 dark:bg-accent-900',
                'text-accent-900 dark:text-accent-100'
              )}>
                {typeof toolCall.result === 'string'
                  ? toolCall.result
                  : JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
