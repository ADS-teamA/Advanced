/**
 * Header Component - Top navigation with model selector, theme toggle, and sidebar control
 */

import { useState, useEffect } from 'react';
import { Sun, Moon, Settings, Menu } from 'lucide-react';
import { apiService } from '../../services/api';
import { cn } from '../../design-system/utils/cn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import type { Model } from '../../types';

interface HeaderProps {
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  selectedModel: string;
  onModelChange: (model: string) => void;
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export function Header({
  theme,
  onToggleTheme,
  selectedModel,
  onModelChange,
  onToggleSidebar,
  sidebarOpen,
}: HeaderProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [showModelSelector, setShowModelSelector] = useState(false);

  // Animation hooks
  const hamburgerHover = useHover({ scale: true });
  const themeHover = useHover({ scale: true });

  // Load available models on mount
  useEffect(() => {
    apiService
      .getModels()
      .then((data) => setModels(data.models))
      .catch((error) => {
        console.error('Failed to load models:', error);
      });
  }, []);

  const handleModelChange = async (modelId: string) => {
    try {
      await onModelChange(modelId);
      setShowModelSelector(false);
    } catch (error) {
      console.error('Failed to switch model:', error);
    }
  };

  const currentModel = models.find((m) => m.id === selectedModel);

  return (
    <header className={cn(
      'bg-white dark:bg-accent-900',
      'border-b border-accent-200 dark:border-accent-800',
      'px-6 py-4'
    )}>
      <div className="flex items-center justify-between">
        {/* Left side: Hamburger + Logo */}
        <div className="flex items-center gap-3">
          {/* Hamburger button */}
          <button
            onClick={onToggleSidebar}
            {...hamburgerHover.hoverProps}
            style={hamburgerHover.style}
            className={cn(
              'p-2 rounded-lg',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-150'
            )}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <Menu size={20} />
          </button>

          {/* Logo and title */}
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-8 h-8 rounded-lg',
              'bg-accent-700 dark:bg-accent-300',
              'flex items-center justify-center',
              'text-white dark:text-accent-900',
              'font-bold'
            )}>
              S2
            </div>
            <div>
              <h1 className={cn(
                'text-lg font-bold',
                'text-accent-900 dark:text-accent-100'
              )}>SUJBOT2</h1>
              <p className={cn(
                'text-xs',
                'text-accent-500 dark:text-accent-400'
              )}>
                RAG-Powered Document Assistant
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* Model selector */}
          <div className="relative">
            <button
              onClick={() => setShowModelSelector(!showModelSelector)}
              className={cn(
                'flex items-center gap-2 px-3 py-1.5 rounded-lg',
                'border border-accent-300 dark:border-accent-600',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'transition-all duration-150',
                'hover:scale-105'
              )}
            >
              <Settings size={16} />
              <span className="text-sm font-medium">
                {currentModel?.name || 'Select Model'}
              </span>
            </button>

            {/* Dropdown */}
            {showModelSelector && (
              <div className={cn(
                'absolute right-0 mt-2 w-80',
                'bg-white dark:bg-accent-900',
                'border border-accent-200 dark:border-accent-800',
                'rounded-lg shadow-lg overflow-hidden z-50',
                'animate-scale-in'
              )}>
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelChange(model.id)}
                    className={cn(
                      'w-full text-left px-4 py-3',
                      'hover:bg-accent-100 dark:hover:bg-accent-800',
                      'transition-colors duration-150',
                      'border-b border-accent-100 dark:border-accent-800',
                      'last:border-b-0',
                      selectedModel === model.id && 'bg-accent-200 dark:bg-accent-800/50'
                    )}
                  >
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className={cn(
                      'text-xs mt-0.5',
                      'text-accent-500 dark:text-accent-400'
                    )}>
                      {model.description}
                    </div>
                    <div className={cn(
                      'text-xs mt-1',
                      'text-accent-400 dark:text-accent-500'
                    )}>
                      {model.provider}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Theme toggle */}
          <button
            onClick={onToggleTheme}
            {...themeHover.hoverProps}
            style={themeHover.style}
            className={cn(
              'p-2 rounded-lg',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-150'
            )}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
        </div>
      </div>
    </header>
  );
}
