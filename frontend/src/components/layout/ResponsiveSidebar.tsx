/**
 * Component: ResponsiveSidebar
 *
 * Responsive sidebar with collapsible behavior:
 * - Mobile (<768px): Overlay with hamburger toggle
 * - Desktop (>=768px): Collapsible in-layout sidebar
 *
 * Usage:
 * <ResponsiveSidebar isOpen={isOpen} onToggle={setIsOpen}>
 *   <SidebarContent />
 * </ResponsiveSidebar>
 */

import { type ReactNode } from 'react';
import { X } from 'lucide-react';
import { useMediaQuery } from '../../hooks/useMediaQuery';
import { useSlideIn } from '../../design-system/animations/hooks/useSlideIn';
import { cn } from '../../design-system/utils/cn';

export interface ResponsiveSidebarProps {
  children: ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  className?: string;
}

export function ResponsiveSidebar({
  children,
  isOpen,
  onToggle,
  className,
}: ResponsiveSidebarProps) {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const { style: slideStyle } = useSlideIn({
    direction: 'left',
    duration: 'normal',
    disabled: !isMobile,
  });

  return (
    <>
      {/* Backdrop (mobile only, when open) */}
      {isMobile && isOpen && (
        <div
          onClick={onToggle}
          className={cn(
            'fixed inset-0 bg-black/50 z-40',
            'transition-opacity duration-300',
            'animate-fade-in'
          )}
          aria-label="Close sidebar"
        />
      )}

      {/* Sidebar */}
      <aside
        style={isMobile ? slideStyle : undefined}
        className={cn(
          // Base styles
          'flex flex-col h-full',
          'bg-accent-50 dark:bg-accent-900',
          'border-r border-accent-200 dark:border-accent-800',

          // Mobile: Fixed overlay (animation handled by useSlideIn inline style)
          isMobile && [
            'fixed left-0 top-0 bottom-0 z-50',
            'w-64',
          ],

          // Desktop: Collapsible in-layout
          !isMobile && [
            'relative',
            'transition-all duration-300',
            isOpen ? 'w-64' : 'w-0',
            isOpen ? 'opacity-100' : 'opacity-0',
            'overflow-hidden',
          ],

          className
        )}
      >
        {/* Mobile close button */}
        {isMobile && isOpen && (
          <button
            onClick={onToggle}
            className={cn(
              'absolute top-4 right-4 z-10',
              'p-2 rounded-lg',
              'bg-accent-200 dark:bg-accent-800',
              'hover:bg-accent-300 dark:hover:bg-accent-700',
              'transition-colors duration-150'
            )}
            aria-label="Close sidebar"
          >
            <X size={20} className="text-accent-900 dark:text-accent-100" />
          </button>
        )}

        {/* Sidebar content */}
        <div className={cn('flex-1 overflow-hidden', isOpen && 'overflow-y-auto')}>
          {children}
        </div>
      </aside>
    </>
  );
}
