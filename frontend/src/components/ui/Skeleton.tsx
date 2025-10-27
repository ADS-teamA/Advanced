/**
 * Component: Skeleton
 *
 * Loading placeholder with shimmer animation. Use while content is loading
 * to provide visual feedback and prevent layout shifts.
 *
 * Usage:
 * <Skeleton className="h-4 w-full" />
 * <Skeleton variant="circle" className="w-12 h-12" />
 */

import { cn } from '../../design-system/utils/cn';

export interface SkeletonProps {
  className?: string;
  variant?: 'rectangle' | 'circle' | 'text';
  shimmer?: boolean;
}

export function Skeleton({
  className,
  variant = 'rectangle',
  shimmer = true,
}: SkeletonProps) {
  const baseClasses = 'bg-accent-200 dark:bg-accent-800';

  const variantClasses = {
    rectangle: 'rounded',
    circle: 'rounded-full',
    text: 'rounded h-4',
  };

  const shimmerClasses = shimmer
    ? 'animate-shimmer bg-gradient-to-r from-transparent via-accent-300 dark:via-accent-700 to-transparent bg-[length:200%_100%]'
    : '';

  return (
    <div
      className={cn(
        baseClasses,
        variantClasses[variant],
        shimmerClasses,
        className
      )}
      role="status"
      aria-label="Loading..."
    />
  );
}

/**
 * Skeleton Variants for Common Use Cases
 */

// Text skeleton (single line)
export function SkeletonText({ className }: { className?: string }) {
  return <Skeleton variant="text" className={cn('w-full', className)} />;
}

// Avatar skeleton (circle)
export function SkeletonAvatar({ className }: { className?: string }) {
  return <Skeleton variant="circle" className={cn('w-10 h-10', className)} />;
}

// Card skeleton (rectangle with padding)
export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('p-4 space-y-3', className)}>
      <SkeletonText className="w-3/4" />
      <SkeletonText className="w-full" />
      <SkeletonText className="w-5/6" />
    </div>
  );
}
