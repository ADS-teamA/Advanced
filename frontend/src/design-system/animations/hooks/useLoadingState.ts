/**
 * Hook: useLoadingState
 *
 * Manages loading state with automatic skeleton/shimmer animation support.
 * Returns loading state and CSS properties for skeleton placeholders.
 *
 * Usage:
 * const { isLoading, skeletonStyle } = useLoadingState(fetchingData);
 * {isLoading ? <div style={skeletonStyle} /> : <Content />}
 */

import { useEffect, useState } from 'react';

export interface UseLoadingStateOptions {
  shimmer?: boolean;     // Enable shimmer animation
  minDuration?: number;  // Minimum loading duration (prevents flash)
}

export function useLoadingState(
  isLoading: boolean,
  options: UseLoadingStateOptions = {}
) {
  const {
    shimmer = true,
    minDuration = 300,
  } = options;

  const [internalLoading, setInternalLoading] = useState(isLoading);
  const [loadingStartTime, setLoadingStartTime] = useState<number | null>(null);

  useEffect(() => {
    if (isLoading) {
      // Start loading
      setInternalLoading(true);
      setLoadingStartTime(Date.now());
    } else if (loadingStartTime) {
      // End loading (with minimum duration enforcement)
      const elapsed = Date.now() - loadingStartTime;
      const remaining = Math.max(0, minDuration - elapsed);

      if (remaining > 0) {
        const timer = setTimeout(() => {
          setInternalLoading(false);
          setLoadingStartTime(null);
        }, remaining);

        return () => clearTimeout(timer);
      } else {
        setInternalLoading(false);
        setLoadingStartTime(null);
      }
    }
  }, [isLoading, loadingStartTime, minDuration]);

  // Skeleton placeholder styles
  const skeletonStyle = {
    backgroundColor: 'var(--color-surface)',
    borderRadius: '4px',
    animation: shimmer ? 'shimmer 2s infinite' : undefined,
    backgroundImage: shimmer
      ? 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)'
      : undefined,
    backgroundSize: shimmer ? '200% 100%' : undefined,
  };

  return {
    isLoading: internalLoading,
    skeletonStyle,
  };
}

// Note: Shimmer keyframes should be added to index.css:
// @keyframes shimmer {
//   0% { background-position: -200% 0; }
//   100% { background-position: 200% 0; }
// }
