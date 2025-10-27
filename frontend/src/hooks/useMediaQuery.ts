/**
 * Hook: useMediaQuery
 *
 * Responsive media query hook for detecting viewport size, orientation,
 * and accessibility preferences.
 *
 * Usage:
 * const isMobile = useMediaQuery('(max-width: 768px)');
 * const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
 */

import { useState, useEffect } from 'react';

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    // Server-side rendering guard
    if (typeof window === 'undefined') return false;

    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    // Server-side rendering guard
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia(query);

    // Update state when media query changes
    const handleChange = (event: MediaQueryListEvent) => {
      setMatches(event.matches);
    };

    // Modern API (addEventListener)
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }

    // Legacy API fallback (addListener - deprecated but still supported)
    mediaQuery.addListener(handleChange);
    return () => mediaQuery.removeListener(handleChange);
  }, [query]);

  return matches;
}
