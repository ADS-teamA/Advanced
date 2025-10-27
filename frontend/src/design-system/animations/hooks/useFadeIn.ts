/**
 * Hook: useFadeIn
 *
 * Simple fade-in animation for smooth element appearances. Lighter weight
 * than useSlideIn when you only need opacity transitions.
 *
 * Usage:
 * const { style, isVisible } = useFadeIn({ duration: 'fast', delay: 100 });
 * <div style={style}>Content</div>
 */

import { useEffect, useState } from 'react';
import { timing } from '../../tokens/timing';
import { useMediaQuery } from '../../../hooks/useMediaQuery';

export interface UseFadeInOptions {
  duration?: keyof typeof timing.durations;
  delay?: number;
  disabled?: boolean;
}

export function useFadeIn(options: UseFadeInOptions = {}) {
  const {
    duration = 'fast',
    delay = 0,
    disabled = false,
  } = options;

  const [isVisible, setIsVisible] = useState(disabled);

  // Respect user's accessibility preferences
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');

  useEffect(() => {
    if (disabled || prefersReducedMotion) {
      setIsVisible(true);
      return;
    }

    const timer = setTimeout(() => {
      setIsVisible(true);
    }, delay);

    return () => clearTimeout(timer);
  }, [delay, disabled, prefersReducedMotion]);

  const style = {
    opacity: isVisible ? 1 : 0,
    transition: prefersReducedMotion
      ? 'none'
      : `opacity ${timing.durations[duration]} ${timing.easings.easeIn}`,
  };

  return { style, isVisible };
}
