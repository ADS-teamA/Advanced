/**
 * Hook: useSlideIn
 *
 * Animates elements sliding in from a specified direction with configurable
 * duration and delay. Perfect for message entrances, modal reveals, etc.
 *
 * Usage:
 * const { ref, style, isVisible } = useSlideIn({ direction: 'up', delay: 200 });
 * <div ref={ref} style={style}>Content</div>
 */

import { useEffect, useRef, useState } from 'react';
import { timing } from '../../tokens/timing';
import { useMediaQuery } from '../../../hooks/useMediaQuery';

export interface UseSlideInOptions {
  direction?: 'left' | 'right' | 'up' | 'down';
  duration?: keyof typeof timing.durations;
  delay?: number;
  disabled?: boolean;
}

export function useSlideIn(options: UseSlideInOptions = {}) {
  const {
    direction = 'up',
    duration = 'normal',
    delay = 0,
    disabled = false,
  } = options;

  const [isVisible, setIsVisible] = useState(disabled);
  const ref = useRef<HTMLElement>(null);

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

  // Calculate transform based on direction
  const getTransform = () => {
    if (prefersReducedMotion) return 'none';
    if (isVisible) return 'translate(0, 0)';

    switch (direction) {
      case 'up':
        return 'translateY(20px)';
      case 'down':
        return 'translateY(-20px)';
      case 'left':
        return 'translateX(100px)';
      case 'right':
        return 'translateX(-100px)';
      default:
        return 'translateY(20px)';
    }
  };

  const style = {
    opacity: isVisible ? 1 : 0,
    transform: getTransform(),
    transition: prefersReducedMotion
      ? 'none'
      : `opacity ${timing.durations[duration]} ${timing.easings.easeOut}, transform ${timing.durations[duration]} ${timing.easings.easeOut}`,
  };

  return { ref, style, isVisible };
}
