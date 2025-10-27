/**
 * Hook: useHover
 *
 * Provides hover state tracking with animation support. Returns boolean
 * hover state and CSS transition properties for smooth interactions.
 *
 * Usage:
 * const { isHovered, hoverProps, style } = useHover({ scale: true });
 * <button {...hoverProps} style={style}>Hover me</button>
 */

import { useState } from 'react';
import { timing } from '../../tokens/timing';
import { useMediaQuery } from '../../../hooks/useMediaQuery';

export interface UseHoverOptions {
  scale?: boolean;        // Scale up on hover (1 → 1.05)
  lift?: boolean;         // Lift up on hover (translateY -2px)
  shadow?: boolean;       // Add shadow on hover
  duration?: keyof typeof timing.durations;
}

export function useHover(options: UseHoverOptions = {}) {
  const {
    scale = false,
    lift = false,
    shadow = false,
    duration = 'fast',
  } = options;

  const [isHovered, setIsHovered] = useState(false);

  // Respect user's accessibility preferences
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');

  const hoverProps = {
    onMouseEnter: () => setIsHovered(true),
    onMouseLeave: () => setIsHovered(false),
  };

  // Calculate transform based on options
  const getTransform = () => {
    if (prefersReducedMotion || !isHovered) return 'none';

    const transforms: string[] = [];

    if (scale) transforms.push('scale(1.05)');
    if (lift) transforms.push('translateY(-2px)');

    return transforms.length > 0 ? transforms.join(' ') : 'none';
  };

  const style = {
    transform: getTransform(),
    boxShadow: shadow && isHovered ? '0 4px 6px -1px rgba(0, 0, 0, 0.1)' : undefined,
    transition: prefersReducedMotion
      ? 'none'
      : `transform ${timing.durations[duration]} ${timing.easings.easeOut}, box-shadow ${timing.durations[duration]} ${timing.easings.easeOut}`,
  };

  return { isHovered, hoverProps, style };
}
