/**
 * Animation Primitives
 *
 * Core animation definitions for consistent motion design throughout
 * the application. These primitives are used by animation hooks.
 */

import { timing } from '../tokens/timing';

export const animationPrimitives = {
  // Slide animations (entrance from directions)
  slideInRight: {
    from: {
      transform: 'translateX(20px)',
      opacity: 0
    },
    to: {
      transform: 'translateX(0)',
      opacity: 1
    },
    duration: timing.durations.normal,
    easing: timing.easings.easeOut,
  },

  slideInLeft: {
    from: {
      transform: 'translateX(-20px)',
      opacity: 0
    },
    to: {
      transform: 'translateX(0)',
      opacity: 1
    },
    duration: timing.durations.normal,
    easing: timing.easings.easeOut,
  },

  slideInUp: {
    from: {
      transform: 'translateY(20px)',
      opacity: 0
    },
    to: {
      transform: 'translateY(0)',
      opacity: 1
    },
    duration: timing.durations.normal,
    easing: timing.easings.easeOut,
  },

  slideInDown: {
    from: {
      transform: 'translateY(-20px)',
      opacity: 0
    },
    to: {
      transform: 'translateY(0)',
      opacity: 1
    },
    duration: timing.durations.normal,
    easing: timing.easings.easeOut,
  },

  // Fade animations
  fadeIn: {
    from: { opacity: 0 },
    to: { opacity: 1 },
    duration: timing.durations.fast,
    easing: timing.easings.easeIn,
  },

  fadeOut: {
    from: { opacity: 1 },
    to: { opacity: 0 },
    duration: timing.durations.fast,
    easing: timing.easings.easeOut,
  },

  // Scale animations (zoom effects)
  scaleIn: {
    from: {
      transform: 'scale(0.95)',
      opacity: 0
    },
    to: {
      transform: 'scale(1)',
      opacity: 1
    },
    duration: timing.durations.fast,
    easing: timing.easings.easeOut,
  },

  scaleOut: {
    from: {
      transform: 'scale(1)',
      opacity: 1
    },
    to: {
      transform: 'scale(0.95)',
      opacity: 0
    },
    duration: timing.durations.fast,
    easing: timing.easings.easeIn,
  },

  // Hover effects (subtle interactions)
  liftUp: {
    from: { transform: 'translateY(0)' },
    to: { transform: 'translateY(-2px)' },
    duration: timing.durations.fast,
    easing: timing.easings.easeOut,
  },

  scaleUp: {
    from: { transform: 'scale(1)' },
    to: { transform: 'scale(1.05)' },
    duration: timing.durations.fast,
    easing: timing.easings.easeOut,
  },
} as const;

// Type exports
export type AnimationPrimitive = keyof typeof animationPrimitives;
