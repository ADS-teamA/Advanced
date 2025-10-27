/**
 * Design Tokens: Animation Timing
 *
 * Consistent animation durations and easing functions for smooth,
 * performant transitions throughout the application.
 */

export const timing = {
  // Duration presets
  durations: {
    instant: '0ms',      // No animation (accessibility)
    fast: '150ms',       // Quick feedback (hover, focus)
    normal: '250ms',     // Standard transitions (dropdowns, tooltips)
    slow: '400ms',       // Emphasized transitions (modals, sheets)
    slower: '600ms',     // Page transitions, complex animations
  },

  // Easing functions (cubic-bezier)
  easings: {
    // Standard easings
    linear: 'linear',
    easeOut: 'cubic-bezier(0.0, 0, 0.2, 1)',      // Decelerate (enter)
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',         // Accelerate (exit)
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',    // Symmetric

    // Custom easings
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',  // Bouncy spring effect
    sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',        // Sharp acceleration/deceleration
    smooth: 'cubic-bezier(0.25, 0.1, 0.25, 1)',   // Very smooth, natural
  },
} as const;

// Preset animation combinations (duration + easing)
export const animations = {
  // Entrance animations (ease-out for entering elements)
  enterFast: `${timing.durations.fast} ${timing.easings.easeOut}`,
  enterNormal: `${timing.durations.normal} ${timing.easings.easeOut}`,
  enterSlow: `${timing.durations.slow} ${timing.easings.easeOut}`,

  // Exit animations (ease-in for leaving elements)
  exitFast: `${timing.durations.fast} ${timing.easings.easeIn}`,
  exitNormal: `${timing.durations.normal} ${timing.easings.easeIn}`,
  exitSlow: `${timing.durations.slow} ${timing.easings.easeIn}`,

  // Interactive animations (ease-in-out for state changes)
  interactiveFast: `${timing.durations.fast} ${timing.easings.easeInOut}`,
  interactiveNormal: `${timing.durations.normal} ${timing.easings.easeInOut}`,

  // Special effects
  spring: `${timing.durations.slow} ${timing.easings.spring}`,
  smooth: `${timing.durations.normal} ${timing.easings.smooth}`,
} as const;

// Type exports
export type Duration = keyof typeof timing.durations;
export type Easing = keyof typeof timing.easings;
export type Animation = keyof typeof animations;
