/**
 * Animation System: Exports
 *
 * Centralized export of all animation primitives and hooks.
 *
 * Usage:
 * import { useSlideIn, useFadeIn, useHover } from '@/design-system/animations';
 */

export * from './primitives';
export * from './hooks/useSlideIn';
export * from './hooks/useFadeIn';
export * from './hooks/useHover';
export * from './hooks/useLoadingState';

// Re-export as named imports for convenience
export { animationPrimitives } from './primitives';
export { useSlideIn } from './hooks/useSlideIn';
export { useFadeIn } from './hooks/useFadeIn';
export { useHover } from './hooks/useHover';
export { useLoadingState } from './hooks/useLoadingState';
