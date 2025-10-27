/**
 * Design System: Token Exports
 *
 * Centralized export of all design tokens for easy import throughout
 * the application.
 *
 * Usage:
 * import { colors, timing, spacing } from '@/design-system/tokens';
 */

export * from './colors';
export * from './timing';
export * from './spacing';
export * from './breakpoints';
export * from './typography';

// Re-export as named objects for convenience
export { colors } from './colors';
export { timing, animations } from './timing';
export { spacing, componentSpacing } from './spacing';
export { breakpoints, mediaQueries, devices } from './breakpoints';
export { typography, textStyles } from './typography';
