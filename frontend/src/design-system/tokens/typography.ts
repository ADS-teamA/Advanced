/**
 * Design Tokens: Typography
 *
 * Font families, sizes, weights, and line heights for consistent
 * text hierarchy throughout the application.
 */

export const typography = {
  // Font families
  fontFamily: {
    sans: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },

  // Font sizes (with rem and px equivalents)
  fontSize: {
    xs: { rem: '0.75rem', px: '12px' },     // Extra small
    sm: { rem: '0.875rem', px: '14px' },    // Small
    base: { rem: '1rem', px: '16px' },      // Base (body text)
    lg: { rem: '1.125rem', px: '18px' },    // Large
    xl: { rem: '1.25rem', px: '20px' },     // Extra large
    '2xl': { rem: '1.5rem', px: '24px' },   // 2X large (h3)
    '3xl': { rem: '1.875rem', px: '30px' }, // 3X large (h2)
    '4xl': { rem: '2.25rem', px: '36px' },  // 4X large (h1)
    '5xl': { rem: '3rem', px: '48px' },     // 5X large
    '6xl': { rem: '3.75rem', px: '60px' },  // 6X large
  },

  // Font weights
  fontWeight: {
    thin: '100',
    extralight: '200',
    light: '300',
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
    extrabold: '800',
    black: '900',
  },

  // Line heights
  lineHeight: {
    none: '1',
    tight: '1.25',
    snug: '1.375',
    normal: '1.5',
    relaxed: '1.625',
    loose: '2',
  },

  // Letter spacing
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0em',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em',
  },
} as const;

// Semantic typography (component-specific)
export const textStyles = {
  // Headings
  h1: {
    fontSize: typography.fontSize['4xl'].rem,
    fontWeight: typography.fontWeight.bold,
    lineHeight: typography.lineHeight.tight,
  },
  h2: {
    fontSize: typography.fontSize['3xl'].rem,
    fontWeight: typography.fontWeight.bold,
    lineHeight: typography.lineHeight.tight,
  },
  h3: {
    fontSize: typography.fontSize['2xl'].rem,
    fontWeight: typography.fontWeight.semibold,
    lineHeight: typography.lineHeight.snug,
  },
  h4: {
    fontSize: typography.fontSize.xl.rem,
    fontWeight: typography.fontWeight.semibold,
    lineHeight: typography.lineHeight.snug,
  },

  // Body text
  body: {
    fontSize: typography.fontSize.base.rem,
    fontWeight: typography.fontWeight.normal,
    lineHeight: typography.lineHeight.normal,
  },
  bodyLarge: {
    fontSize: typography.fontSize.lg.rem,
    fontWeight: typography.fontWeight.normal,
    lineHeight: typography.lineHeight.relaxed,
  },
  bodySmall: {
    fontSize: typography.fontSize.sm.rem,
    fontWeight: typography.fontWeight.normal,
    lineHeight: typography.lineHeight.normal,
  },

  // Special text
  caption: {
    fontSize: typography.fontSize.xs.rem,
    fontWeight: typography.fontWeight.normal,
    lineHeight: typography.lineHeight.normal,
  },
  label: {
    fontSize: typography.fontSize.sm.rem,
    fontWeight: typography.fontWeight.medium,
    lineHeight: typography.lineHeight.normal,
  },
  code: {
    fontFamily: typography.fontFamily.mono,
    fontSize: typography.fontSize.sm.rem,
    fontWeight: typography.fontWeight.normal,
  },
} as const;

// Type exports
export type FontSize = keyof typeof typography.fontSize;
export type FontWeight = keyof typeof typography.fontWeight;
export type LineHeight = keyof typeof typography.lineHeight;
export type TextStyle = keyof typeof textStyles;
