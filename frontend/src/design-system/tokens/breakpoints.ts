/**
 * Design Tokens: Responsive Breakpoints
 *
 * Mobile-first responsive breakpoints matching Tailwind defaults.
 */

export const breakpoints = {
  sm: '640px',    // Small devices (phones landscape)
  md: '768px',    // Medium devices (tablets)
  lg: '1024px',   // Large devices (laptops)
  xl: '1280px',   // Extra large (desktops)
  '2xl': '1536px', // 2X large (large desktops)
} as const;

// Media query helpers
export const mediaQueries = {
  sm: `(min-width: ${breakpoints.sm})`,
  md: `(min-width: ${breakpoints.md})`,
  lg: `(min-width: ${breakpoints.lg})`,
  xl: `(min-width: ${breakpoints.xl})`,
  '2xl': `(min-width: ${breakpoints['2xl']})`,

  // Max-width queries (for mobile-first overrides)
  maxSm: `(max-width: ${breakpoints.sm})`,
  maxMd: `(max-width: ${breakpoints.md})`,
  maxLg: `(max-width: ${breakpoints.lg})`,
  maxXl: `(max-width: ${breakpoints.xl})`,

  // Orientation queries
  landscape: '(orientation: landscape)',
  portrait: '(orientation: portrait)',

  // Accessibility queries
  reducedMotion: '(prefers-reduced-motion: reduce)',
  darkMode: '(prefers-color-scheme: dark)',
} as const;

// Device type helpers (semantic breakpoint names)
export const devices = {
  mobile: mediaQueries.maxMd,       // < 768px
  tablet: mediaQueries.md,          // >= 768px
  desktop: mediaQueries.lg,         // >= 1024px
  widescreen: mediaQueries.xl,      // >= 1280px
} as const;

// Type exports
export type Breakpoint = keyof typeof breakpoints;
export type MediaQuery = keyof typeof mediaQueries;
