/**
 * Design Tokens: Colors
 *
 * Grayscale-only color palette for minimalist black-and-white design.
 * Inspired by ChatGPT's clean aesthetic.
 */

export const colors = {
  // Base grayscale palette (12 shades)
  gray: {
    50: '#FAFAFA',   // Lightest - subtle backgrounds
    100: '#F5F5F5',  // Very light - hover states
    200: '#E5E5E5',  // Light - borders, dividers
    300: '#D4D4D4',  // Light-medium - disabled states
    400: '#A3A3A3',  // Medium-light - secondary text
    500: '#737373',  // Mid-tone - icons, tertiary text
    600: '#525252',  // Medium-dark - body text
    700: '#404040',  // Dark - emphasis
    800: '#262626',  // Very dark - headings
    900: '#171717',  // Darkest - backgrounds (dark mode)
    950: '#0A0A0A',  // Near black - maximum contrast
  },

  // Semantic colors (grayscale tones for meaning)
  background: {
    light: '#FFFFFF',
    dark: '#0A0A0A',
  },

  surface: {
    light: '#F5F5F5',
    lightElevated: '#FFFFFF',
    dark: '#171717',
    darkElevated: '#262626',
  },

  border: {
    light: '#E5E5E5',
    lightSubtle: '#F5F5F5',
    dark: '#404040',
    darkSubtle: '#262626',
  },

  text: {
    primary: {
      light: '#0A0A0A',
      dark: '#FAFAFA',
    },
    secondary: {
      light: '#525252',
      dark: '#A3A3A3',
    },
    tertiary: {
      light: '#A3A3A3',
      dark: '#737373',
    },
    disabled: {
      light: '#D4D4D4',
      dark: '#404040',
    },
  },

  // Accent (subtle gray emphasis for interactive elements)
  accent: {
    default: {
      light: '#262626',
      dark: '#F5F5F5',
    },
    hover: {
      light: '#404040',
      dark: '#E5E5E5',
    },
    active: {
      light: '#525252',
      dark: '#D4D4D4',
    },
  },

  // Status colors (grayscale variations)
  status: {
    online: {
      light: '#525252',
      dark: '#A3A3A3',
    },
    offline: {
      light: '#A3A3A3',
      dark: '#737373',
    },
    active: {
      light: '#404040',
      dark: '#E5E5E5',
    },
  },
} as const;

// Type exports for TypeScript autocomplete
export type GrayShade = keyof typeof colors.gray;
export type ColorToken = keyof typeof colors;
