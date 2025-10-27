/**
 * Utility: Classname Helper
 *
 * Combines clsx (conditional classnames) with tailwind-merge (intelligent
 * Tailwind class merging) to prevent conflicting utility classes.
 *
 * Usage:
 * cn('px-4 py-2', isActive && 'bg-accent-600', className)
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
