import { Injectable, signal, effect } from '@angular/core';

export type Theme = 'light' | 'dark' | 'auto';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private readonly THEME_KEY = 'argentic-theme';

  // Signal for reactive theme state
  theme = signal<Theme>(this.getInitialTheme());

  // Computed actual theme (resolves 'auto' to light/dark)
  private mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

  constructor() {
    // Apply theme on initialization
    this.applyTheme(this.theme());

    // Listen for system theme changes
    this.mediaQuery.addEventListener('change', () => {
      if (this.theme() === 'auto') {
        this.applyTheme('auto');
      }
    });

    // Effect to apply theme when it changes
    effect(() => {
      const currentTheme = this.theme();
      this.applyTheme(currentTheme);
      this.saveTheme(currentTheme);
    });
  }

  /**
   * Set the theme
   */
  setTheme(theme: Theme): void {
    this.theme.set(theme);
  }

  /**
   * Toggle between light and dark themes
   */
  toggleTheme(): void {
    const current = this.getActualTheme();
    this.theme.set(current === 'dark' ? 'light' : 'dark');
  }

  /**
   * Get the actual theme (resolves 'auto')
   */
  getActualTheme(): 'light' | 'dark' {
    const theme = this.theme();
    if (theme === 'auto') {
      return this.mediaQuery.matches ? 'dark' : 'light';
    }
    return theme;
  }

  /**
   * Apply theme to document
   */
  private applyTheme(theme: Theme): void {
    const actualTheme = theme === 'auto'
      ? (this.mediaQuery.matches ? 'dark' : 'light')
      : theme;

    document.documentElement.setAttribute('data-theme', actualTheme);

    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      const color = actualTheme === 'dark' ? '#2e3440' : '#eceff4';
      metaThemeColor.setAttribute('content', color);
    }
  }

  /**
   * Get initial theme from localStorage or system preference
   */
  private getInitialTheme(): Theme {
    const savedTheme = localStorage.getItem(this.THEME_KEY) as Theme;
    if (savedTheme && ['light', 'dark', 'auto'].includes(savedTheme)) {
      return savedTheme;
    }
    return 'auto';
  }

  /**
   * Save theme to localStorage
   */
  private saveTheme(theme: Theme): void {
    localStorage.setItem(this.THEME_KEY, theme);
  }
}
