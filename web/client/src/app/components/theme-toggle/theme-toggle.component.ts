import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ThemeService, Theme } from '../../core/theme.service';

@Component({
  selector: 'app-theme-toggle',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="theme-toggle">
      <button
        class="theme-toggle-btn"
        (click)="toggleTheme()"
        [attr.aria-label]="'Switch to ' + nextTheme() + ' theme'"
        title="Toggle theme">
        <span class="theme-icon">{{ themeIcon() }}</span>
      </button>

      <div class="theme-menu" *ngIf="showMenu">
        <button
          class="theme-option"
          [class.active]="themeService.theme() === 'light'"
          (click)="setTheme('light')">
          <span class="theme-icon">☀️</span>
          <span>Light</span>
        </button>
        <button
          class="theme-option"
          [class.active]="themeService.theme() === 'dark'"
          (click)="setTheme('dark')">
          <span class="theme-icon">🌙</span>
          <span>Dark</span>
        </button>
        <button
          class="theme-option"
          [class.active]="themeService.theme() === 'auto'"
          (click)="setTheme('auto')">
          <span class="theme-icon">💻</span>
          <span>Auto</span>
        </button>
      </div>
    </div>
  `,
  styles: [`
    .theme-toggle {
      position: relative;
    }

    .theme-toggle-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 8px;
      border: 1px solid var(--border-primary);
      background-color: var(--bg-secondary);
      cursor: pointer;
      transition: all var(--transition-base);

      &:hover {
        background-color: var(--bg-tertiary);
        border-color: var(--color-primary);
      }

      &:focus-visible {
        outline: 2px solid var(--border-focus);
        outline-offset: 2px;
      }
    }

    .theme-icon {
      font-size: 1.25rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .theme-menu {
      position: absolute;
      top: calc(100% + 8px);
      right: 0;
      background-color: var(--card-bg);
      border: 1px solid var(--border-primary);
      border-radius: 8px;
      box-shadow: var(--shadow-lg);
      padding: 8px;
      min-width: 150px;
      z-index: 1000;
    }

    .theme-option {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      padding: 10px 12px;
      border: none;
      background: transparent;
      color: var(--text-primary);
      border-radius: 6px;
      cursor: pointer;
      transition: all var(--transition-base);
      font-size: 14px;
      font-weight: 500;

      &:hover {
        background-color: var(--bg-secondary);
      }

      &.active {
        background-color: var(--color-primary-light);
        color: var(--color-primary);
      }

      .theme-icon {
        font-size: 1.1rem;
      }
    }
  `]
})
export class ThemeToggleComponent {
  showMenu = false;

  constructor(public themeService: ThemeService) {}

  themeIcon = computed(() => {
    const theme = this.themeService.theme();
    const actualTheme = this.themeService.getActualTheme();

    if (theme === 'auto') {
      return '💻';
    }
    return actualTheme === 'dark' ? '🌙' : '☀️';
  });

  nextTheme = computed(() => {
    const current = this.themeService.getActualTheme();
    return current === 'dark' ? 'light' : 'dark';
  });

  toggleTheme(): void {
    this.themeService.toggleTheme();
  }

  setTheme(theme: Theme): void {
    this.themeService.setTheme(theme);
    this.showMenu = false;
  }

  toggleMenu(): void {
    this.showMenu = !this.showMenu;
  }
}
