# Argentic Web Design Guide

## Overview

Argentic Web uses the **Nord color palette** as its design foundation, providing a beautiful arctic-inspired aesthetic with excellent accessibility and readability in both light and dark themes.

## Nord Color Palette

### Polar Night (Dark Theme Backgrounds)
- `nord0` (#2e3440) - Base background
- `nord1` (#3b4252) - Lighter background
- `nord2` (#434c5e) - Selection background
- `nord3` (#4c566a) - Comments, subtle elements

### Snow Storm (Light Theme Backgrounds)
- `nord4` (#d8dee9) - Dark text
- `nord5` (#e5e9f0) - Base foreground
- `nord6` (#eceff4) - Base background (light)

### Frost (Blues - Primary colors)
- `nord7` (#8fbcbb) - Cyan
- `nord8` (#88c0d0) - Bright cyan
- `nord9` (#81a1c1) - Blue *(Primary Brand Color)*
- `nord10` (#5e81ac) - Dark blue

### Aurora (Accent colors)
- `nord11` (#bf616a) - Red (Errors)
- `nord12` (#d08770) - Orange
- `nord13` (#ebcb8b) - Yellow (Warnings)
- `nord14` (#a3be8c) - Green (Success)
- `nord15` (#b48ead) - Purple

## Theme System

### CSS Variables

All colors are exposed as CSS custom properties:

```css
/* Background colors */
--bg-primary        /* Main background */
--bg-secondary      /* Cards, elevated surfaces */
--bg-tertiary       /* Hover states */
--bg-elevated       /* Modals, dropdowns */

/* Text colors */
--text-primary      /* Main text */
--text-secondary    /* Subtext */
--text-inverse      /* Text on colored backgrounds */
--text-muted        /* Placeholder, disabled */

/* Border colors */
--border-primary
--border-secondary
--border-focus      /* Focus rings */

/* Semantic colors */
--color-primary
--color-success
--color-warning
--color-error
--color-info
```

### Theme Switching

Users can choose from three theme modes:

1. **Light Mode** - Snow Storm palette
2. **Dark Mode** - Polar Night palette
3. **Auto Mode** - Follows system preference

The theme persists in `localStorage` and applies system preference by default.

```typescript
// Using the theme service
import { ThemeService } from '@app/core/theme.service';

constructor(private themeService: ThemeService) {}

// Set theme
this.themeService.setTheme('dark');

// Toggle theme
this.themeService.toggleTheme();

// Get current theme
const theme = this.themeService.theme();
```

## Typography

### Font Family

```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', ...;
```

### Font Weights
- **Regular (400)** - Body text
- **Medium (500)** - Subheadings, buttons
- **Semibold (600)** - Headings
- **Bold (700)** - Titles, emphasis

### Font Sizes

| Element | Size | Use Case |
|---------|------|----------|
| H1 | 2rem (32px) | Page titles |
| H2 | 1.5rem (24px) | Section headers |
| H3 | 1.25rem (20px) | Card headers |
| Body | 1rem (16px) | Main text |
| Small | 0.875rem (14px) | Helper text, labels |
| Tiny | 0.75rem (12px) | Badges, captions |

## Components

### Buttons

#### Primary Button
```html
<button class="btn btn-primary">Primary Action</button>
```
- **Use**: Main calls-to-action
- **Color**: Nord9 (Blue)
- **Hover**: Slight elevation + Nord8 (Bright Cyan)

#### Secondary Button
```html
<button class="btn btn-secondary">Secondary Action</button>
```
- **Use**: Secondary actions
- **Color**: Nord10 (Dark Blue)

#### Outline Button
```html
<button class="btn btn-outline">Cancel</button>
```
- **Use**: Tertiary actions, cancel
- **Style**: Transparent with border

#### Success / Danger Buttons
```html
<button class="btn btn-success">Confirm</button>
<button class="btn btn-danger">Delete</button>
```
- **Success**: Nord14 (Green)
- **Danger**: Nord11 (Red)

### Cards

```html
<div class="card">
  <div class="card-header">Title</div>
  <p>Content goes here</p>
</div>
```

**Features:**
- Border radius: 12px
- Shadow: Subtle elevation
- Hover: Increased shadow
- Border: 1px solid border-primary

### Forms

#### Input Fields
```html
<div class="form-group">
  <label class="form-label">Field Name</label>
  <input type="text" class="form-control" placeholder="Enter value">
</div>
```

**States:**
- **Default**: Border-primary
- **Focus**: Border-focus (Nord9) + subtle shadow
- **Disabled**: 60% opacity
- **Error**: Border-error (Nord11)

#### Select Dropdowns
```html
<select class="form-control">
  <option>Option 1</option>
  <option>Option 2</option>
</select>
```

### Status Badges

```html
<span class="badge badge-success">Running</span>
<span class="badge badge-warning">Pending</span>
<span class="badge badge-danger">Error</span>
<span class="badge badge-secondary">Stopped</span>
```

**Design:**
- Uppercase text
- Letter spacing: 0.05em
- Border + light background
- Font weight: 600

### Loading Spinner

```html
<div class="spinner"></div>
```

**Appearance:**
- Circle with rotating top border
- Color: Primary blue
- Size: 40px × 40px

## Spacing System

Based on 8px grid:

| Class | Value |
|-------|-------|
| mt-1 | 8px |
| mt-2 | 16px |
| mt-3 | 24px |
| mb-1 | 8px |
| mb-2 | 16px |
| mb-3 | 24px |

**Container padding:** 20px
**Card padding:** 20px
**Form group margin:** 20px

## Shadows

```css
--shadow-sm: 0 1px 2px rgba(...)     /* Subtle */
--shadow-md: 0 4px 6px rgba(...)     /* Cards, elevated */
--shadow-lg: 0 10px 15px rgba(...)   /* Modals, hover */
--shadow-xl: 0 20px 25px rgba(...)   /* Overlays */
```

Shadows are context-aware:
- **Light theme**: rgba(46, 52, 64, 0.1-0.3)
- **Dark theme**: rgba(0, 0, 0, 0.2-0.5)

## Border Radius

| Element | Radius |
|---------|--------|
| Buttons | 8px |
| Cards | 12px |
| Inputs | 8px |
| Badges | 6px |
| Modals | 12px |

## Transitions

```css
--transition-fast: 150ms ease
--transition-base: 200ms ease
--transition-slow: 300ms ease
```

**Properties with transitions:**
- `background-color`
- `border-color`
- `color`
- `transform`
- `box-shadow`

## Layout

### Sidebar Navigation

**Width:** 260px
**Background:** Nord0 (always dark)
**Active link:** Primary blue with shadow
**Hover:** Slight translate-x animation

### Main Content

**Background:** Follows theme
**Max width:** 1400px (centered)
**Padding:** 20px

### Grid Layouts

Cards use CSS Grid with auto-fill:

```scss
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
}
```

## Accessibility

### Focus States

All interactive elements have visible focus rings:

```css
&:focus-visible {
  outline: 2px solid var(--border-focus);
  outline-offset: 2px;
}
```

### Color Contrast

- **Light theme**: WCAG AA compliant (4.5:1+)
- **Dark theme**: WCAG AA compliant (4.5:1+)
- Text on backgrounds meets contrast requirements

### Keyboard Navigation

- Tab order follows visual hierarchy
- All actions accessible via keyboard
- Skip links for screen readers

## Icons

Currently using emoji icons for simplicity:

- 📊 Dashboard
- 🤖 Agents
- 👥 Supervisors
- 🔄 Workflows
- ⚙️ Configuration
- ☀️ Light theme
- 🌙 Dark theme
- 💻 Auto theme

*Future: Consider icon library like Lucide or Heroicons*

## Responsive Design

### Breakpoints

```scss
// Mobile
@media (max-width: 640px) { ... }

// Tablet
@media (max-width: 1024px) { ... }

// Desktop
@media (min-width: 1024px) { ... }
```

### Mobile-First Approach

Design for mobile first, then enhance for larger screens.

## Best Practices

### Do's ✅

- Use CSS variables for all colors
- Maintain 8px spacing grid
- Use semantic HTML elements
- Add proper ARIA labels
- Test in both themes
- Use transition variables
- Follow Nord palette strictly

### Don'ts ❌

- Don't use hard-coded colors
- Don't ignore accessibility
- Don't mix color systems
- Don't use inline styles
- Don't skip hover/focus states
- Don't use < 12px font sizes

## Component Examples

### Agent Card

```html
<div class="agent-card">
  <div class="card-header-row">
    <h3>Agent Name</h3>
    <span class="badge badge-success">Running</span>
  </div>
  <p class="agent-description">Description text</p>
  <div class="card-actions">
    <button class="btn btn-sm btn-primary">Edit</button>
    <button class="btn btn-sm btn-outline">Delete</button>
  </div>
</div>
```

### Modal

```html
<div class="modal">
  <div class="modal-backdrop"></div>
  <div class="modal-content">
    <div class="modal-header">
      <h2>Modal Title</h2>
      <button class="modal-close">&times;</button>
    </div>
    <div class="modal-body">
      <!-- Content -->
    </div>
    <div class="modal-footer">
      <button class="btn btn-secondary">Cancel</button>
      <button class="btn btn-primary">Confirm</button>
    </div>
  </div>
</div>
```

## Resources

- **Nord Theme**: https://www.nordtheme.com/
- **WCAG Guidelines**: https://www.w3.org/WAI/WCAG21/quickref/
- **Inter Font**: https://rsms.me/inter/
- **CSS Variables**: https://developer.mozilla.org/en-US/docs/Web/CSS/--*

## Future Enhancements

- [ ] Custom icon set
- [ ] Animation library
- [ ] Component showcase/storybook
- [ ] Dark mode illustrations
- [ ] Custom form validation styles
- [ ] Toast/notification system
- [ ] Progress indicators
- [ ] Skeleton loaders

---

**Version:** 1.0.0
**Last Updated:** 2024
**Maintainer:** Argentic Team
