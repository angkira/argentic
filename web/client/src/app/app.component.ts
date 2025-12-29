import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { ThemeToggleComponent } from './components/theme-toggle/theme-toggle.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, ThemeToggleComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'Argentic Agent Builder';

  navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/agents', label: 'Agents', icon: '🤖' },
    { path: '/supervisors', label: 'Supervisors', icon: '👥' },
    { path: '/workflows', label: 'Workflows', icon: '🔄' },
    { path: '/logs', label: 'Message Bus', icon: '📡' },
    { path: '/config', label: 'Configuration', icon: '⚙️' }
  ];
}
