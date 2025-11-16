import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test('should display dashboard page', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('should show stats cards', async ({ page }) => {
    await page.goto('/dashboard');

    // Check for stat cards
    await expect(page.locator('.stat-card')).toHaveCount(3);
    await expect(page.getByText('Agents')).toBeVisible();
    await expect(page.getByText('Supervisors')).toBeVisible();
    await expect(page.getByText('Workflows')).toBeVisible();
  });

  test('should have quick action buttons', async ({ page }) => {
    await page.goto('/dashboard');

    await expect(page.getByRole('link', { name: 'Create Agent' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Create Supervisor' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Create Workflow' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Configure Settings' })).toBeVisible();
  });

  test('should navigate to agents page', async ({ page }) => {
    await page.goto('/dashboard');
    await page.getByRole('link', { name: 'Create Agent' }).click();
    await expect(page).toHaveURL(/.*agents/);
  });
});
