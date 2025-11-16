import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should have sidebar with all navigation items', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('link', { name: /Dashboard/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Agents/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Supervisors/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Workflows/ })).toBeVisible();
    await expect(page.getByRole('link', { name: /Configuration/ })).toBeVisible();
  });

  test('should navigate to dashboard', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /Dashboard/ }).click();
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('should navigate to agents page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /Agents/ }).click();
    await expect(page).toHaveURL(/.*agents/);
    await expect(page.locator('h1')).toContainText('Agents');
  });

  test('should navigate to supervisors page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /Supervisors/ }).click();
    await expect(page).toHaveURL(/.*supervisors/);
    await expect(page.locator('h1')).toContainText('Supervisors');
  });

  test('should navigate to workflows page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /Workflows/ }).click();
    await expect(page).toHaveURL(/.*workflows/);
    await expect(page.locator('h1')).toContainText('Workflows');
  });

  test('should navigate to config page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: /Configuration/ }).click();
    await expect(page).toHaveURL(/.*config/);
    await expect(page.locator('h1')).toContainText('Configuration');
  });

  test('should highlight active navigation item', async ({ page }) => {
    await page.goto('/agents');

    const agentsLink = page.getByRole('link', { name: /Agents/ });
    await expect(agentsLink).toHaveClass(/active/);
  });
});
