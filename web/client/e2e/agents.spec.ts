import { test, expect } from '@playwright/test';

test.describe('Agents Management', () => {
  test('should display agents page', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator('h1')).toContainText('Agents');
  });

  test('should open create agent modal', async ({ page }) => {
    await page.goto('/agents');
    await page.getByRole('button', { name: 'Create Agent' }).click();

    await expect(page.getByText('Create New Agent')).toBeVisible();
    await expect(page.getByLabel('Role *')).toBeVisible();
    await expect(page.getByLabel('Description *')).toBeVisible();
  });

  test('should validate required fields in create form', async ({ page }) => {
    await page.goto('/agents');
    await page.getByRole('button', { name: 'Create Agent' }).click();

    // Try to submit without filling required fields
    const createButton = page.getByRole('button', { name: 'Create Agent', exact: true });
    await expect(createButton).toBeDisabled();

    // Fill role only
    await page.getByLabel('Role *').fill('test_agent');
    await expect(createButton).toBeDisabled();

    // Fill description
    await page.getByLabel('Description *').fill('Test agent description');
    await expect(createButton).toBeEnabled();
  });

  test('should close modal on cancel', async ({ page }) => {
    await page.goto('/agents');
    await page.getByRole('button', { name: 'Create Agent' }).click();

    await expect(page.getByText('Create New Agent')).toBeVisible();

    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByText('Create New Agent')).not.toBeVisible();
  });

  test('should close modal on backdrop click', async ({ page }) => {
    await page.goto('/agents');
    await page.getByRole('button', { name: 'Create Agent' }).click();

    await expect(page.getByText('Create New Agent')).toBeVisible();

    // Click backdrop
    await page.locator('.modal-backdrop').click();
    await expect(page.getByText('Create New Agent')).not.toBeVisible();
  });
});
