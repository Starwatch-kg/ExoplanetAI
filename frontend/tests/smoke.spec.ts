import { test, expect } from '@playwright/test';

test('landing has title and links work', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Exoplanet AI/);
  await expect(page.getByText('Как это работает')).toBeVisible();
  await page.click('text=Как это работает');
  await expect(page).toHaveURL(/how-it-works/);
});

test('login page renders', async ({ page }) => {
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: 'Вход' })).toBeVisible();
});

test('demo route renders', async ({ page }) => {
  await page.goto('/demo');
  await expect(page.getByText(/Модель:/)).toBeVisible();
});
