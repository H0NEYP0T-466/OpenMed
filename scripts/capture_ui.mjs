import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const outDir = '/home/honeypot/Projects/FAST_API/OpenMed/designs/screenshots';
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist', '--no-sandbox']
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  console.log('Navigating to http://localhost:5173...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3500);

  // 1. Whole body atlas
  console.log('Capturing whole body atlas...');
  await page.screenshot({ path: path.join(outDir, '01_whole_body_atlas.png') });

  // 2. Click Organ Workspaces button to go to heart
  console.log('Switching to Heart view...');
  const organWorkspaceBtn = page.locator('button:has-text("Organ Workspaces")');
  if (await organWorkspaceBtn.count() > 0) {
    await organWorkspaceBtn.first().click();
  } else {
    await page.locator('.pill:has-text("Heart")').first().click();
  }
  await page.waitForTimeout(4500);
  await page.screenshot({ path: path.join(outDir, '02_organ_heart.png') });

  // 3. Click Brain
  console.log('Switching to Brain view...');
  const brainPill = page.locator('.pill:has-text("Brain")');
  if (await brainPill.count() > 0) {
    await brainPill.first().click();
    await page.waitForTimeout(4500);
    await page.screenshot({ path: path.join(outDir, '03_organ_brain.png') });
  }

  // 4. Click Lungs
  console.log('Switching to Lungs view...');
  const lungsPill = page.locator('.pill:has-text("Lungs")');
  if (await lungsPill.count() > 0) {
    await lungsPill.first().click();
    await page.waitForTimeout(4500);
    await page.screenshot({ path: path.join(outDir, '04_organ_lungs.png') });
  }

  console.log('Screenshots completed successfully!');
  await browser.close();
})();
