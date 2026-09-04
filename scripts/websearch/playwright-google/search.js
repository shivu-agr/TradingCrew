#!/usr/bin/env node
//
// playwright-google/search.js — drive a persistent headless Chromium against
// google.com/search and print the result URLs, one per line.
//
// Usage:
//   node search.js "<query>"
//
// Env vars:
//   WS_NUM=20        Max URLs to print (default 20).
//   PW_HEADLESS=0    Show the browser window (useful to solve a CAPTCHA once,
//                    which then stays cleared in the persistent profile).
//
// The user profile lives at ~/.cache/playwright-google, so cookies (including
// EU consent, and Google's "trust this browser" state) persist across calls.
// That's what keeps CAPTCHA rates low under sustained scripted use.

'use strict';

// playwright-extra lets us stack puppeteer-extra plugins on top of Playwright.
// The stealth plugin patches the ~20 most common "we're a bot" tells
// (navigator.webdriver, WebGL vendor spoof, chrome runtime shims, etc.).
// It's the difference between hitting a CAPTCHA on query 3 vs. query 30+.
const { chromium } = require('playwright-extra');
const stealth      = require('puppeteer-extra-plugin-stealth')();
chromium.use(stealth);

const path = require('path');
const os   = require('os');
const fs   = require('fs');

// PW_CHANNEL=chrome drives the locally-installed Google Chrome instead of the
// bundled Chromium.  Real Chrome is far less bot-detectable, so headless
// Google searches survive much longer before hitting a CAPTCHA.  Keep a
// channel-specific profile dir so Chrome and Chromium state never collide.
const CHANNEL   = process.env.PW_CHANNEL || '';
const USER_DATA = path.join(os.homedir(), '.cache',
  CHANNEL ? `playwright-google-${CHANNEL}` : 'playwright-google');
const NUM       = parseInt(process.env.WS_NUM || '20', 10);
const HEADLESS  = process.env.PW_HEADLESS !== '0';

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
           '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36';

async function maybeHandleConsent(page) {
  // Google EU consent screen appears on the first-ever visit. Click one of
  // the resolve buttons — "Reject all" is preferred because it's shorter and
  // still lets results render.
  const selectors = [
    'button:has-text("Reject all")',
    'button[aria-label="Reject all"]',
    '#W0wltc',                    // "Reject all" id on some locales
    'button:has-text("Accept all")',
    '#L2AGLb',                    // "Accept all" id on some locales
  ];
  for (const sel of selectors) {
    const loc = page.locator(sel).first();
    if (await loc.count().catch(() => 0)) {
      try {
        await loc.click({ timeout: 3000 });
        await page.waitForLoadState('domcontentloaded', { timeout: 5000 }).catch(() => {});
        return true;
      } catch { /* try next */ }
    }
  }
  return false;
}

function isCaptcha(page) {
  return page.locator('form#captcha-form, div#recaptcha, #captcha-form').count()
    .then((n) => n > 0)
    .catch(() => false);
}

async function main() {
  const query = process.argv[2];
  if (!query) {
    console.error('usage: node search.js "<query>"');
    process.exit(1);
  }

  fs.mkdirSync(USER_DATA, { recursive: true });

  const launchOpts = {
    headless: HEADLESS,
    viewport: { width: 1366, height: 800 },
    userAgent: UA,
    locale: 'en-US',
    // Reduce the couple of most obvious "we're a bot" tells.
    args: [
      '--disable-blink-features=AutomationControlled',
      '--no-default-browser-check',
      '--no-first-run',
    ],
  };
  if (CHANNEL) launchOpts.channel = CHANNEL;
  const ctx = await chromium.launchPersistentContext(USER_DATA, launchOpts);

  try {
    const page = ctx.pages()[0] || (await ctx.newPage());
    // Extra polish: strip the navigator.webdriver flag before anything runs.
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'webdriver', { get: () => false });
    });

    const url =
      'https://www.google.com/search?q=' + encodeURIComponent(query) +
      '&num=20&hl=en&gl=us';

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await maybeHandleConsent(page);

    if (await isCaptcha(page)) {
      if (HEADLESS) {
        console.error('[google-headless] Google served a CAPTCHA. Options:');
        console.error('  • Wait a few minutes and retry.');
        console.error('  • Run once with PW_HEADLESS=0 to solve it interactively —');
        console.error('    the persistent profile keeps you cleared afterwards.');
        process.exit(4);
      }
      // Headed mode: keep the window open so a human can solve the CAPTCHA.
      // The persistent profile then stays cleared for hours of headless runs.
      const waitMs = parseInt(process.env.PW_CAPTCHA_WAIT_MS || '300000', 10);
      console.error(
        `[google-headless] CAPTCHA shown — solve it in the browser window. ` +
        `Waiting up to ${Math.round(waitMs / 1000)}s for you to finish...`);
      const deadline = Date.now() + waitMs;
      while (Date.now() < deadline) {
        await page.waitForTimeout(2000);
        if (!(await isCaptcha(page))) break;
      }
      if (await isCaptcha(page)) {
        console.error('[google-headless] still on a CAPTCHA after waiting; giving up.');
        process.exit(4);
      }
      console.error('[google-headless] CAPTCHA cleared — continuing.');
      // Google usually redirects to the original query after a solve, but
      // re-navigate to be sure we land on the results page.
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 }).catch(() => {});
      await maybeHandleConsent(page);
    }

    // Give Google's SSR + partial hydration a moment to settle.
    await page.waitForSelector('#rso, #search', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(250);

    const urls = await page.evaluate((limit) => {
      const out = [];
      const seen = new Set();
      // Stable pattern in 2026: every organic result has an <h3> beneath an
      // <a href="https://…">. Class names are randomised per deploy, so key
      // off the h3 → anchor relationship instead.
      const nodes = document.querySelectorAll('#rso a h3, #search a h3, div.g a h3');
      nodes.forEach((h3) => {
        const a = h3.closest('a');
        if (!a) return;
        const href = a.href;
        if (!href || !href.startsWith('http')) return;
        if (seen.has(href)) return;
        seen.add(href);
        out.push(href);
      });
      return out.slice(0, limit);
    }, NUM);

    // Filter ad / redirect / google-own URLs.
    const clean = urls.filter((u) => {
      try {
        const h = new URL(u).hostname.toLowerCase();
        if (h.includes('google.')) return false;
        if (h.includes('gstatic')) return false;
        if (h.includes('googleusercontent')) return false;
        return true;
      } catch { return false; }
    });

    if (!clean.length) {
      console.error('[google-headless] no results parsed. Try PW_HEADLESS=0 to inspect.');
      process.exit(5);
    }

    console.log(clean.join('\n'));
    process.exit(0);
  } catch (e) {
    console.error('[google-headless] error:', e && e.message ? e.message : e);
    process.exit(3);
  } finally {
    await ctx.close();
  }
}

main();
