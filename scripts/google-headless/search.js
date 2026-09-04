#!/usr/bin/env node
//
// google-headless/search.js — drive a persistent Chromium against
// google.com/search and print the result URLs, one per line.
//
// Vendored into the LegalGuard repo (see legalguard/search/google_headless.py,
// which shells out to this script) so every environment gets the identical,
// stealth-patched driver without depending on a machine-local checkout. The
// heavy Chromium binary + node_modules stay gitignored; run the one-time setup
// documented in the README / `make google-headless-setup`.
//
// Usage:
//   node search.js "<query>"
//
// The browser always runs headful on a virtual display (Xvfb): Google
// fingerprints true-headless Chromium on datacenter IPs and CAPTCHAs it, whereas
// a headful browser on Xvfb is rarely challenged. So this needs a DISPLAY — the
// LegalGuard provider injects GOOGLE_HEADLESS_DISPLAY, and Docker runs it under
// xvfb-run.
//
// Env vars:
//   WS_NUM=20        Max URLs to print (default 20).
//   PW_USER_DATA     Override the persistent profile dir (see below).
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

// Persistent Chromium profile (cookies + EU-consent state). Defaults to
// ~/.cache/playwright-google, but can be overridden via PW_USER_DATA — e.g.
// pointed at a mounted volume in Docker so the consent state survives container
// restarts / redeploys.
const USER_DATA = process.env.PW_USER_DATA
  ? path.resolve(process.env.PW_USER_DATA)
  : path.join(os.homedir(), '.cache', 'playwright-google');
const NUM       = parseInt(process.env.WS_NUM || '20', 10);

// Interactive "prime" mode: when GH_INTERACTIVE=1 and Google serves a CAPTCHA,
// DON'T bail — keep the (headful) window open and poll until a human solves it,
// then continue and parse results.  Solving it once writes Google's
// "trust this browser" cookie into the persistent profile (USER_DATA), so every
// later non-interactive run sails past the anti-bot page.  Run this once (see
// `make google-headless-prime`) before wiring the provider into the batch.
const INTERACTIVE   = process.env.GH_INTERACTIVE === '1';
// How long (ms) to wait for the human to solve the CAPTCHA in interactive mode.
const SOLVE_TIMEOUT = parseInt(process.env.GH_SOLVE_TIMEOUT_MS || '300000', 10);

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

  // Rendering mode (GH_MODE):
  //   headful   — a real visible window (Linux/Docker uses Xvfb; on macOS a
  //               window pops up).  Lowest CAPTCHA rate but intrusive on a Mac.
  //   offscreen — headful (so it keeps the low-CAPTCHA fingerprint) but the
  //               window is parked far off-screen, so nothing shows on the Mac
  //               desktop.  Best "invisible on macOS without Xvfb" option.
  //   headless  — Chrome's *new* headless (--headless=new), which renders like
  //               a real browser and, with the stealth patches + a primed
  //               profile, is far less detectable than legacy headless.  Truly
  //               windowless; test the CAPTCHA rate on your IP before trusting.
  // GH_INTERACTIVE (prime) forces headful so a human can see + solve a CAPTCHA.
  const MODE = INTERACTIVE ? 'headful' : (process.env.GH_MODE || 'headful');
  const extraArgs = [];
  if (MODE === 'offscreen') {
    // Park the window off the visible desktop; still a genuine headful browser.
    extraArgs.push('--window-position=-32000,-32000', '--window-size=1366,800');
  }
  if (MODE === 'headless') {
    // New headless (Chrome 112+): much closer to headful than the old mode.
    extraArgs.push('--headless=new');
  }

  const ctx = await chromium.launchPersistentContext(USER_DATA, {
    // Always pass headless:false so Playwright doesn't force the easily-detected
    // LEGACY headless; the real mode is driven by --headless=new in args (for
    // MODE=headless) or by the off-screen window position (MODE=offscreen).
    headless: false,
    viewport: { width: 1366, height: 800 },
    userAgent: UA,
    locale: 'en-US',
    // Reduce the couple of most obvious "we're a bot" tells.
    args: [
      '--disable-blink-features=AutomationControlled',
      '--no-default-browser-check',
      '--no-first-run',
      ...extraArgs,
    ],
  });

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
      if (INTERACTIVE) {
        // Prime mode: leave the window open so a human can solve the CAPTCHA;
        // solving it seeds the persistent profile's "trust this browser" cookie
        // so all later non-interactive runs skip the anti-bot page.
        console.error(
          '[google-headless] CAPTCHA shown — SOLVE IT in the open browser window. ' +
          'Waiting up to ' + Math.round(SOLVE_TIMEOUT / 1000) + 's…'
        );
        const deadline = Date.now() + SOLVE_TIMEOUT;
        while (Date.now() < deadline) {
          await page.waitForTimeout(1500);
          if (!(await isCaptcha(page))) {
            console.error('[google-headless] CAPTCHA cleared — profile primed.');
            break;
          }
        }
        if (await isCaptcha(page)) {
          console.error('[google-headless] CAPTCHA still present after timeout — giving up.');
          process.exit(4);
        }
        await page.waitForLoadState('domcontentloaded', { timeout: 8000 }).catch(() => {});
      } else {
        // A primed profile is rarely challenged, but there's no human to solve
        // one if it happens — fail fast (exit 4) so the composite chain falls
        // through to the next provider instead of hanging.
        console.error('[google-headless] Google served a CAPTCHA — bailing so the chain can fall back.');
        process.exit(4);
      }
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
      console.error('[google-headless] no results parsed (page markup may have changed).');
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
