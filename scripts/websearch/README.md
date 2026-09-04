# websearch — free CLI web search for the LegalGuard investigation step

Command-line search that returns a plain list of result URLs, one per line.
Multiple backends behind a single entrypoint, auto-fallback chain, no paid
API keys required for the default configuration.

Written as scratch-tooling for the R&D on
[SAASAI-904 — Improve open-web search-result quality in the review pipeline](https://jira.nutanix.com/browse/SAASAI-904).
Used to sanity-check different search sources against the same query so we
can pick which one(s) to plug into the LegalGuard agent's investigation
step.

## Backends

| Backend | Needs | Realistic ceiling | Result quality |
|---|---|---|---|
| `google-headless` | Playwright + Chromium (one-time 250 MB) | ~2–3 rapid queries before CAPTCHA; hours of clean queries after one manual solve | Real google.com |
| `searxng` | Self-hosted SearXNG container | Effectively unlimited | Aggregated across ~70 upstreams |
| `google` | `GOOGLE_CSE_KEY` + `GOOGLE_CSE_CX` env vars | 100 queries / day free tier | Google Custom Search index |
| `brave` | none | ~7 rapid queries before soft-block | Brave web index |
| `mojeek` | none | ~5 rapid queries before 403 | Mojeek's own crawler |
| `ddg` | none | ~2–5 rapid queries before challenge | DuckDuckGo |

`--backend auto` (the default) walks an ordered list and returns the first
non-empty result. Default order is
`google-headless → searxng → google → brave → mojeek → ddg`, which prefers
real Google results whenever it can and quietly falls back to the free
scrape engines when Google balks.

## Install

Nothing extra is needed for `brave`, `mojeek`, `ddg` — plain `curl` + a
Python 3 that ships on macOS.

### Optional: `google-headless` (recommended)

```bash
cd tools/websearch/playwright-google
npm install
npx playwright install chromium
```

That's a ~250 MB one-time download (Chromium + a headless-shell binary in
`~/Library/Caches/ms-playwright/`). Everything else is under
`playwright-google/node_modules/` which is gitignored.

### Optional: `searxng`

For unlimited-frequency searching across many engines at once, run a
local SearXNG:

```bash
mkdir -p ~/searxng && cd ~/searxng
docker run --rm --name searxng -d \
  -p 8080:8080 \
  -v "$PWD:/etc/searxng" \
  -e "BASE_URL=http://localhost:8080/" \
  -e "INSTANCE_NAME=local" \
  searxng/searxng
```

Then edit `~/searxng/settings.yml` to enable JSON output (the default is
HTML-only):

```yaml
search:
  formats:
    - html
    - json
```

Then `docker restart searxng`. Sanity check:

```bash
curl -s 'http://localhost:8080/search?q=hello&format=json' | jq '.results | length'
```

Point `websearch.sh` at a non-default URL via `SEARXNG_URL` if it's not on
`localhost:8080`.

### Optional: `google` (paid Custom Search API)

Create keys at [programmablesearchengine.google.com](https://programmablesearchengine.google.com/),
then export:

```bash
export GOOGLE_CSE_KEY='AIza…'
export GOOGLE_CSE_CX='0123456789:abcdefg'
```

Free tier is 100 queries / day per key.

## Use

```bash
# auto chain (default order or your WEBSEARCH_BACKENDS)
./websearch.sh "site:github.com typescript decorators"

# force one backend
./websearch.sh --backend brave "site:reddit.com jira"

# skip one for this call
./websearch.sh --disable ddg,mojeek "..."

# cap the result count
./websearch.sh -n 5 "..."

# see which engine each URL came from
./websearch.sh -v "..."
```

Set a permanent preferred order in your shell rc:

```bash
export WEBSEARCH_BACKENDS='google-headless,searxng,brave,mojeek,ddg'
```

## When `google-headless` hits a CAPTCHA

Google will rate-limit *any* single home IP after a small handful of
rapid searches, even through a headless browser. When it does:

```bash
PW_HEADLESS=0 ./playwright-google/search.js "any query"
```

Solve the CAPTCHA in the visible window that pops up, close it, and the
persistent profile at `~/.cache/playwright-google/` stays trusted for
hours of subsequent headless queries.

## Exit codes

- `0` — one or more URLs printed.
- `1` — bad CLI usage.
- `2` — the backend you forced is disabled.
- `3` — transport / config error (missing dep, no keys, endpoint unreachable).
- `4` — every enabled backend was rate-limited or empty.

The auto chain treats `3` and `4` as "move on to the next backend", so a
non-zero exit only leaks up when every enabled engine has failed.

## Layout

```
tools/websearch/
├── websearch.sh                 # entrypoint (bash + inline Python parsers)
├── playwright-google/
│   ├── package.json             # pinned playwright + playwright-extra + stealth
│   ├── package-lock.json
│   └── search.js                # Chromium driver
├── .gitignore                   # keeps node_modules out
└── README.md
```
