#!/usr/bin/env bash
#
# websearch.sh — quick CLI web search that just prints the result URLs.
#
# Available backends:
#   google-headless  Real google.com results via Playwright + Chromium + stealth.
#                    ~3-6 s per query. Soft-CAPTCHAs after ~2-3 rapid queries;
#                    resolve once with PW_HEADLESS=0 and the persistent profile
#                    stays cleared for hours. Requires the one-time setup in
#                    ./playwright-google (npm install + playwright install
#                    chromium, ~250 MB on disk). See README.md.
#   searxng          Self-hosted metasearch aggregator. Truly unlimited when
#                    reachable. Reads from ${SEARXNG_URL:-http://localhost:8080}
#                    and expects the JSON output format enabled (see
#                    "SearXNG setup" below).
#   google           Google Custom Search JSON API. Reliable, but capped at
#                    100 queries/day on the free tier. Skipped unless both
#                    GOOGLE_CSE_KEY and GOOGLE_CSE_CX are exported.
#   brave            search.brave.com HTML. No key, ~20 results per query,
#                    soft-blocks after ~7 back-to-back queries.
#   mojeek           mojeek.com HTML. No key. Independent index (own crawler,
#                    not a G/B reseller). Soft-blocks after ~5 rapid queries.
#   ddg              html.duckduckgo.com. No key. Soft-blocks after ~2-5
#                    rapid queries.
#
# --backend auto is the default. It walks a configurable ordered list of
# backends and returns the first one that produced results. The default
# order puts google-headless first so you get real Google ranking whenever
# it's willing, then falls through to searxng, then the paid API, then the
# HTML-scrape fallbacks:
#
#   google-headless → searxng → google → brave → mojeek → ddg
#
# Override the order with WEBSEARCH_BACKENDS (comma-separated), and/or
# temporarily disable specific ones with --disable at call time.
#
# Environment:
#   WEBSEARCH_BACKENDS   Ordered comma-separated backend list for --backend
#                        auto. Set once in ~/.zshrc to lock in your preference.
#                        Example: WEBSEARCH_BACKENDS="searxng,brave,ddg"
#   SEARXNG_URL          Base URL of your SearXNG instance.
#                        Default: http://localhost:8080
#   GOOGLE_CSE_KEY,      Credentials for the paid Custom Search API. Only
#   GOOGLE_CSE_CX        needed if you want the google backend.
#   PW_HEADLESS          Set to 0 to run google-headless with a visible
#                        window (used once to solve a CAPTCHA by hand).
#
# Usage:
#   websearch.sh <query...>
#   websearch.sh --backend {auto|google-headless|searxng|google|brave|mojeek|ddg} <query...>
#   websearch.sh --disable ddg,mojeek <query...>
#   websearch.sh -n 15 <query...>
#   websearch.sh --setup-searxng           # print SearXNG Docker instructions
#
# Options:
#   -b, --backend BACKEND   Which engine to hit (default: auto).
#   -n, --num N             Max results to print (default: 20).
#   -v, --verbose           Print which engine each URL came from on stderr.
#   -d, --disable LIST      Comma-separated backends to skip in --backend auto
#                           this invocation only. Applies on top of any
#                           WEBSEARCH_BACKENDS ordering.
#       --setup-searxng     Print the SearXNG local-setup instructions and exit.
#   -h, --help              This help.

set -uo pipefail

# Resolve the script's own directory so the Playwright driver location is
# not tied to any single machine layout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

backend='auto'
num=20
verbose=false
disable_list=''

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--backend)      backend="$2";      shift 2 ;;
    -n|--num)          num="$2";          shift 2 ;;
    -v|--verbose)      verbose=true;      shift ;;
    -d|--disable)      disable_list="$2"; shift 2 ;;
       --setup-searxng)
         sed -n '/^# SearXNG setup:/,/^# END SearXNG setup/p' "$0" | sed 's/^# \{0,1\}//' | sed '/END SearXNG setup/d'
         exit 0 ;;
    -h|--help)         sed -n '/^# websearch.sh/,/^#   -h,/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $(basename "$0") [-b BACKEND] [-n N] [-d LIST] <query...>" >&2
  exit 1
fi

query="$*"
ua='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
q=$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote_plus(sys.argv[1]))' "$query")
export WS_NUM="$num"

DEFAULT_BACKENDS='google-headless,searxng,google,brave,mojeek,ddg'
BACKEND_ORDER="${WEBSEARCH_BACKENDS:-$DEFAULT_BACKENDS}"

is_disabled() {
  # Returns 0 (true) if $1 appears in the disable_list.
  [[ -z "$disable_list" ]] && return 1
  case ",${disable_list}," in *,"$1",*) return 0 ;; esac
  return 1
}

# ── Per-engine functions ──────────────────────────────────────────────────
# Each writes URLs to stdout, one per line, and returns:
#   0  → success, at least one URL printed
#   3  → transport error / dependency missing / not reachable (skip in auto)
#   4  → engine served a challenge / block page (skip in auto)
# Empty stdout with rc=0 is also treated as "empty, move on" by the auto loop.

fetch_google() {
  if [[ -z "${GOOGLE_CSE_KEY:-}" || -z "${GOOGLE_CSE_CX:-}" ]]; then
    echo "[google] GOOGLE_CSE_KEY / GOOGLE_CSE_CX not set" >&2
    return 3
  fi
  curl -sSL "https://www.googleapis.com/customsearch/v1?key=${GOOGLE_CSE_KEY}&cx=${GOOGLE_CSE_CX}&q=${q}&num=$(( num > 10 ? 10 : num ))" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin)
if "error" in data:
    print(f"[google] {data[\"error\"].get(\"message\")}", file=sys.stderr); sys.exit(3)
for item in data.get("items", []):
    print(item["link"])
'
}

fetch_searxng() {
  local base="${SEARXNG_URL:-http://localhost:8080}"
  local body
  body=$(curl -sSL --connect-timeout 2 --max-time 15 \
    "${base%/}/search?q=${q}&format=json&safesearch=0" 2>/dev/null) || {
    echo "[searxng] not reachable at $base (run: websearch.sh --setup-searxng)" >&2
    return 3
  }
  printf '%s' "$body" | python3 -c '
import sys, json, os
try:
    data = json.loads(sys.stdin.read())
except Exception as e:
    print(f"[searxng] bad JSON — is `search.formats: [html, json]` enabled? ({e})", file=sys.stderr); sys.exit(3)
limit = int(os.environ.get("WS_NUM", "20"))
seen, count = set(), 0
for r in data.get("results", []):
    u = r.get("url", "")
    if u and u.startswith(("http://", "https://")) and u not in seen:
        seen.add(u); print(u); count += 1
        if count >= limit: break
'
}

fetch_google_headless() {
  local driver_dir="$SCRIPT_DIR/playwright-google"
  local script="$driver_dir/search.js"
  if [[ ! -f "$script" ]]; then
    echo "[google-headless] driver not installed. One-time setup:" >&2
    echo "    cd \"$driver_dir\" && npm install && npx playwright install chromium" >&2
    return 3
  fi
  if [[ ! -d "$driver_dir/node_modules/playwright" ]]; then
    echo "[google-headless] node_modules missing. Run:" >&2
    echo "    cd \"$driver_dir\" && npm install" >&2
    return 3
  fi
  # Bubble the node exit code up so try_engine can distinguish rate-limit (4)
  # from real failures (3).
  WS_NUM="$num" node "$script" "$query"
}

fetch_brave() {
  local html
  html=$(curl -sSL -A "$ua" -H 'Accept-Language: en-US,en;q=0.9' \
    "https://search.brave.com/search?q=${q}&source=web") || return 3
  printf '%s' "$html" | python3 -c '
import sys, re, os
from urllib.parse import urlparse
html = sys.stdin.read()
limit = int(os.environ.get("WS_NUM", "20"))
if re.search(r"challenge|verify.you.re.human|captcha", html, re.I) and len(html) < 100000:
    print("[brave] rate-limited (challenge page)", file=sys.stderr); sys.exit(4)
urls, seen = [], set()
for m in re.finditer(r"data-type=\"web\"[\s\S]{0,500}?<a[^>]+href=\"(https?://[^\"]+)\"", html):
    u = m.group(1)
    host = urlparse(u).netloc.lower()
    if any(bad in host for bad in ("brave.com", "search.brave.com")): continue
    if u in seen: continue
    seen.add(u); urls.append(u)
    if len(urls) >= limit: break
for u in urls: print(u)
'
}

fetch_mojeek() {
  local html
  html=$(curl -sSL -A "$ua" -H 'Accept-Language: en-US,en;q=0.9' \
    "https://www.mojeek.com/search?q=${q}") || return 3
  printf '%s' "$html" | python3 -c '
import sys, re, os
html = sys.stdin.read()
limit = int(os.environ.get("WS_NUM", "20"))
if "403 - Forbidden" in html or "automated queries" in html.lower():
    print("[mojeek] rate-limited (403)", file=sys.stderr); sys.exit(4)
urls, seen = [], set()
for m in re.finditer(r"<a\s+class=\"title\"[^>]+href=\"([^\"]+)\"", html):
    u = m.group(1)
    if u.startswith(("http://", "https://")) and u not in seen:
        seen.add(u); urls.append(u)
    if len(urls) >= limit: break
for u in urls: print(u)
'
}

fetch_ddg() {
  local html
  html=$(curl -sSL -A "$ua" -H 'Accept-Language: en-US,en;q=0.9' \
    "https://html.duckduckgo.com/html/?q=${q}") || return 3
  printf '%s' "$html" | python3 -c '
import sys, re, os
from urllib.parse import urlparse, parse_qs, unquote
html = sys.stdin.read()
limit = int(os.environ.get("WS_NUM", "20"))
if "anomaly" in html.lower() or "challenge-submit" in html or "please solve" in html.lower():
    print("[ddg] rate-limited (challenge page)", file=sys.stderr); sys.exit(4)
urls, seen = [], set()
for m in re.finditer(r"class=\"result__a\"[^>]+href=\"([^\"]+)\"", html):
    href = m.group(1)
    if href.startswith("//"): href = "https:" + href
    p = urlparse(href)
    real = None
    if "duckduckgo.com" in p.netloc and p.path.startswith("/l/"):
        qs = parse_qs(p.query)
        if "uddg" in qs: real = unquote(qs["uddg"][0])
    else:
        real = href
    if not real or not real.startswith(("http://", "https://")): continue
    host = urlparse(real).netloc.lower()
    if any(x in host for x in ("duckduckgo.com", "bing.com/aclick")): continue
    if real in seen: continue
    seen.add(real); urls.append(real)
    if len(urls) >= limit: break
for u in urls: print(u)
'
}

# ── Dispatch ──────────────────────────────────────────────────────────────

try_engine() {
  local name="$1" fn="$2"
  local out rc
  out=$("$fn"); rc=$?
  if [[ $rc -eq 0 && -n "$out" ]]; then
    if $verbose; then echo "[$name] hit" >&2; fi
    printf '%s\n' "$out"
    return 0
  fi
  if $verbose; then
    local emptiness=empty; [[ -n $out ]] && emptiness=has-output
    echo "[$name] rc=$rc, $emptiness" >&2
  fi
  return 1
}

run_backend() {
  case "$1" in
    google-headless) fetch_google_headless ;;
    searxng)         fetch_searxng ;;
    google)          fetch_google ;;
    brave)           fetch_brave ;;
    mojeek)          fetch_mojeek ;;
    ddg)             fetch_ddg ;;
    *) echo "[websearch] unknown backend: $1" >&2; return 3 ;;
  esac
}

if [[ "$backend" != 'auto' ]]; then
  if is_disabled "$backend"; then
    echo "[websearch] backend '$backend' is in --disable list" >&2
    exit 2
  fi
  run_backend "$backend"
  exit $?
fi

# ── auto: walk the ordered list until something returns results ───────────
IFS=',' read -ra order <<< "$BACKEND_ORDER"
tried=()
for raw in "${order[@]}"; do
  b="${raw// /}"  # trim spaces
  [[ -z "$b" ]] && continue
  if is_disabled "$b"; then
    if $verbose; then echo "[$b] disabled, skipping" >&2; fi
    continue
  fi
  tried+=("$b")
  try_engine "$b" "$(printf 'fetch_%s' "${b//-/_}")" && exit 0
done

echo "[websearch] no results from any enabled backend (tried: ${tried[*]:-none})." >&2
echo "            Wait a minute and retry, adjust WEBSEARCH_BACKENDS, or run" >&2
echo "            'websearch.sh --setup-searxng' for a truly unlimited local option." >&2
exit 4

# ─────────────────────────────────────────────────────────────────────────
# SearXNG setup:
#
# SearXNG is a self-hosted metasearch engine. When it's running locally,
# the `searxng` backend gives you effectively unlimited free search because
# each query fans out to ~70 upstream engines behind the scenes; no single
# upstream sees enough traffic from you to rate-limit.
#
# One-time setup with Docker:
#
#     mkdir -p ~/searxng && cd ~/searxng
#     docker run --rm --name searxng -d \
#       -p 8080:8080 \
#       -v "$PWD:/etc/searxng" \
#       -e "BASE_URL=http://localhost:8080/" \
#       -e "INSTANCE_NAME=local" \
#       searxng/searxng
#
# On first launch it writes ~/searxng/settings.yml. Open that file and add
# `json` to the enabled output formats — the default is HTML-only:
#
#     search:
#       formats:
#         - html
#         - json
#
# Then restart:
#
#     docker restart searxng
#
# Test it:
#
#     curl -s 'http://localhost:8080/search?q=hello&format=json' | jq '.results | length'
#
# From that point on, `websearch.sh --backend searxng "..."` (or the auto
# chain when it falls through to searxng) will use it.
#
# END SearXNG setup
