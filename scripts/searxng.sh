#!/usr/bin/env bash
# Manage a local SearXNG instance for the web-research search backend.
#
# SearXNG fans each query out to ~70 upstream engines and returns JSON, which
# is what trading_crew/data_sources/web_research.py's `searxng` backend hits at
#   http://localhost:${SEARXNG_PORT}/search?q=...&format=json
#
# Works with whichever container engine is actually running:
#   * docker   (Docker Desktop, or Rancher Desktop in "dockerd/moby" mode), or
#   * nerdctl  (Rancher Desktop in the default "containerd" mode).
# It auto-detects the live engine; no need to switch Rancher's engine setting.
#
# On first run it writes a minimal settings.yml that:
#   * enables the JSON output format (off by default — our code needs it), and
#   * disables the built-in limiter (a single self-hosted instance feeding our
#     own batch shouldn't 429 itself; rotation/cooldown in web_research.py still
#     covers the public-instance case).
#
# Usage:
#   scripts/searxng.sh start      # create (if needed) + start on :18080
#   scripts/searxng.sh stop       # stop the container
#   scripts/searxng.sh restart    # stop + start
#   scripts/searxng.sh status     # container state + JSON health probe
#   scripts/searxng.sh logs       # tail container logs
#   scripts/searxng.sh remove     # stop + delete the container (keeps config)
#
# Env overrides:
#   SEARXNG_PORT   host port to expose (default 18080)
#   SEARXNG_DIR    config dir on the host (default ~/searxng)
#   SEARXNG_NAME   container name (default searxng)
#   SEARXNG_ENGINE force engine: "docker" or "nerdctl" (default: auto-detect)
#   TRADINGCREW_CA_BUNDLE  CA bundle to trust for upstream TLS (default
#                          ~/.trading_crew/ca-bundle.pem, built by run_web.sh).
#                          Needed behind a TLS-inspecting corporate proxy so the
#                          container's engines don't fail cert verification.
set -euo pipefail

PORT="${SEARXNG_PORT:-18080}"
DIR="${SEARXNG_DIR:-$HOME/searxng}"
NAME="${SEARXNG_NAME:-searxng}"
IMAGE="searxng/searxng:latest"
CA_BUNDLE="${TRADINGCREW_CA_BUNDLE:-$HOME/.trading_crew/ca-bundle.pem}"
ENGINE=""   # resolved by resolve_engine()

die() { echo "error: $*" >&2; exit 1; }

# Pick a container CLI whose runtime is actually reachable.  Prefer docker when
# its daemon answers; otherwise use Rancher Desktop's nerdctl (containerd).
resolve_engine() {
  if [[ -n "${SEARXNG_ENGINE:-}" ]]; then ENGINE="$SEARXNG_ENGINE"; fi

  if [[ -z "$ENGINE" ]] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    ENGINE="docker"
  fi
  if [[ -z "$ENGINE" ]]; then
    local nc
    nc="$(command -v nerdctl || echo "$HOME/.rd/bin/nerdctl")"
    if [[ -x "$nc" ]] && "$nc" info >/dev/null 2>&1; then ENGINE="$nc"; fi
  fi

  # Nothing reachable yet — try to boot an installed engine, then re-probe.
  if [[ -z "$ENGINE" ]]; then
    echo "No container runtime reachable — attempting to start one..." >&2
    if [[ "$(uname)" == "Darwin" ]]; then
      if command -v rdctl >/dev/null 2>&1 || [[ -x "$HOME/.rd/bin/rdctl" ]]; then
        "$(command -v rdctl || echo "$HOME/.rd/bin/rdctl")" start >/dev/null 2>&1 || \
          open -a "Rancher Desktop" 2>/dev/null || true
      elif [[ -d "/Applications/Docker.app" ]]; then
        open -a Docker 2>/dev/null || true
      fi
    fi
    echo -n "waiting for a container runtime " >&2
    local nc; nc="$(command -v nerdctl || echo "$HOME/.rd/bin/nerdctl")"
    for _ in $(seq 1 90); do
      if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then ENGINE="docker"; break; fi
      if [[ -x "$nc" ]] && "$nc" info >/dev/null 2>&1; then ENGINE="$nc"; break; fi
      echo -n "." >&2; sleep 2
    done
    echo "" >&2
  fi

  [[ -n "$ENGINE" ]] || die "No usable container engine. Open Rancher Desktop (or Docker) and retry."
  echo "using container engine: $ENGINE" >&2
}

ensure_config() {
  mkdir -p "$DIR"
  local cfg="$DIR/settings.yml"
  if [[ ! -f "$cfg" ]]; then
    local secret
    secret="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p | tr -d '\n')"
    cat > "$cfg" <<YML
# Minimal SearXNG config for programmatic JSON search (managed by scripts/searxng.sh).
use_default_settings: true
server:
  secret_key: "${secret}"
  limiter: false          # self-hosted single-tenant: don't 429 ourselves
  image_proxy: false
search:
  formats:                # JSON is OFF by default; our code needs it
    - html
    - json
  # Under a high-volume batch the upstream engines (duckduckgo / bing / brave /
  # startpage / google) intermittently answer with CAPTCHA or "too many
  # requests".  SearXNG's DEFAULT reaction is to SUSPEND that engine for 24h
  # (SearxEngineCaptcha / SearxEngineAccessDenied = 86400s), so a single trip
  # kills the engine for the rest of the day and the JSON API starts returning
  # 0 results.  Drop the suspension to ~1 min so engines recover fast and the
  # aggregate keeps returning links (maintainer-recommended; searxng#2750).
  suspended_times:
    SearxEngineAccessDenied: 60
    SearxEngineCaptcha: 60
    SearxEngineTooManyRequests: 120
    cf_SearxEngineCaptcha: 60
    cf_SearxEngineAccessDenied: 60
    recaptcha_SearxEngineCaptcha: 60
outgoing:
  # Give slow engines a little longer to answer so they return results instead
  # of timing out to empty (the default request_timeout is only ~3s).
  request_timeout: 6.0
  max_request_timeout: 15.0
  pool_connections: 100
  pool_maxsize: 20
  enable_http2: true
YML
    echo "wrote $cfg"
  fi
}

probe_json() {
  curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/search?q=test&format=json" 2>/dev/null || echo 000
}

probe_results() {
  curl -s "http://localhost:${PORT}/search?q=apple+stock+news&format=json" 2>/dev/null \
    | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('results',[])))" 2>/dev/null || echo 0
}

# Discover certifi's cacert.pem inside the image so we can shadow it with our
# combined bundle.  httpx (SearXNG's HTTP client) verifies against certifi, so
# behind a TLS-inspecting proxy every upstream engine fails cert verification
# (HTTP 200 but zero results) unless certifi trusts the corporate root.
ca_mount_args() {
  [[ -f "$CA_BUNDLE" ]] || { echo "" ; return; }
  local certifi
  certifi="$("$ENGINE" run --rm --entrypoint /usr/local/searxng/.venv/bin/python \
    "$IMAGE" -c 'import certifi;print(certifi.where())' 2>/dev/null | tr -d '\r')"
  [[ -n "$certifi" ]] || { echo "warn: could not resolve certifi path in image; upstream TLS may fail" >&2; echo ""; return; }
  echo "shadowing container certifi ($certifi) with $CA_BUNDLE" >&2
  printf -- '-v\n%s:%s:ro\n-e\nSSL_CERT_FILE=%s\n-e\nREQUESTS_CA_BUNDLE=%s\n' \
    "$CA_BUNDLE" "$certifi" "$certifi" "$certifi"
}

start() {
  resolve_engine
  ensure_config
  if "$ENGINE" ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    "$ENGINE" start "$NAME" >/dev/null
    echo "started existing container '$NAME'"
  else
    local ca_args=()
    while IFS= read -r line; do [[ -n "$line" ]] && ca_args+=("$line"); done < <(ca_mount_args)
    "$ENGINE" run -d --name "$NAME" --restart unless-stopped \
      -p "${PORT}:8080" \
      -v "${DIR}:/etc/searxng" \
      -e "SEARXNG_BASE_URL=http://localhost:${PORT}/" \
      "${ca_args[@]}" \
      "$IMAGE" >/dev/null
    echo "created + started container '$NAME'"
  fi
  echo -n "waiting for JSON endpoint on :${PORT} "
  local ready=""
  for _ in $(seq 1 40); do
    if [[ "$(probe_json)" == "200" ]]; then ready=1; echo "-> OK"; break; fi
    echo -n "."; sleep 1
  done
  [[ -n "$ready" ]] || { echo "-> NOT READY"; die "SearXNG did not answer; check: scripts/searxng.sh logs"; }
  echo -n "checking engines return results "
  for _ in $(seq 1 15); do
    if [[ "$(probe_results)" -gt 0 ]]; then echo "-> OK"; status; return 0; fi
    echo -n "."; sleep 2
  done
  echo "-> ZERO RESULTS"
  echo "warn: endpoint healthy but upstream engines returned nothing — usually a" >&2
  echo "      TLS/cert issue (check: scripts/searxng.sh logs | grep -i cert)." >&2
  status
}

stop()   { resolve_engine; "$ENGINE" stop "$NAME" >/dev/null 2>&1 && echo "stopped '$NAME'" || echo "'$NAME' not running"; }
remove() { resolve_engine; "$ENGINE" rm -f "$NAME" >/dev/null 2>&1 && echo "removed '$NAME' (config kept in $DIR)" || echo "'$NAME' not present"; }
logs()   { resolve_engine; "$ENGINE" logs -f --tail 100 "$NAME"; }

status() {
  resolve_engine
  echo "container: $("$ENGINE" ps -a --filter "name=${NAME}" --format '{{.Names}} | {{.Status}} | {{.Ports}}' 2>/dev/null || echo 'absent')"
  echo "json probe (http://localhost:${PORT}/search?format=json): $(probe_json)"
  echo "results probe ('apple stock news'): $(probe_results)"
  echo "config dir: $DIR"
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  logs)    logs ;;
  remove)  remove ;;
  *) die "unknown command '${1}'. Use: start|stop|restart|status|logs|remove" ;;
esac
