#!/usr/bin/env bash
# Manage a local Redpanda (Kafka-compatible) broker + Console UI.
#
# Redpanda speaks the Kafka protocol, so the app's aiokafka producer/consumers
# connect to it exactly as they would to Apache Kafka.  We use Redpanda because
# it is a single container (no ZooKeeper, no JVM), light on a laptop, and ships
# a first-class web Console for *visualising* topics, messages, consumer groups
# and lag — which is what the workflow / web-search pub-sub model publishes to.
#
# Two containers on a shared bridge network:
#   * redpanda          — the broker.  Two listeners:
#       - internal://redpanda:9092   (reached by the Console container)
#       - external://localhost:19092 (reached by the app on the host)
#   * redpanda-console  — web UI on http://localhost:${CONSOLE_PORT}
#
# Works with whichever container engine is actually running (auto-detected):
#   * docker   (Docker Desktop, or Rancher Desktop in "dockerd/moby" mode), or
#   * nerdctl  (Rancher Desktop in the default "containerd" mode).
#
# Usage:
#   scripts/redpanda.sh start      # create (if needed) + start broker + console
#   scripts/redpanda.sh stop       # stop both containers
#   scripts/redpanda.sh restart    # stop + start
#   scripts/redpanda.sh status     # container state + broker/topic health
#   scripts/redpanda.sh logs       # tail broker logs
#   scripts/redpanda.sh topics     # list topics (via rpk inside the broker)
#   scripts/redpanda.sh remove     # stop + delete both containers
#
# Env overrides:
#   KAFKA_PORT     host port for the external Kafka listener (default 19092)
#   CONSOLE_PORT   host port for the web Console        (default 18090)
#   REDPANDA_ENGINE  force engine: "docker" or "nerdctl" (default: auto-detect)
set -euo pipefail

KAFKA_PORT="${KAFKA_PORT:-19092}"
CONSOLE_PORT="${CONSOLE_PORT:-18090}"
NET="${REDPANDA_NET:-redpanda-net}"
BROKER="${REDPANDA_NAME:-redpanda}"
CONSOLE="${REDPANDA_CONSOLE_NAME:-redpanda-console}"
RP_IMAGE="${REDPANDA_IMAGE:-redpandadata/redpanda:latest}"
CONSOLE_IMAGE="${REDPANDA_CONSOLE_IMAGE:-redpandadata/console:latest}"
ENGINE=""   # resolved by resolve_engine()

die() { echo "error: $*" >&2; exit 1; }

# Pick a container CLI whose runtime is actually reachable.  Prefer docker when
# its daemon answers; otherwise use Rancher Desktop's nerdctl (containerd).
resolve_engine() {
  if [[ -n "${REDPANDA_ENGINE:-}" ]]; then ENGINE="$REDPANDA_ENGINE"; fi

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

ensure_network() {
  "$ENGINE" network inspect "$NET" >/dev/null 2>&1 || "$ENGINE" network create "$NET" >/dev/null
}

start_broker() {
  if "$ENGINE" ps -a --format '{{.Names}}' | grep -qx "$BROKER"; then
    "$ENGINE" start "$BROKER" >/dev/null
    echo "started existing broker '$BROKER'"
    return
  fi
  # dev-container mode = single node, no replication requirements.  Two Kafka
  # listeners so both the Console (in-network) and the app (on the host) work.
  "$ENGINE" run -d --name "$BROKER" --network "$NET" --restart unless-stopped \
    -p "${KAFKA_PORT}:19092" \
    -p 18081:18081 \
    -p 18082:18082 \
    -p 19644:9644 \
    "$RP_IMAGE" \
    redpanda start \
      --mode dev-container \
      --smp 1 \
      --default-log-level=info \
      --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092 \
      --advertise-kafka-addr "internal://${BROKER}:9092,external://localhost:${KAFKA_PORT}" \
      --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082 \
      --advertise-pandaproxy-addr "internal://${BROKER}:8082,external://localhost:18082" \
      --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081 \
    >/dev/null
  echo "created + started broker '$BROKER'"
}

start_console() {
  if "$ENGINE" ps -a --format '{{.Names}}' | grep -qx "$CONSOLE"; then
    "$ENGINE" start "$CONSOLE" >/dev/null
    echo "started existing console '$CONSOLE'"
    return
  fi
  "$ENGINE" run -d --name "$CONSOLE" --network "$NET" --restart unless-stopped \
    -p "${CONSOLE_PORT}:8080" \
    -e "KAFKA_BROKERS=${BROKER}:9092" \
    "$CONSOLE_IMAGE" >/dev/null
  echo "created + started console '$CONSOLE'"
}

# rpk (bundled in the broker image) is the source of truth for cluster health.
probe_cluster() {
  "$ENGINE" exec "$BROKER" rpk cluster info -X brokers=localhost:9092 >/dev/null 2>&1 && echo ok || echo down
}

start() {
  resolve_engine
  ensure_network
  start_broker
  echo -n "waiting for broker to be ready "
  local ready=""
  for _ in $(seq 1 60); do
    if [[ "$(probe_cluster)" == "ok" ]]; then ready=1; echo "-> OK"; break; fi
    echo -n "."; sleep 1
  done
  [[ -n "$ready" ]] || { echo "-> NOT READY"; die "broker did not come up; check: scripts/redpanda.sh logs"; }
  start_console
  echo ""
  echo "Kafka bootstrap (host apps): localhost:${KAFKA_PORT}"
  echo "Redpanda Console (browser) : http://localhost:${CONSOLE_PORT}"
  status
}

stop() {
  resolve_engine
  for c in "$CONSOLE" "$BROKER"; do
    "$ENGINE" stop "$c" >/dev/null 2>&1 && echo "stopped '$c'" || echo "'$c' not running"
  done
}

remove() {
  resolve_engine
  for c in "$CONSOLE" "$BROKER"; do
    "$ENGINE" rm -f "$c" >/dev/null 2>&1 && echo "removed '$c'" || echo "'$c' not present"
  done
}

logs()   { resolve_engine; "$ENGINE" logs -f --tail 100 "$BROKER"; }
topics() { resolve_engine; "$ENGINE" exec "$BROKER" rpk topic list -X brokers=localhost:9092; }

status() {
  resolve_engine
  echo "broker : $("$ENGINE" ps -a --filter "name=^${BROKER}$" --format '{{.Names}} | {{.Status}} | {{.Ports}}' 2>/dev/null || echo 'absent')"
  echo "console: $("$ENGINE" ps -a --filter "name=^${CONSOLE}$" --format '{{.Names}} | {{.Status}} | {{.Ports}}' 2>/dev/null || echo 'absent')"
  echo "cluster health probe: $(probe_cluster)"
  echo "Kafka bootstrap: localhost:${KAFKA_PORT}   Console: http://localhost:${CONSOLE_PORT}"
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  logs)    logs ;;
  topics)  topics ;;
  remove)  remove ;;
  *) die "unknown command '${1}'. Use: start|stop|restart|status|logs|topics|remove" ;;
esac
