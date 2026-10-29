#!/usr/bin/env bash
set -Eeuo pipefail

case "${FIBO_MODE:-pod}" in
  pod)
    /opt/venv/bin/python - <<'PY'
import sys
from fibo_runtime.gateway import auth_from_environment
try:
    auth_from_environment()
except ValueError as error:
    print(str(error), file=sys.stderr)
    sys.exit(2)
PY
    # Let the official Runpod base initialize SSH before the foreground app.
    # The hook acknowledges initialization; it does not background the app.
    FIBO_START_SIGNAL="$(mktemp /tmp/fibo-base-ready.XXXXXX)"
    export FIBO_START_SIGNAL
    # Use Runpod's SSH setup. The foreground supervisor starts Jupyter on 8888
    # with private environment/config credentials instead of command arguments.
    env -u JUPYTER_PASSWORD -u JUPYTER_TOKEN -u JUPYTER_DISABLE_AUTH -u FIBO_PASSWORD -u FILEBROWSER_PASSWORD /start.sh &
    base_pid=$!
    for attempt in {1..120}; do
      [[ -s "$FIBO_START_SIGNAL" ]] && break
      kill -0 "$base_pid" 2>/dev/null || { printf 'Runpod base initialization exited.\n' >&2; exit 1; }
      sleep 0.5
    done
    [[ -s "$FIBO_START_SIGNAL" ]] || { printf 'Runpod base initialization timed out.\n' >&2; exit 1; }
    rm -f -- "$FIBO_START_SIGNAL"
    unset FIBO_START_SIGNAL
    ;;
  serverless) ;;
  *) printf '%s\n' 'FIBO_MODE must be pod or serverless.' >&2; exit 2 ;;
esac

# A failed application now fails the container. tini forwards signals/reaps
# children; the runtime supervises ComfyUI and the Hub worker together.
exec /opt/venv/bin/python -m fibo_runtime.bootstrap "$@"
