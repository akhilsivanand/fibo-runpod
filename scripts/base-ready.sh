#!/usr/bin/env bash
set -euo pipefail
if [[ -n ${FIBO_START_SIGNAL:-} ]]; then
  printf 'ready\n' > "$FIBO_START_SIGNAL"
fi
