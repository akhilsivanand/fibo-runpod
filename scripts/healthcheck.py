#!/usr/bin/env python3
"""Readiness probe; contains no secrets and never reads logs."""
import os
import sys
from urllib.request import urlopen
from fibo_runtime.gateway import auth_from_environment

try:
    auth = auth_from_environment(os.environ.get('FIBO_MODE', 'pod'))
    with urlopen(f'http://127.0.0.1:{auth.port}/readyz', timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
