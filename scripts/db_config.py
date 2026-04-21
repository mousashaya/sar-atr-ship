"""
scripts/db_config.py
====================
Database credentials — loaded from .env at project root.
Imported by train.py, dataset.py, ingest_atrnet.py.
"""

import os
from pathlib import Path


def _load_env():
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

DB_CONFIG = {
    'host':     os.environ.get('DB_HOST', 'localhost'),
    'port':     int(os.environ.get('DB_PORT', '5432')),
    'dbname':   os.environ.get('DB_NAME', 'fungidb'),
    'user':     os.environ.get('DB_USER', ''),
    'password': os.environ.get('DB_PASSWORD', ''),
}
