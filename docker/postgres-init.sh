#!/bin/bash
# Set password for argus user on first run
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "ALTER USER argus WITH PASSWORD 'argus_secret';" 2>/dev/null || true
