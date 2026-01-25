#!/bin/bash
# Argus Prime - Services Startup Script
# Ensures all required services are running

set -e

echo "🚀 Starting Argus Prime Services..."

# Start PostgreSQL if not running
if ! service postgresql status > /dev/null 2>&1; then
    echo "📦 Starting PostgreSQL..."
    service postgresql start
    sleep 2
fi

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
timeout=30
while ! sudo -u postgres psql -c "SELECT 1" >/dev/null 2>&1; do
    sleep 1
    timeout=$((timeout - 1))
    if [ $timeout -le 0 ]; then
        echo "❌ PostgreSQL failed to start"
        exit 1
    fi
done
echo "✅ PostgreSQL is ready"

# Check if database exists
if ! sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw argus_prime; then
    echo "📦 Creating database..."
    sudo -u postgres psql -c "CREATE USER argus WITH PASSWORD 'argus';" || true
    sudo -u postgres psql -c "CREATE DATABASE argus_prime OWNER argus;" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE argus_prime TO argus;"
    
    # Initialize tables
    echo "📦 Initializing database tables..."
    cd /app/backend && python -m src.db.init_db
fi

echo "✅ All services ready!"
