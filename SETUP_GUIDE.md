# Argus Prime - Quick Start Guide

## Prerequisites

Before running Argus Prime, ensure you have:

1. **Python 3.10+** with pip
2. **PostgreSQL 15+** (for database)
3. **Redis 7+** (optional, for caching)
4. **Node.js 18+** (for frontend)
5. **Docker & Docker Compose** (optional, recommended)

---

## Option 1: Docker Compose (Recommended)

### Start All Services
```bash
cd /Users/Dataghost/Argus_Prime
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Redis on port 6379  
- Backend API on port 8000

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

---

## Option 2: Manual Setup

### Step 1: Install PostgreSQL

**Using Homebrew (macOS):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Or download from:**
https://www.postgresql.org/download/

### Step 2: Create Database

```bash
# Create user and database
createuser -s argus
createdb argus_prime -O argus
psql -c "ALTER USER argus WITH PASSWORD 'argus_secret';"
```

### Step 3: Install Redis (Optional)

```bash
brew install redis
brew services start redis
```

### Step 4: Create Environment File

```bash
cd /Users/Dataghost/Argus_Prime
cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://argus:argus_secret@localhost:5432/argus_prime
REDIS_HOST=localhost
REDIS_PORT=6379
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false
EOF
```

### Step 5: Create Database Schema

```bash
psql -U argus -d argus_prime << 'EOF'
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_size INTEGER NOT NULL,
    content_type VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_file_hash ON analyses(file_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);

CREATE TABLE IF NOT EXISTS video_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    frames_analyzed INTEGER NOT NULL DEFAULT 0,
    faces_detected INTEGER NOT NULL DEFAULT 0,
    overall_score FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    is_manipulated BOOLEAN NOT NULL DEFAULT FALSE,
    branch_scores JSONB,
    per_frame_scores JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audio_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    duration_seconds FLOAT NOT NULL DEFAULT 0.0,
    segments_analyzed INTEGER NOT NULL DEFAULT 0,
    overall_score FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    is_manipulated BOOLEAN NOT NULL DEFAULT FALSE,
    per_segment_scores JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS text_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    word_count INTEGER NOT NULL DEFAULT 0,
    language_detected VARCHAR(10) NOT NULL DEFAULT 'unknown',
    overall_score FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    is_manipulated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trust_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    overall_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    confidence_level VARCHAR(20) NOT NULL,
    verdict VARCHAR(30) NOT NULL,
    breakdown JSONB,
    fused_score FLOAT NOT NULL,
    weights_used JSONB,
    consistency_score FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS share_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    share_token VARCHAR(100) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_share_links_token ON share_links(share_token);
EOF
```

### Step 6: Create Missing Static Files

```bash
cd /Users/Dataghost/Argus_Prime/frontend/public

# Create manifest.json
cat > manifest.json << 'EOF'
{
  "name": "Argus Prime",
  "short_name": "Argus",
  "description": "Multi-Modal Deepfake Detection & Forensic Platform",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "icons": [{ "src": "/icon.svg", "sizes": "any", "type": "image/svg+xml" }]
}
EOF

# Create grid.svg
cat > grid.svg << 'EOF'
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="100" height="100" fill="url(#grid)"/>
</svg>
EOF

# Create icon.svg
cat > icon.svg << 'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none">
  <rect width="32" height="32" rx="4" fill="#000000"/>
  <path d="M8 12h16M8 16h16M8 20h10" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
</svg>
EOF

echo "Static files created!"
```

### Step 7: Install Dependencies

```bash
cd /Users/Dataghost/Argus_Prime

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -e .

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Step 8: Run the Application

**Terminal 1 - Backend:**
```bash
cd /Users/Dataghost/Argus_Prime
source venv/bin/activate
python src/main.py
```

**Terminal 2 - Frontend:**
```bash
cd /Users/Dataghost/Argus_Prime/frontend
npm run dev
```

---

## Verification

Once running, check:

1. **Backend Health:** http://localhost:8000/api/v1/health
2. **API Docs:** http://localhost:8000/docs
3. **Frontend:** http://localhost:3000

---

## Troubleshooting

### "Connection refused" on port 5432
PostgreSQL is not running:
```bash
brew services start postgresql@15
# or
docker run -d --name argus-postgres -e POSTGRES_USER=argus -e POSTGRES_PASSWORD=argus_secret -p 5432:5432 postgres:15-alpine
```

### "Connection refused" on port 6379
Redis is not running:
```bash
brew services start redis
# or
docker run -d --name argus-redis -p 6379:6379 redis:7-alpine
```

### CORS Errors
Ensure CORS origins match. Backend should allow:
```python
allow_origins=["http://localhost:3000"]
```

### Database Schema Errors
Recreate the schema:
```bash
psql -U argus -d argus_prime -c "DROP TABLE IF EXISTS analyses CASCADE;"
# Then run the schema creation commands from Step 5
```
