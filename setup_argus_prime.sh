#!/bin/bash

# Argus Prime - Complete Setup Script
# This script sets up PostgreSQL, Redis, and all required dependencies

set -e

echo "=============================================="
echo "     ARGUS PRIME - COMPLETE SETUP SCRIPT"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_NAME="argus_prime"
DB_USER="argus"
DB_PASS="argus_secret"
DB_HOST="localhost"
DB_PORT="5432"

# ==============================================================================
# STEP 1: PostgreSQL Setup
# ==============================================================================
echo -e "${YELLOW}STEP 1: PostgreSQL Setup${NC}"
echo "----------------------------------------"

check_postgres() {
    if command -v psql &> /dev/null; then
        return 0
    fi
    return 1
}

setup_postgres_docker() {
    echo "Setting up PostgreSQL with Docker..."
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
    
    # Stop any existing containers
    docker stop argus-postgres 2>/dev/null || true
    docker rm argus-postgres 2>/dev/null || true
    
    # Start PostgreSQL container
    echo "Starting PostgreSQL container..."
    docker run -d \
        --name argus-postgres \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASS" \
        -e POSTGRES_DB="$DB_NAME" \
        -p "${DB_PORT}:5432" \
        -v argus_postgres_data:/var/lib/postgresql/data \
        postgres:15-alpine
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec argus-postgres pg_isready -U "$DB_USER" > /dev/null 2>&1; then
            echo -e "${GREEN}PostgreSQL is ready!${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}Error: PostgreSQL failed to start${NC}"
    exit 1
}

setup_postgres_native() {
    echo "Setting up PostgreSQL natively..."
    
    # Check if Homebrew is available
    if command -v brew &> /dev/null; then
        echo "Installing PostgreSQL via Homebrew..."
        brew install postgresql@15
        
        # Start PostgreSQL service
        if brew services list | grep -q "postgresql@15"; then
            brew services stop postgresql@15 2>/dev/null || true
        fi
        
        # Initialize database
        initdb /usr/local/var/postgresql@15 -E utf8 || true
        
        # Start service
        brew services start postgresql@15
        
        # Wait for PostgreSQL to be ready
        sleep 3
        
        # Create database and user
        echo "Creating database and user..."
        createuser -s "$DB_USER" 2>/dev/null || true
        createdb "$DB_NAME" -O "$DB_USER" 2>/dev/null || true
        psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASS';" 2>/dev/null || true
        
        echo -e "${GREEN}PostgreSQL setup complete!${NC}"
        return 0
    fi
    
    echo -e "${RED}Error: Homebrew not found. Please install PostgreSQL manually.${NC}"
    exit 1
}

# Main PostgreSQL setup logic
if check_postgres; then
    echo "PostgreSQL is already installed."
    # Check if our database exists
    if psql -lqt -U "$DB_USER" -h "$DB_HOST" | cut -d\| -f1 | grep -qw "$DB_NAME"; then
        echo "Database '$DB_NAME' already exists."
    else
        echo "Creating database '$DB_NAME'..."
        createdb "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" 2>/dev/null || \
        psql -c "CREATE DATABASE $DB_NAME;" -U "$DB_USER" -h "$DB_HOST"
        echo -e "${GREEN}Database created!${NC}"
    fi
else
    echo "PostgreSQL not found. Installing..."
    if command -v docker &> /dev/null; then
        setup_postgres_docker
    else
        setup_postgres_native
    fi
fi

# ==============================================================================
# STEP 2: Redis Setup (Optional but recommended)
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 2: Redis Setup (Optional)${NC}"
echo "----------------------------------------"

check_redis() {
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

setup_redis_docker() {
    echo "Setting up Redis with Docker..."
    
    # Stop any existing containers
    docker stop argus-redis 2>/dev/null || true
    docker rm argus-redis 2>/dev/null || true
    
    # Start Redis container
    docker run -d \
        --name argus-redis \
        -p 6379:6379 \
        -v argus_redis_data:/data \
        redis:7-alpine
    
    echo -e "${GREEN}Redis started!${NC}"
}

if command -v docker &> /dev/null; then
    if ! check_redis; then
        setup_redis_docker
    else
        echo "Redis is already running."
    fi
else
    if ! check_redis; then
        echo "Redis not installed. Install with: brew install redis && brew services start redis"
    else
        echo "Redis is already running."
    fi
fi

# ==============================================================================
# STEP 3: Create Database Schema
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 3: Create Database Schema${NC}"
echo "----------------------------------------"

create_schema() {
    echo "Creating database tables..."
    
    # Create tables using SQL
    psql -U "$DB_USER" -h "$DB_HOST" -d "$DB_NAME" << 'EOF'
-- Analyses table
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
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);

-- Video results table
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
    heatmap_path VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Audio results table
CREATE TABLE IF NOT EXISTS audio_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    duration_seconds FLOAT NOT NULL DEFAULT 0.0,
    segments_analyzed INTEGER NOT NULL DEFAULT 0,
    overall_score FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    is_manipulated BOOLEAN NOT NULL DEFAULT FALSE,
    vocoder_detected VARCHAR(50),
    artifact_evidence JSONB,
    per_segment_scores JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Text results table
CREATE TABLE IF NOT EXISTS text_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    word_count INTEGER NOT NULL DEFAULT 0,
    language_detected VARCHAR(10) NOT NULL DEFAULT 'unknown',
    overall_score FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    is_manipulated BOOLEAN NOT NULL DEFAULT FALSE,
    perplexity_scores JSONB,
    burstiness_metrics JSONB,
    per_sentence_scores JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Trust scores table
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

-- Share links table
CREATE TABLE IF NOT EXISTS share_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    share_token VARCHAR(100) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_share_links_token ON share_links(share_token);
CREATE INDEX IF NOT EXISTS idx_share_links_analysis ON share_links(analysis_id);

COMMENT ON TABLE analyses IS 'Analysis request tracking';
COMMENT ON TABLE video_results IS 'Video analysis results';
COMMENT ON TABLE audio_results IS 'Audio analysis results';
COMMENT ON TABLE text_results IS 'Text analysis results';
COMMENT ON TABLE trust_scores IS 'Trust score and fusion results';
COMMENT ON TABLE share_links IS 'Share links for analysis results';
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Database schema created successfully!${NC}"
    else
        echo -e "${RED}Error creating database schema${NC}"
        exit 1
    fi
}

create_schema

# ==============================================================================
# STEP 4: Create Environment Configuration
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 4: Environment Configuration${NC}"
echo "----------------------------------------"

ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << EOF
# Argus Prime Environment Configuration
# ======================================

# Database Configuration
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# Redis Configuration (Optional)
REDIS_HOST=${DB_HOST}
REDIS_PORT=6379

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Storage Configuration
STORAGE_TEMP_DIR=/tmp/argus_uploads
STORAGE_MAX_FILE_SIZE=500000000

# API Keys (if needed for external services)
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}.env file created!${NC}"
else
    echo ".env file already exists."
fi

# ==============================================================================
# STEP 5: Create Missing Static Files
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 5: Create Missing Static Files${NC}"
echo "----------------------------------------"

PUBLIC_DIR="$PROJECT_DIR/frontend/public"
mkdir -p "$PUBLIC_DIR"

# Create manifest.json
cat > "$PUBLIC_DIR/manifest.json" << 'EOF'
{
  "name": "Argus Prime",
  "short_name": "Argus",
  "description": "Multi-Modal Deepfake Detection & Forensic Platform",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "icons": [
    {
      "src": "/icon.svg",
      "sizes": "any",
      "type": "image/svg+xml"
    }
  ]
}
EOF
echo "Created: manifest.json"

# Create grid.svg for background patterns
cat > "$PUBLIC_DIR/grid.svg" << 'EOF'
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="100" height="100" fill="url(#grid)"/>
</svg>
EOF
echo "Created: grid.svg"

# Create favicon.ico placeholder
cat > "$PUBLIC_DIR/favicon.ico" << 'EOF'
EOF
echo "Created: favicon.ico"

# Create icon.svg
cat > "$PUBLIC_DIR/icon.svg" << 'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none">
  <rect width="32" height="32" rx="4" fill="#000000"/>
  <path d="M8 12h16M8 16h16M8 20h10" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
  <circle cx="24" cy="20" r="3" fill="#ffffff"/>
  <circle cx="24" cy="20" r="1" fill="#000000"/>
</svg>
EOF
echo "Created: icon.svg"

# Create apple-touch-icon.png placeholder
cat > "$PUBLIC_DIR/apple-touch-icon.png" << 'EOF'
EOF
echo "Created: apple-touch-icon.png (placeholder)"

# ==============================================================================
# STEP 6: Install Python Dependencies
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 6: Install Python Dependencies${NC}"
echo "----------------------------------------"

cd "$PROJECT_DIR"
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing/updating dependencies..."
    pip install -e . --quiet
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${YELLOW}Some dependencies may have failed. Check output above.${NC}"
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    echo -e "${GREEN}Virtual environment created and dependencies installed!${NC}"
fi

# ==============================================================================
# STEP 7: Frontend Setup
# ==============================================================================
echo ""
echo -e "${YELLOW}STEP 7: Frontend Setup${NC}"
echo "----------------------------------------"

FRONTEND_DIR="$PROJECT_DIR/frontend"
if [ -d "$FRONTEND_DIR" ]; then
    cd "$FRONTEND_DIR"
    
    if [ -d "node_modules" ]; then
        echo "Frontend dependencies already installed."
    else
        echo "Installing frontend dependencies..."
        npm install
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Frontend dependencies installed!${NC}"
        else
            echo -e "${RED}Error installing frontend dependencies${NC}"
        fi
    fi
else
    echo "Frontend directory not found."
fi

# ==============================================================================
# COMPLETION
# ==============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}     SETUP COMPLETE!${NC}"
echo "=============================================="
echo ""
echo "Next steps to run the application:"
echo ""
echo "1. Start the backend server:"
echo "   cd $PROJECT_DIR"
echo "   source venv/bin/activate"
echo "   python src/main.py"
echo ""
echo "2. Start the frontend (in a new terminal):"
echo "   cd $PROJECT_DIR/frontend"
echo "   npm run dev"
echo ""
echo "3. Or use Docker Compose (recommended):"
echo "   docker-compose up -d"
echo ""
echo "4. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Note: Make sure PostgreSQL is running before starting the backend.${NC}"
echo ""
