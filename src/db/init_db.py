#!/usr/bin/env python3
"""
Argus Prime - Database Initialization Script

Creates all required tables in the PostgreSQL database.
Run this script after starting the database containers.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from src.config import get_config
from src.core.logging import logger
from src.db.database import get_database
from src.db.models import Base


async def create_tables():
    """Create all database tables."""
    logger.info("Starting database initialization...")
    
    config = get_config()
    db_url = config.database.url
    logger.info(f"Connecting to database: {db_url.split('@')[0]}@...")
    
    db = get_database()
    
    try:
        # Create all tables
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("All tables created successfully!")
        
        # Verify tables exist
        async with db.get_session() as session:
            result = await session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"Tables in database: {tables}")
            
            expected_tables = [
                'analyses',
                'video_results', 
                'audio_results',
                'text_results',
                'trust_scores',
                'share_links'
            ]
            
            missing = [t for t in expected_tables if t not in tables]
            if missing:
                logger.warning(f"Missing tables: {missing}")
            else:
                logger.info("All expected tables exist!")
                
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def drop_tables():
    """Drop all database tables (use with caution!)."""
    logger.warning("Dropping all tables...")
    
    db = get_database()
    
    try:
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("All tables dropped successfully!")
        
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Argus Prime Database Manager")
    parser.add_argument(
        "--drop", 
        action="store_true",
        help="Drop all tables before creating (WARNING: data loss!)"
    )
    
    args = parser.parse_args()
    
    if args.drop:
        asyncio.run(drop_tables())
    else:
        asyncio.run(create_tables())


if __name__ == "__main__":
    main()
