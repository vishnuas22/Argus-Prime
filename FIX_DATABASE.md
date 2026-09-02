# Database Tables Fix

The error `relation "analyses" does not exist` means the database tables haven't been created.

## Quick Fix - Rebuild Docker Containers

```bash
# Stop and remove existing containers
docker-compose down -v

# Rebuild and start
docker-compose up -d --build

# Watch logs
docker-compose logs -f app
```

## Alternative - Run Init Script Manually

If you want to keep your existing data:

```bash
# Rebuild Python package (includes new init script)
pip install -e .

# Run database initialization
argus-init-db

# Or with Python directly
python -m src.db.init_db
```

## Verify Tables Created

```bash
# Check tables exist
docker-compose exec postgres psql -U argus -d argus_prime -c "\dt"

# Expected output:
#                   List of relations
#  Schema |     Name      | Type  |  Owner   
# --------+--------------+-------+----------
#  public | analyses     | table | argus
#  public | audio_results| table | argus
#  public | share_links  | table | argus
#  public | text_results | table | argus
#  public | trust_scores | table | argus
#  public | video_results| table | argus
# (6 rows)
```

## Test the Application

After tables are created:

```bash
# Upload a test file
curl -X POST -F "file=@/tmp/test.mp4" http://localhost:8000/api/v1/analyze

# Expected response:
# {"request_id":"...","status":"pending"}
```
