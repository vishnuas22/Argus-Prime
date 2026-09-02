"""
Backend entry point for supervisor.
Imports the FastAPI app from src/main.py
"""
import sys
import os

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable for app port
os.environ.setdefault('APP_PORT', '8001')

from src.main import app

__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
