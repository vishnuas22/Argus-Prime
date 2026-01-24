"""
Backend entry point for supervisor.
Imports the FastAPI app from src/main.py
"""
import sys
import os

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app

__all__ = ['app']
