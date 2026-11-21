"""ASGI entry point:  uvicorn src.api.asgi:app --port 8000"""
from src.api.main import create_app

app = create_app()
