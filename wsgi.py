"""
WSGI entry point for the Ops Planner web application.

This module simply imports the Flask application instance from ``app.py``.
It does not modify any of the application logic; its sole purpose is to
provide a top‑level ``app`` object that a WSGI server such as gunicorn can
reference. Keeping the import here ensures that your ``app.py`` remains
unchanged and your project structure stays intact.

Usage (for example with gunicorn)::

    gunicorn wsgi:app

"""

from app import app  # type: ignore  # pragma: no cover

__all__ = ["app"]