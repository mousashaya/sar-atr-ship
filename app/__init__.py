"""
app/__init__.py
===============
Creates and configures the Flask application for the Defense ML Viewer.
"""

from flask import Flask


def create_app():
    app = Flask(__name__)
    from .routes import bp
    app.register_blueprint(bp)
    return app
