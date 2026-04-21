"""
run_viewer.py
=============
Starts the Defense ML Viewer web app.

Usage:
    venv/bin/python run_viewer.py
    # Then open http://localhost:5051 in your browser
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051, debug=False)
