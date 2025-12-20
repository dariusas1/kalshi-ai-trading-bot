#!/usr/bin/env python3
"""
Main entry point for Railway deployment.
This file serves as a fallback entry point for Railpack auto-detection.
Redirects to the Streamlit dashboard.
"""

import subprocess
import sys
import os

def main():
    """Main entry point that starts the Streamlit dashboard."""
    # Get port from environment variable (Railway sets this)
    port = os.environ.get('PORT', '8501')

    print(f"🚀 Starting Kalshi Trading Dashboard on port {port}...")

    # Start Streamlit dashboard
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'trading_dashboard.py',
            '--server.port', port,
            '--server.address', '0.0.0.0',
            '--browser.gatherUsageStats', 'false'
        ])
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()