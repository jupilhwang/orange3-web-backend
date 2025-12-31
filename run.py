#!/usr/bin/env python3
"""
Orange3-Web Backend Runner.

Reads configuration from orange3-web.properties or environment variables
and starts the uvicorn server with the configured settings.

Usage:
    python run.py                    # Use config file or env vars
    python run.py --port 8080        # Override port
    python run.py --workers 4        # Override workers
    python run.py --reload           # Enable auto-reload (dev mode)
"""

import argparse
import sys
import uvicorn

from app.core.config import get_config


def main():
    """Main entry point."""
    config = get_config()
    
    # Parse command line arguments (override config)
    parser = argparse.ArgumentParser(description="Orange3-Web Backend Server")
    parser.add_argument(
        "--host", 
        type=str, 
        default=None,
        help=f"Host to bind (default: {config.server.host})"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help=f"Port to bind (default: {config.server.port})"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help=f"Number of worker processes (default: {config.server.workers})"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        default=None,
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {config.log.level.lower()})"
    )
    
    args = parser.parse_args()
    
    # Merge: CLI args > config file > defaults
    host = args.host or config.server.host
    port = args.port or config.server.port
    workers = args.workers or config.server.workers
    reload = args.reload if args.reload is not None else config.server.reload
    log_level = args.log_level or config.log.level.lower()
    
    # Print configuration
    print("=" * 60)
    print("Orange3-Web Backend Server")
    print("=" * 60)
    print(f"  Host:     {host}")
    print(f"  Port:     {port}")
    print(f"  Workers:  {workers}")
    print(f"  Reload:   {reload}")
    print(f"  LogLevel: {log_level}")
    print("=" * 60)
    
    # Uvicorn config
    uvicorn_config = {
        "app": "app.main:app",
        "host": host,
        "port": port,
        "log_level": log_level,
    }
    
    # Workers and reload are mutually exclusive
    if reload:
        uvicorn_config["reload"] = True
        if workers > 1:
            print("Warning: --reload is incompatible with multiple workers. Using 1 worker.")
    else:
        uvicorn_config["workers"] = workers
    
    # Run server
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()

