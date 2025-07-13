#!/bin/bash

# macOS-use API Server Startup Script

# Set the working directory
cd "/Users/alexanderdeutsch/workspace/web-ui/macOS-use"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate the virtual environment
source .venv/bin/activate

# Start the Ollama-optimized API server
python api_server_ollama.py > /tmp/macos-use-api.log 2>&1
