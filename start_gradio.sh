#!/bin/bash

# macOS-use Gradio Web UI Startup Script

# Set the working directory
cd "/Users/alexanderdeutsch/workspace/web-ui/macOS-use"

# Activate the virtual environment
source .venv/bin/activate

# Start the Gradio app
python gradio_app/app.py > /tmp/macos-use-gradio.log 2>&1
