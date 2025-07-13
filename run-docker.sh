#!/bin/bash

# Script to build and run macOS-use in Docker

set -e

echo "Building macOS-use Docker image..."
docker build -t macos-use:latest .

echo "Creating .env file if it doesn't exist..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your API keys before running."
fi

echo "Starting macOS-use container..."
docker run -it --rm \
    --name macos-use-container \
    -p 7860:7860 \
    -e DISPLAY=:99 \
    --env-file .env \
    -v "$(pwd)":/app \
    macos-use:latest

echo "Container stopped."
