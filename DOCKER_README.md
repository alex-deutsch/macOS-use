# macOS-use Docker Setup

This guide will help you run macOS-use in a Docker container. Note that since this is a macOS-specific application, running it in Docker (Linux) will have limitations - primarily the Gradio web interface and core LLM functionality will work, but macOS-specific system interactions won't function.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t macos-use:latest .
```

### 2. Set Up Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your favorite editor and add your API keys
```

### 3. Run the Container

#### Interactive Mode (Recommended for development)
```bash
docker run -it --rm \
    --name macos-use-container \
    -p 7860:7860 \
    --env-file .env \
    -v "$(pwd)":/app \
    macos-use:latest
```

#### Run Gradio Web Interface
```bash
docker run -it --rm \
    --name macos-use-gradio \
    -p 7860:7860 \
    --env-file .env \
    -v "$(pwd)":/app \
    macos-use:latest \
    python gradio_app/app.py
```

### 4. Test the Installation

Run the test script to verify everything is working:

```bash
docker run --rm --env-file .env macos-use:latest python docker_test.py
```

## Alternative: Using Docker Compose

You can also use Docker Compose for easier management:

```bash
# Build and start the services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the services
docker-compose down
```

## Convenient Script

Use the provided script for easy setup:

```bash
./run-docker.sh
```

## What Works in Docker

✅ **Core LLM functionality**
- LangChain integration
- OpenAI, Anthropic, Google Gemini APIs
- Basic chat and reasoning

✅ **Web Interface**
- Gradio web app
- API interactions
- Text processing

❌ **macOS-Specific Features** (These won't work in Linux Docker)
- Desktop automation
- App interaction
- Screenshot capture
- macOS system calls

## API Keys

You need at least one of these API keys:

- `OPENAI_API_KEY` - OpenAI GPT models
- `ANTHROPIC_API_KEY` - Claude models  
- `GEMINI_API_KEY` - Google Gemini models

Add them to your `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## Troubleshooting

### Container Fails to Start
- Make sure Docker is running
- Check that ports 7860 is not already in use
- Verify your `.env` file exists and has proper format

### API Errors
- Verify your API keys are correctly set in `.env`
- Check that you have credits/quota remaining for your chosen API

### Permission Issues
- The container runs as a non-root user for security
- If you need to modify files, you can run as root: `docker run --user root ...`

## Development

To develop inside the container:

```bash
# Start container with bash
docker run -it --rm \
    --name macos-use-dev \
    -p 7860:7860 \
    --env-file .env \
    -v "$(pwd)":/app \
    macos-use:latest \
    /bin/bash

# Inside container, you can now:
python docker_test.py              # Test setup
python gradio_app/app.py          # Start web interface
# Edit files on your host, changes reflect in container
```

## Limitations

Since this is running in a Linux container:

1. **No macOS Desktop Integration** - The core value proposition of macOS-use (desktop automation) won't work
2. **Limited Examples** - Most examples require macOS-specific features
3. **Testing Platform** - This Docker setup is mainly useful for:
   - Testing LLM integrations
   - Web interface development  
   - API functionality verification
   - Learning the codebase structure

For full functionality, you'll need to run macOS-use natively on macOS.

## Next Steps

1. **For Web Development**: Use the Gradio interface to build web-based AI applications
2. **For macOS Use**: Install natively on macOS following the main README
3. **For Learning**: Explore the codebase and understand the architecture

## Files Created for Docker

- `Dockerfile` - Main container definition
- `Dockerfile.gradio` - Gradio-specific container
- `docker-compose.yml` - Multi-service setup
- `run-docker.sh` - Convenience script
- `docker_test.py` - Installation verification
- `DOCKER_README.md` - This guide
