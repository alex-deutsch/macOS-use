FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install basic dependencies first
COPY pyproject.toml .
COPY .env.example .env

# Install only core dependencies without macOS-specific packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        beautifulsoup4>=4.12.3 \
        httpx>=0.27.2 \
        langchain==0.3.14 \
        langchain-openai==0.3.1 \
        langchain-anthropic==0.3.3 \
        langchain-google-genai==2.0.8 \
        pydantic>=2.10.4 \
        python-dotenv>=1.0.1 \
        requests>=2.32.3 \
        gradio>=5.16.1

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for Gradio app
EXPOSE 7860

# Default command - start with bash for interactive use
CMD ["/bin/bash"]
