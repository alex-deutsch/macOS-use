#!/usr/bin/env python3
"""
Simple test script for Docker environment
This script tests the core functionality without macOS-specific features
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if core dependencies are available"""
    try:
        import langchain
        print(f"✓ LangChain imported successfully (version: {langchain.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import LangChain: {e}")
        return False
    
    try:
        import gradio
        print(f"✓ Gradio imported successfully (version: {gradio.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import Gradio: {e}")
        return False
    
    try:
        import requests
        print(f"✓ Requests imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Requests: {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("\n=== Environment Test ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check working directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found")
    else:
        print("✗ .env file not found")
    
    # Check API keys (without revealing them)
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.getenv(key)
        if value:
            print(f"✓ {key} is set")
        else:
            print(f"○ {key} is not set")

def main():
    print("=== macOS-use Docker Test ===")
    print("Testing core functionality in Docker environment...\n")
    
    print("=== Import Test ===")
    if not test_imports():
        print("Import test failed!")
        sys.exit(1)
    
    test_environment()
    
    print("\n=== Test Complete ===")
    print("Core dependencies are working!")
    print("You can now:")
    print("1. Set your API keys in the .env file")
    print("2. Run Gradio app: python gradio_app/app.py")
    print("3. Or explore the examples (note: macOS-specific features won't work in Docker)")

if __name__ == "__main__":
    main()
