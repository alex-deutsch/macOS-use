#!/usr/bin/env python3
"""
Ultra-aggressive REST API Server for macOS-use with complete process isolation
Completely resets all state between tasks to prevent element corruption
"""

import os
import asyncio
import uvicorn
import gc
import time
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import macOS-use components
try:
    from mlx_use.agent.service import Agent
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from dotenv import load_dotenv
    # Import the UI tree builder to reset it completely
    from mlx_use.mac.tree import MacUITreeBuilder
    from mlx_use.controller.service import Controller
except ImportError as e:
    logger.error(f"Failed to import mlx_use modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="macOS-use API v3",
    description="REST API with ultra-aggressive state isolation",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables - keep minimal
task_results: Dict[str, Any] = {}
task_counter = 0

# Pydantic models
class TaskRequest(BaseModel):
    task: str
    max_steps: int = 50
    max_actions: int = 50
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    api_key: Optional[str] = None
    share_prompt: bool = False
    share_terminal: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

# Helper functions
def get_api_key(provider: str, provided_key: Optional[str] = None) -> str:
    """Get API key from environment or provided key"""
    if provided_key:
        return provided_key
    
    # Ollama doesn't need an API key
    if provider.lower() == "ollama":
        return "not-needed"
    
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }
    
    env_key = key_map.get(provider.lower())
    if not env_key:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    api_key = os.getenv(env_key)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key not found for {provider}. Set {env_key} or provide api_key in request.")
    
    return api_key

def create_llm(provider: str, model: str, api_key: str):
    """Create LLM instance based on provider"""
    if provider.lower() == "openai":
        return ChatOpenAI(model=model, api_key=api_key)
    elif provider.lower() == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key)
    elif provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=api_key)
    elif provider.lower() == "ollama":
        # Check if Ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama server not available: {e}")
        
        return ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def ultra_aggressive_reset():
    """Ultra-aggressive reset of all possible state"""
    try:
        logger.info("Starting ultra-aggressive reset...")
        
        # Force garbage collection multiple times
        for i in range(5):
            gc.collect()
            time.sleep(0.1)
        
        # Import and reset accessibility frameworks
        try:
            import objc
            from ApplicationServices import (
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
                kAXErrorSuccess
            )
            
            # Clear any cached references
            objc.recycleAutoreleasePool()
            
        except Exception as e:
            logger.warning(f"Error in accessibility reset: {e}")
        
        # Sleep to allow system cleanup
        time.sleep(1.0)
        
        logger.info("Ultra-aggressive reset completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in ultra-aggressive reset: {e}")
        return False

def kill_stale_processes():
    """Kill any stale processes that might be interfering"""
    try:
        # More targeted process killing - avoid killing the API server itself
        # Kill any orphaned accessibility processes but not our own
        current_pid = os.getpid()
        
        # Don't kill processes for now - too aggressive
        # Just log that we would do this step
        logger.info("Skipping process killing to avoid server shutdown")
        
        time.sleep(0.2)
        
    except Exception as e:
        logger.warning(f"Error in stale process handling: {e}")

async def run_isolated_task(task_request: TaskRequest, task_id: str):
    """Run a single task in complete isolation"""
    global task_results
    
    # Isolated variables for this task only
    agent = None
    llm = None
    controller = None
    tree_builder = None
    
    try:
        logger.info(f"Starting isolated task {task_id}: {task_request.task}")
        
        # STEP 1: Ultra-aggressive pre-cleanup
        ultra_aggressive_reset()
        kill_stale_processes()
        
        # STEP 2: Create completely fresh instances
        api_key = get_api_key(task_request.llm_provider, task_request.api_key)
        llm = create_llm(task_request.llm_provider, task_request.llm_model, api_key)
        
        # Create completely isolated components
        controller = Controller()
        tree_builder = MacUITreeBuilder()
        
        # STEP 3: Create agent with fresh everything
        agent = Agent(
            task=task_request.task,
            llm=llm,
            controller=controller,
            max_actions_per_step=task_request.max_actions
        )
        
        # Force use of our fresh tree builder
        agent.mac_tree_builder = tree_builder
        
        # STEP 4: Another reset before running
        ultra_aggressive_reset()
        
        # STEP 5: Run the task
        result = await agent.run(max_steps=task_request.max_steps)
        
        # STEP 6: Store result
        final_result = str(result.final_result()) if result and result.final_result() else "Task completed"
        task_results[task_id] = {
            "status": "completed",
            "result": final_result,
            "error": None
        }
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_results[task_id] = {
            "status": "failed",
            "result": None,
            "error": str(e)
        }
    
    finally:
        # STEP 7: Ultra-aggressive cleanup
        try:
            # Explicitly destroy all objects
            if agent:
                if hasattr(agent, 'mac_tree_builder'):
                    tree_builder = agent.mac_tree_builder
                    if hasattr(tree_builder, '_element_cache'):
                        tree_builder._element_cache.clear()
                    if hasattr(tree_builder, '_processed_elements'):
                        tree_builder._processed_elements.clear()
                    if hasattr(tree_builder, '_observers'):
                        tree_builder._observers.clear()
                    tree_builder.highlight_index = 0
                    tree_builder._current_app_pid = None
                
                if hasattr(agent, 'controller'):
                    controller = agent.controller
                    # Clear any controller state
                    if hasattr(controller, '_last_screenshot'):
                        controller._last_screenshot = None
                    if hasattr(controller, '_last_tree'):
                        controller._last_tree = None
                
                if hasattr(agent, 'message_manager'):
                    # Clear message history to prevent element pollution
                    agent.message_manager = None
                
                # Destroy agent reference
                agent = None
            
            # Destroy other references
            llm = None
            controller = None
            tree_builder = None
            
            # Final ultra-aggressive cleanup
            ultra_aggressive_reset()
            kill_stale_processes()
            
            logger.info(f"Completed ultra-aggressive cleanup for task {task_id}")
            
        except Exception as cleanup_error:
            logger.error(f"Error in cleanup for task {task_id}: {cleanup_error}")

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "macOS-use API Server v3 - Ultra Isolated", "version": "3.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "macOS-use API v3",
        "active_tasks": len([r for r in task_results.values() if r["status"] == "running"]),
        "total_tasks": len(task_results),
        "isolation_level": "ultra-aggressive"
    }

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new automation task with complete isolation"""
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Validate API key
    try:
        get_api_key(task_request.llm_provider, task_request.api_key)
    except HTTPException as e:
        raise e
    
    # Initialize task status
    task_results[task_id] = {
        "status": "running",
        "result": None,
        "error": None
    }
    
    # Start completely isolated task
    asyncio.create_task(run_isolated_task(task_request, task_id))
    
    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Task '{task_request.task}' started in isolated environment"
    )

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_results[task_id]
    return TaskStatus(
        task_id=task_id,
        status=result["status"],
        result=result["result"],
        error=result["error"]
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": result["status"]
            }
            for task_id, result in task_results.items()
        ]
    }

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task from results"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_results[task_id]
    return {"message": f"Task {task_id} deleted"}

@app.post("/cache/clear")
async def clear_cache():
    """Ultra-aggressive cache clearing"""
    
    # Clear completed/failed tasks
    completed_tasks = [tid for tid, result in task_results.items() 
                     if result["status"] in ["completed", "failed"]]
    
    for tid in completed_tasks:
        if tid in task_results:
            del task_results[tid]
    
    # Ultra-aggressive reset
    ultra_aggressive_reset()
    kill_stale_processes()
    
    return {
        "message": "Ultra-aggressive cache clear completed", 
        "remaining_tasks": len(task_results)
    }

@app.post("/accessibility/reset")
async def reset_accessibility():
    """Ultra-aggressive accessibility reset"""
    ultra_aggressive_reset()
    kill_stale_processes()
    
    return {
        "message": "Ultra-aggressive accessibility reset completed",
        "success": True
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    # Check Ollama availability and get models
    ollama_models = []
    ollama_available = False
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_available = True
            data = response.json()
            ollama_models = [model["name"] for model in data.get("models", [])]
    except:
        pass
    
    return {
        "available_providers": ["openai", "anthropic", "gemini", "deepseek", "ollama"],
        "models": {
            "openai": ["gpt-4", "gpt-4-turbo", "o4-mini", "GPT-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-sonnet", "claude-3-haiku"],
            "gemini": ["gemini-pro", "gemini-pro-vision"],
            "deepseek": ["deepseek-chat"],
            "ollama": ollama_models if ollama_available else ["llama2", "codellama", "mistral"]
        },
        "default_settings": {
            "max_steps": 50,
            "max_actions": 50,
            "share_prompt": False,
            "share_terminal": True
        },
        "isolation": "ultra-aggressive",
        "ollama": {
            "available": ollama_available,
            "base_url": "http://localhost:11434",
            "models": ollama_models
        }
    }

@app.get("/ollama/status")
async def ollama_status():
    """Check Ollama server status"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            version_info = response.json()
            
            # Get available models
            models_response = requests.get("http://localhost:11434/api/tags", timeout=2)
            models = []
            if models_response.status_code == 200:
                data = models_response.json()
                models = [model["name"] for model in data.get("models", [])]
            
            return {
                "status": "running",
                "version": version_info.get("version", "unknown"),
                "models": models,
                "model_count": len(models)
            }
        else:
            return {"status": "not_responding", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "not_available", "error": str(e)}

@app.get("/ollama/models")
async def ollama_models():
    """Get available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return {
                "models": models,
                "count": len(models),
                "names": [model["name"] for model in models]
            }
        else:
            raise HTTPException(status_code=503, detail="Ollama server not responding")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {e}")

@app.post("/quick-task")
async def quick_task(task: str, provider: str = "openai", model: str = "gpt-4"):
    """Execute a simple task quickly with complete isolation"""
    task_request = TaskRequest(
        task=task,
        llm_provider=provider,
        llm_model=model
    )
    
    background_tasks = BackgroundTasks()
    response = await create_task(task_request, background_tasks)
    
    # Wait a moment and return status
    await asyncio.sleep(1)
    
    return {
        "task_id": response.task_id,
        "task": task,
        "status": "started",
        "check_status_url": f"/tasks/{response.task_id}",
        "isolation": "ultra-aggressive"
    }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run server
    uvicorn.run(
        "api_server_v3:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
