#!/usr/bin/env python3
"""
Ollama-optimized REST API Server for macOS-use
Reduced overhead and optimized for local model performance
"""

import os
import asyncio
import uvicorn
import gc
import time
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
    from mlx_use.mac.tree import MacUITreeBuilder
    from mlx_use.controller.service import Controller
except ImportError as e:
    logger.error(f"Failed to import mlx_use modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="macOS-use API - Ollama Optimized",
    description="REST API optimized for Ollama local models",
    version="3.1.0",
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

# Global variables
task_results: Dict[str, Any] = {}

# Pydantic models
class TaskRequest(BaseModel):
    task: str
    max_steps: int = 15  # Reduced for cost control
    max_actions: int = 5   # Reduced for cost control
    llm_provider: str = "openai"
    llm_model: str = "o1-mini"  # Default to cost-effective model
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
    """Create LLM instance based on provider with Ollama optimizations"""
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
            response = requests.get("http://localhost:11434/api/version", timeout=3)
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama server not available: {e}")
        
        # Ollama optimizations
        return ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.0,  # Reduced for consistency
            num_ctx=4096,     # Reduced context for speed
            num_predict=512,  # Limit output length
            repeat_penalty=1.1,
            timeout=60,       # Shorter timeout
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def quick_reset():
    """Quick and lightweight reset instead of ultra-aggressive"""
    try:
        # Just basic garbage collection
        gc.collect()
        time.sleep(0.1)
        logger.info("Quick reset completed")
        return True
    except Exception as e:
        logger.warning(f"Error in quick reset: {e}")
        return False

async def run_optimized_task(task_request: TaskRequest, task_id: str):
    """Run task with Ollama optimizations"""
    global task_results
    
    try:
        logger.info(f"Starting optimized task {task_id}: {task_request.task}")
        
        # Minimal pre-cleanup for Ollama
        if task_request.llm_provider.lower() == "ollama":
            quick_reset()
        else:
            # More thorough reset for cloud models
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
        
        # Create LLM
        api_key = get_api_key(task_request.llm_provider, task_request.api_key)
        llm = create_llm(task_request.llm_provider, task_request.llm_model, api_key)
        
        # Create fresh components
        controller = Controller()
        
        # Create agent with optimized settings
        agent = Agent(
            task=task_request.task,
            llm=llm,
            controller=controller,
            max_actions_per_step=task_request.max_actions,
            use_vision=True
        )
        
        # Ensure fresh tree builder
        agent.mac_tree_builder = MacUITreeBuilder()
        
        # Run task with reduced steps for Ollama
        max_steps = task_request.max_steps
        if task_request.llm_provider.lower() == "ollama":
            max_steps = min(max_steps, 20)  # Limit steps for Ollama
        
        logger.info(f"Running task with max_steps={max_steps}")
        result = await agent.run(max_steps=max_steps)
        
        # Store result
        final_result = str(result.final_result()) if result and result.final_result() else "Task completed"
        task_results[task_id] = {
            "status": "completed",
            "result": final_result,
            "error": None
        }
        
        logger.info(f"Task {task_id} completed successfully: {final_result}")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_results[task_id] = {
            "status": "failed", 
            "result": None,
            "error": str(e)
        }
    finally:
        # Lightweight cleanup
        quick_reset()

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "macOS-use API - Ollama Optimized", "version": "3.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "macOS-use API - Ollama Optimized",
        "active_tasks": len([r for r in task_results.values() if r["status"] == "running"]),
        "total_tasks": len(task_results),
        "optimization": "ollama-friendly"
    }

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new automation task with Ollama optimizations"""
    
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
    
    # Start optimized task
    asyncio.create_task(run_optimized_task(task_request, task_id))
    
    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Task '{task_request.task}' started with {task_request.llm_provider} optimization"
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
    """Quick cache clearing"""
    completed_tasks = [tid for tid, result in task_results.items() 
                     if result["status"] in ["completed", "failed"]]
    
    for tid in completed_tasks:
        if tid in task_results:
            del task_results[tid]
    
    quick_reset()
    
    return {
        "message": "Quick cache clear completed", 
        "remaining_tasks": len(task_results)
    }

@app.post("/accessibility/reset")
async def reset_accessibility():
    """Quick accessibility reset"""
    quick_reset()
    
    return {
        "message": "Quick accessibility reset completed",
        "success": True
    }

@app.get("/config")
async def get_config():
    """Get current configuration with Ollama status"""
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
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-sonnet", "claude-3-haiku"],
            "gemini": ["gemini-pro", "gemini-pro-vision"],
            "deepseek": ["deepseek-chat"],
            "ollama": ollama_models if ollama_available else []
        },
        "ollama_optimizations": {
            "reduced_context": 4096,
            "reduced_steps": 20,
            "faster_timeout": 60,
            "temperature": 0.0
        },
        "default_settings": {
            "max_steps": 30,  # Reduced from 50
            "max_actions": 10,  # Reduced from 50
            "share_prompt": False,
            "share_terminal": True
        },
        "optimization": "ollama-friendly",
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
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        if response.status_code == 200:
            version_info = response.json()
            
            models_response = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = []
            if models_response.status_code == 200:
                data = models_response.json()
                models = [model["name"] for model in data.get("models", [])]
            
            return {
                "status": "running",
                "version": version_info.get("version", "unknown"),
                "models": models,
                "model_count": len(models),
                "optimizations": "enabled"
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
                "names": [model["name"] for model in models],
                "recommended": ["qwen3:8b", "qwen3:14b", "qwen2.5-coder:7b"]
            }
        else:
            raise HTTPException(status_code=503, detail="Ollama server not responding")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {e}")

@app.post("/quick-task")
async def quick_task(task: str, provider: str = "openai", model: str = "gpt-4"):
    """Execute a simple task quickly with provider-specific optimizations"""
    # Optimize settings for Ollama
    max_steps = 30
    max_actions = 10
    
    if provider.lower() == "ollama":
        max_steps = 15  # Even more aggressive for Ollama
        max_actions = 5
        
    task_request = TaskRequest(
        task=task,
        llm_provider=provider,
        llm_model=model,
        max_steps=max_steps,
        max_actions=max_actions
    )
    
    background_tasks = BackgroundTasks()
    response = await create_task(task_request, background_tasks)
    
    await asyncio.sleep(0.5)  # Reduced wait time
    
    return {
        "task_id": response.task_id,
        "task": task,
        "status": "started",
        "check_status_url": f"/tasks/{response.task_id}",
        "optimization": f"{provider}-optimized",
        "max_steps": max_steps
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    uvicorn.run(
        "api_server_ollama:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
