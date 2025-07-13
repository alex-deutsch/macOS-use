#!/usr/bin/env python3
"""
Enhanced REST API Server for macOS-use with aggressive cache management
Provides HTTP endpoints to control macOS automation with better reliability
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
    # Import the UI tree builder to reset it completely
    from mlx_use.mac.tree import MacUITreeBuilder
    from mlx_use.controller.service import Controller
except ImportError as e:
    logger.error(f"Failed to import mlx_use modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="macOS-use API v2",
    description="REST API for macOS automation with aggressive cache management",
    version="2.0.0",
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
active_agents: Dict[str, Agent] = {}
task_results: Dict[str, Any] = {}
task_counter = 0
global_controller: Optional[Controller] = None

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

def reset_accessibility_state():
    """Aggressively reset all accessibility state"""
    try:
        # Force a complete reset of accessibility state
        import objc
        from ApplicationServices import AXUIElementGetPid
        
        # Sleep to allow any pending accessibility operations to complete
        time.sleep(0.5)
        
        # Force garbage collection
        gc.collect()
        
        # Additional sleep to ensure cleanup
        time.sleep(0.2)
        
        logger.info("Accessibility state reset completed")
        return True
    except Exception as e:
        logger.warning(f"Error resetting accessibility state: {e}")
        return False

def cleanup_agent_aggressively(task_id: str):
    """Aggressively clean up agent resources"""
    if task_id in active_agents:
        try:
            agent = active_agents[task_id]
            
            # Reset the tree builder completely
            if hasattr(agent, 'mac_tree_builder'):
                tree_builder = agent.mac_tree_builder
                
                # Clear all caches
                tree_builder._element_cache.clear()
                tree_builder._processed_elements.clear()
                tree_builder._observers.clear()
                
                # Reset all counters and state
                tree_builder.highlight_index = 0
                tree_builder._current_app_pid = None
                
                # Create a completely new tree builder instance
                agent.mac_tree_builder = MacUITreeBuilder()
            
            # Reset controller state if available
            if hasattr(agent, 'controller'):
                controller = agent.controller
                # Reset any cached state in controller
                if hasattr(controller, '_last_screenshot'):
                    controller._last_screenshot = None
                if hasattr(controller, '_last_tree'):
                    controller._last_tree = None
            
            # Clear message manager state
            if hasattr(agent, 'message_manager'):
                message_manager = agent.message_manager
                # Reset conversation history to prevent element reference pollution
                if hasattr(message_manager, '_messages'):
                    # Keep system messages but clear state messages
                    system_messages = [msg for msg in message_manager._messages 
                                     if hasattr(msg, 'type') and msg.type == 'system']
                    message_manager._messages = system_messages
            
            del active_agents[task_id]
            logger.info(f"Aggressively cleaned up agent for task {task_id}")
        except Exception as e:
            logger.warning(f"Error in aggressive cleanup for agent {task_id}: {e}")
    
    # Reset global accessibility state
    reset_accessibility_state()

async def run_agent_task(task_request: TaskRequest, task_id: str):
    """Run agent task with aggressive state management"""
    global task_results, task_counter, active_agents, global_controller
    
    try:
        # Always start with a clean slate
        reset_accessibility_state()
        
        # Get API key
        api_key = get_api_key(task_request.llm_provider, task_request.api_key)
        
        # Create LLM instance
        llm = create_llm(task_request.llm_provider, task_request.llm_model, api_key)
        
        # Create a completely fresh controller for each task
        controller = Controller()
        
        # Create a completely fresh tree builder
        tree_builder = MacUITreeBuilder()
        
        # Initialize agent with fresh components
        agent = Agent(
            task=task_request.task,
            llm=llm,
            controller=controller,
            max_actions_per_step=task_request.max_actions
        )
        
        # Override with our fresh tree builder
        agent.mac_tree_builder = tree_builder
        
        # Store active agent
        active_agents[task_id] = agent
        
        # Additional reset before running
        reset_accessibility_state()
        
        # Run task
        logger.info(f"Starting task {task_id}: {task_request.task}")
        result = await agent.run(max_steps=task_request.max_steps)
        
        # Store result
        task_results[task_id] = {
            "status": "completed",
            "result": str(result.final_result()) if result.final_result() else "Task completed",
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
        # Always perform aggressive cleanup after task completion
        cleanup_agent_aggressively(task_id)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "macOS-use API Server v2", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "macOS-use API v2",
        "active_tasks": len([r for r in task_results.values() if r["status"] == "running"]),
        "total_tasks": len(task_results)
    }

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new automation task"""
    
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
    
    # Start background task using asyncio
    asyncio.create_task(run_agent_task(task_request, task_id))
    
    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Task '{task_request.task}' started successfully"
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
    
    # Clean up agent if still active
    cleanup_agent_aggressively(task_id)
    
    del task_results[task_id]
    return {"message": f"Task {task_id} deleted"}

@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    """Stop a running task"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update status to stopped
    if task_results[task_id]["status"] == "running":
        task_results[task_id]["status"] = "stopped"
        task_results[task_id]["error"] = "Task stopped by user"
        cleanup_agent_aggressively(task_id)
    
    return {"message": f"Task {task_id} stopped"}

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches and reset accessibility state"""
    # Cleanup all active agents aggressively
    for task_id in list(active_agents.keys()):
        cleanup_agent_aggressively(task_id)
    
    # Clear completed/failed tasks
    completed_tasks = [tid for tid, result in task_results.items() 
                     if result["status"] in ["completed", "failed"]]
    
    for tid in completed_tasks:
        if tid in task_results:
            del task_results[tid]
    
    # Reset global accessibility state
    reset_accessibility_state()
    
    # Force multiple garbage collections
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)
    
    return {
        "message": "Cache and accessibility state cleared successfully", 
        "remaining_tasks": len(task_results)
    }

@app.post("/accessibility/reset")
async def reset_accessibility():
    """Reset accessibility state without clearing task history"""
    success = reset_accessibility_state()
    
    # Force garbage collection
    gc.collect()
    
    return {
        "message": "Accessibility state reset",
        "success": success
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
    
    config = {
        "available_providers": ["openai", "anthropic", "gemini", "deepseek", "ollama"],
        "models": {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
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
        "ollama": {
            "available": ollama_available,
            "base_url": "http://localhost:11434",
            "models": ollama_models
        }
    }
    
    return config

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
    """Execute a simple task quickly with default settings"""
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
        "check_status_url": f"/tasks/{response.task_id}"
    }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run server
    uvicorn.run(
        "api_server_v2:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
