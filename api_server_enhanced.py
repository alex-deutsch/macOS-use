#!/usr/bin/env python3
"""
Enhanced REST API Server for macOS-use with cache management
Provides HTTP endpoints to control macOS automation with better reliability
"""

import os
import asyncio
import uvicorn
import gc
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
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import mlx_use modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="macOS-use API Enhanced",
    description="REST API for macOS automation with cache management",
    version="1.1.0",
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
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def cleanup_agent(task_id: str):
    """Clean up agent resources"""
    if task_id in active_agents:
        try:
            agent = active_agents[task_id]
            # Clear caches
            if hasattr(agent, 'mac_tree_builder'):
                agent.mac_tree_builder._element_cache.clear()
                agent.mac_tree_builder._processed_elements.clear()
                agent.mac_tree_builder.highlight_index = 0
                agent.mac_tree_builder._current_app_pid = None
            del active_agents[task_id]
            logger.info(f"Cleaned up agent for task {task_id}")
        except Exception as e:
            logger.warning(f"Error cleaning up agent {task_id}: {e}")
    
    # Force garbage collection
    gc.collect()

async def run_agent_task(task_request: TaskRequest, task_id: str):
    """Run agent task in background with enhanced cleanup"""
    global task_results, task_counter, active_agents
    
    try:
        # Get API key
        api_key = get_api_key(task_request.llm_provider, task_request.api_key)
        
        # Create LLM instance
        llm = create_llm(task_request.llm_provider, task_request.llm_model, api_key)
        
        # Clean up old agents periodically
        task_counter += 1
        if task_counter % 5 == 0:  # Every 5 tasks
            logger.info("Performing periodic cleanup...")
            completed_tasks = [tid for tid, result in task_results.items() 
                             if result["status"] in ["completed", "failed"]]
            for tid in completed_tasks[:10]:  # Clean up oldest 10 completed tasks
                cleanup_agent(tid)
                if tid in task_results:
                    del task_results[tid]
        
        # Initialize agent with the existing structure
        agent = Agent(
            task=task_request.task,
            llm=llm,
            max_actions_per_step=task_request.max_actions
        )
        
        # Store active agent
        active_agents[task_id] = agent
        
        # Clear any existing cache before running
        if hasattr(agent, 'mac_tree_builder'):
            agent.mac_tree_builder._element_cache.clear()
            agent.mac_tree_builder._processed_elements.clear()
            agent.mac_tree_builder.highlight_index = 0
        
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
        # Always cleanup after task completion
        cleanup_agent(task_id)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "macOS-use API Server Enhanced", "version": "1.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "macOS-use API Enhanced",
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
    cleanup_agent(task_id)
    
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
        cleanup_agent(task_id)
    
    return {"message": f"Task {task_id} stopped"}

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches and cleanup resources"""
    # Cleanup all active agents
    for task_id in list(active_agents.keys()):
        cleanup_agent(task_id)
    
    # Clear completed/failed tasks older than running tasks
    completed_tasks = [tid for tid, result in task_results.items() 
                     if result["status"] in ["completed", "failed"]]
    
    for tid in completed_tasks:
        if tid in task_results:
            del task_results[tid]
    
    # Force garbage collection
    gc.collect()
    
    return {
        "message": "Cache cleared successfully", 
        "remaining_tasks": len(task_results)
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "available_providers": ["openai", "anthropic", "gemini", "deepseek"],
        "models": {
            "openai": ["gpt-4", "gpt-4-turbo", "o4-mini", "GPT-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-sonnet", "claude-3-haiku"],
            "gemini": ["gemini-pro", "gemini-pro-vision"],
            "deepseek": ["deepseek-chat"]
        },
        "default_settings": {
            "max_steps": 50,
            "max_actions": 50,
            "share_prompt": False,
            "share_terminal": True
        }
    }

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
        "api_server_enhanced:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
