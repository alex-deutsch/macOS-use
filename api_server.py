#!/usr/bin/env python3
"""
REST API Server for macOS-use
Provides HTTP endpoints to control macOS automation
"""

import os
import asyncio
import uvicorn
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
    title="macOS-use API",
    description="REST API for macOS automation using AI agents",
    version="1.0.0",
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
current_agent: Optional[Agent] = None
current_task: Optional[asyncio.Task] = None
task_results: Dict[str, Any] = {}

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

class ConfigUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None

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

async def run_agent_task(task_request: TaskRequest, task_id: str):
    """Run agent task in background"""
    global task_results
    
    try:
        # Get API key
        api_key = get_api_key(task_request.llm_provider, task_request.api_key)
        
        # Create LLM instance
        llm = create_llm(task_request.llm_provider, task_request.llm_model, api_key)
        
        # Initialize agent with the existing structure
        agent = Agent(
            task=task_request.task,
            llm=llm,
            max_actions_per_step=task_request.max_actions
        )
        
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

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "macOS-use API Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "macOS-use API"}

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new automation task"""
    global current_task
    
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
    
    # Start background task using asyncio instead of FastAPI background tasks
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
    
    return {"message": f"Task {task_id} stopped"}

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
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
