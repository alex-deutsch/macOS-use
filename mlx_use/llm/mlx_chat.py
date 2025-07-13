"""
MLX-Swift integration for macOS-use
Provides local inference capabilities using Apple's MLX framework
"""

import asyncio
import json
import subprocess
import tempfile
import os
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from pydantic import Field, SecretStr
from langchain_core.utils import get_from_dict_or_env


class MLXChatModel(BaseChatModel):
    """
    MLX-Swift chat model that runs local inference on macOS using Apple's MLX framework.
    
    This model integrates with the existing langchain framework used in macOS-use
    and provides local inference capabilities without requiring external API calls.
    """
    
    model_name: str = Field(default="mlx-swift-qwen2.5-7b-instruct")
    model_path: Optional[str] = Field(default=None, description="Path to the MLX model directory")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    
    # MLX Swift executable path
    mlx_swift_path: str = Field(default="mlx-swift-chat", description="Path to mlx-swift-chat executable")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_mlx_setup()
    
    def _validate_mlx_setup(self):
        """Validate that MLX Swift is properly installed and configured."""
        try:
            result = subprocess.run(
                [self.mlx_swift_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"MLX Swift not found or not working: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("MLX Swift command timed out")
        except FileNotFoundError:
            raise RuntimeError(
                f"MLX Swift executable not found at {self.mlx_swift_path}. "
                "Please install mlx-swift and ensure it's in your PATH or provide the correct path."
            )
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages for MLX Swift input."""
        formatted_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                formatted_messages.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"Assistant: {message.content}")
            else:
                # Fallback for other message types
                formatted_messages.append(f"User: {message.content}")
        
        return "\n".join(formatted_messages)
    
    def _run_mlx_inference(self, prompt: str) -> str:
        """Run MLX Swift inference with the given prompt."""
        cmd = [
            self.mlx_swift_path,
            "--model", self.model_path or self.model_name,
            "--temperature", str(self.temperature),
            "--max-tokens", str(self.max_tokens),
            "--top-p", str(self.top_p),
            "--repetition-penalty", str(self.repetition_penalty),
            "--prompt", prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for inference
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"MLX Swift inference failed: {result.stderr}")
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("MLX Swift inference timed out")
        except Exception as e:
            raise RuntimeError(f"MLX Swift inference error: {str(e)}")
    
    async def _arun_mlx_inference(self, prompt: str) -> str:
        """Async version of MLX Swift inference."""
        cmd = [
            self.mlx_swift_path,
            "--model", self.model_path or self.model_name,
            "--temperature", str(self.temperature),
            "--max-tokens", str(self.max_tokens),
            "--top-p", str(self.top_p),
            "--repetition-penalty", str(self.repetition_penalty),
            "--prompt", prompt
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # 2 minute timeout
            )
            
            if process.returncode != 0:
                raise RuntimeError(f"MLX Swift inference failed: {stderr.decode()}")
            
            return stdout.decode().strip()
            
        except asyncio.TimeoutError:
            raise RuntimeError("MLX Swift inference timed out")
        except Exception as e:
            raise RuntimeError(f"MLX Swift inference error: {str(e)}")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using MLX Swift."""
        prompt = self._format_messages(messages)
        
        if run_manager:
            run_manager.on_llm_start({"messages": [m.content for m in messages]})
        
        try:
            response = self._run_mlx_inference(prompt)
            
            if run_manager:
                run_manager.on_llm_end(response)
            
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion using MLX Swift."""
        prompt = self._format_messages(messages)
        
        if run_manager:
            await run_manager.on_llm_start({"messages": [m.content for m in messages]})
        
        try:
            response = await self._arun_mlx_inference(prompt)
            
            if run_manager:
                await run_manager.on_llm_end(response)
            
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e)
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mlx-swift"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }


class MLXSwiftChatModel(MLXChatModel):
    """
    Alias for MLXChatModel to maintain consistency with naming conventions.
    """
    pass


# Available MLX Swift models
MLX_SWIFT_MODELS = {
    "qwen2.5-7b-instruct": {
        "name": "qwen2.5-7b-instruct",
        "description": "Qwen2.5 7B Instruct model optimized for MLX",
        "size": "7B",
        "recommended_memory": "8GB"
    },
    "qwen2.5-14b-instruct": {
        "name": "qwen2.5-14b-instruct", 
        "description": "Qwen2.5 14B Instruct model optimized for MLX",
        "size": "14B",
        "recommended_memory": "16GB"
    },
    "llama-3.1-8b-instruct": {
        "name": "llama-3.1-8b-instruct",
        "description": "Llama 3.1 8B Instruct model optimized for MLX",
        "size": "8B", 
        "recommended_memory": "8GB"
    },
    "mistral-7b-instruct": {
        "name": "mistral-7b-instruct",
        "description": "Mistral 7B Instruct model optimized for MLX",
        "size": "7B",
        "recommended_memory": "8GB"
    }
}


def get_available_mlx_models() -> Dict[str, Dict[str, str]]:
    """Get list of available MLX Swift models."""
    return MLX_SWIFT_MODELS


def create_mlx_chat_model(
    model_name: str = "qwen2.5-7b-instruct",
    model_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs
) -> MLXChatModel:
    """
    Create an MLX Swift chat model with the specified parameters.
    
    Args:
        model_name: Name of the MLX model to use
        model_path: Optional path to the model directory
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for the model
    
    Returns:
        MLXChatModel instance
    """
    return MLXChatModel(
        model_name=model_name,
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
