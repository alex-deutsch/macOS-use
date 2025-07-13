from typing import Optional
from pydantic import SecretStr
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import MLX-Swift support
try:
    from mlx_use.llm import create_mlx_chat_model, MLX_SWIFT_MODELS
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# LLM model mappings
LLM_MODELS = {
    "OpenAI": ["gpt-4o", "o3-mini"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"],
    "Google": ["gemini-1.5-flash-002", "gemini-2.0-flash-exp"],
    "alibaba": ["qwen-2.5-72b-instruct"]
}

# Add MLX-Swift models if available
if MLX_AVAILABLE:
    LLM_MODELS["MLX-Swift (Local)"] = list(MLX_SWIFT_MODELS.keys())

def get_llm(provider: str, model: str, api_key: str = None) -> Optional[object]:
    """Initialize LLM based on provider"""
    try:
        if provider == "OpenAI":
            return ChatOpenAI(model=model, api_key=SecretStr(api_key))
        elif provider == "Anthropic":
            return ChatAnthropic(model=model, api_key=SecretStr(api_key))
        elif provider == "Google":
            return ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
        elif provider == "MLX-Swift (Local)":
            if not MLX_AVAILABLE:
                raise ValueError("MLX-Swift is not available. Please install mlx-swift.")
            # For local MLX models, no API key is required
            return create_mlx_chat_model(
                model_name=model,
                temperature=0.7,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        raise ValueError(f"Failed to initialize {provider} LLM: {str(e)}")
