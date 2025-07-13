"""
Example demonstrating MLX-Swift local inference integration
This example shows how to use MLX-Swift for local inference without external API calls
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path to import mlx_use
sys.path.append(str(Path(__file__).parent.parent))

from mlx_use import Agent
from mlx_use.llm import create_mlx_chat_model, get_available_mlx_models, MLX_SWIFT_MODELS
from mlx_use.controller.service import Controller


def print_available_models():
    """Print available MLX models and their specifications."""
    print("🔧 Available MLX-Swift Models:")
    print("=" * 50)
    
    for model_id, model_info in MLX_SWIFT_MODELS.items():
        print(f"Model: {model_info['name']}")
        print(f"  Description: {model_info['description']}")
        print(f"  Size: {model_info['size']}")
        print(f"  Recommended Memory: {model_info['recommended_memory']}")
        print()


def create_mlx_model(model_name: str = "qwen2.5-7b-instruct", model_path: str = None):
    """Create an MLX model with the specified parameters."""
    try:
        print(f"🚀 Initializing MLX-Swift model: {model_name}")
        
        # Create the MLX chat model
        llm = create_mlx_chat_model(
            model_name=model_name,
            model_path=model_path,
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        print(f"✅ MLX-Swift model '{model_name}' initialized successfully")
        return llm
        
    except Exception as e:
        print(f"❌ Error initializing MLX-Swift model: {e}")
        print("   Please ensure mlx-swift is installed and the model is available")
        return None


async def run_mlx_agent_example():
    """Run an example agent using MLX-Swift for local inference."""
    print("🤖 MLX-Swift Local Inference Example")
    print("=" * 50)
    
    # Print available models
    print_available_models()
    
    # Create the MLX model
    model_name = "qwen2.5-7b-instruct"  # You can change this to any supported model
    model_path = None  # Set to a specific path if you have custom models
    
    llm = create_mlx_model(model_name, model_path)
    if not llm:
        return
    
    # Create controller
    controller = Controller()
    
    # Create agent with MLX model
    print(f"🎯 Creating agent with MLX-Swift model...")
    agent = Agent(
        task='Open the calculator app and calculate 15 + 27, then tell me the result.',
        llm=llm,
        controller=controller,
        use_vision=False,  # Set to True if you want vision capabilities
        max_actions_per_step=3,
        max_failures=5,
        max_input_tokens=4096  # Adjust based on your model's context length
    )
    
    print("🏃 Running agent...")
    try:
        # Run the agent
        result = await agent.run(max_steps=10)
        
        print("\n📊 Agent Results:")
        print("=" * 30)
        print(f"Task completed: {result.is_done()}")
        print(f"Steps taken: {len(result.history)}")
        
        # Print task results
        if result.is_done():
            print("✅ Task completed successfully!")
            if result.history:
                last_result = result.history[-1].result
                if last_result and last_result[-1].extracted_content:
                    print(f"Result: {last_result[-1].extracted_content}")
        else:
            print("❌ Task was not completed")
            
    except Exception as e:
        print(f"❌ Error running agent: {e}")


async def interactive_mlx_example():
    """Interactive example allowing users to input custom tasks."""
    print("🎮 Interactive MLX-Swift Example")
    print("=" * 40)
    
    # Create the MLX model
    model_name = input("Enter model name (default: qwen2.5-7b-instruct): ").strip()
    if not model_name:
        model_name = "qwen2.5-7b-instruct"
    
    llm = create_mlx_model(model_name)
    if not llm:
        return
    
    controller = Controller()
    
    while True:
        print("\n" + "=" * 50)
        task = input("Enter a task for the agent (or 'quit' to exit): ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not task:
            print("⚠️  Please enter a valid task")
            continue
        
        # Create agent for this task
        agent = Agent(
            task=task,
            llm=llm,
            controller=controller,
            use_vision=False,
            max_actions_per_step=3,
            max_failures=5,
            max_input_tokens=4096
        )
        
        print(f"🚀 Executing task: {task}")
        
        try:
            result = await agent.run(max_steps=15)
            
            print(f"\n📊 Task Result:")
            print(f"Completed: {result.is_done()}")
            print(f"Steps: {len(result.history)}")
            
            if result.is_done() and result.history:
                last_result = result.history[-1].result
                if last_result and last_result[-1].extracted_content:
                    print(f"Output: {last_result[-1].extracted_content}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to run the MLX-Swift examples."""
    print("🍎 macOS-use MLX-Swift Integration Examples")
    print("=" * 50)
    
    # Check if MLX-Swift is available
    print("🔍 Checking MLX-Swift availability...")
    
    try:
        # Try to create a model to check if MLX-Swift is available
        test_model = create_mlx_chat_model("qwen2.5-7b-instruct")
        print("✅ MLX-Swift is available!")
    except Exception as e:
        print(f"❌ MLX-Swift is not available: {e}")
        print("   Please install mlx-swift to use local inference")
        return
    
    # Show menu
    print("\nChoose an example:")
    print("1. Run predefined calculator example")
    print("2. Interactive mode (enter custom tasks)")
    print("3. Show available models only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_mlx_agent_example())
    elif choice == "2":
        asyncio.run(interactive_mlx_example())
    elif choice == "3":
        print_available_models()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
