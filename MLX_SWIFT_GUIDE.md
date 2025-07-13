# MLX-Swift Integration Guide

This guide explains how to set up and use MLX-Swift for local inference with macOS-use, eliminating the need for external API keys and providing completely private AI inference on your Mac.

## What is MLX-Swift?

MLX-Swift is Apple's machine learning framework optimized for Apple Silicon (M1/M2/M3/M4 chips). It allows you to run large language models locally on your Mac with excellent performance and efficiency.

## Benefits of MLX-Swift Integration

- **🔒 Complete Privacy**: No data leaves your machine
- **💰 Zero Cost**: No API fees or usage limits
- **⚡ Fast Performance**: Optimized for Apple Silicon
- **🔌 Offline Capability**: Works without internet connection
- **🛡️ No Rate Limits**: Run as many tasks as you want

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or later
- At least 8GB of RAM (16GB recommended for larger models)

### Step 1: Install MLX-Swift

```bash
# Install MLX-Swift via pip
pip install mlx-swift

# Or using conda
conda install -c conda-forge mlx-swift
```

### Step 2: Install/Update macOS-use

```bash
# If you already have macOS-use installed
pip install --upgrade mlx-use

# Or install from source with MLX support
git clone https://github.com/browser-use/macOS-use.git
cd macOS-use
pip install -e .
```

### Step 3: Download Models

MLX-Swift will automatically download models on first use, but you can pre-download them:

```bash
# Download a recommended model
mlx-swift download qwen2.5-7b-instruct

# Or download multiple models
mlx-swift download llama-3.1-8b-instruct
mlx-swift download mistral-7b-instruct
```

## Available Models

| Model | Size | Memory Required | Description |
|-------|------|----------------|-------------|
| `qwen2.5-7b-instruct` | 7B | 8GB | Recommended for most users |
| `qwen2.5-14b-instruct` | 14B | 16GB | Higher quality, needs more memory |
| `llama-3.1-8b-instruct` | 8B | 8GB | Good alternative to Qwen |
| `mistral-7b-instruct` | 7B | 8GB | Fast and efficient |

## Usage Examples

### Basic Usage

```python
from mlx_use import Agent
from mlx_use.llm import create_mlx_chat_model
from mlx_use.controller.service import Controller

# Create MLX model
llm = create_mlx_chat_model(
    model_name="qwen2.5-7b-instruct",
    temperature=0.7,
    max_tokens=2048
)

# Create agent
agent = Agent(
    task="Open calculator and compute 15 + 27",
    llm=llm,
    controller=Controller(),
    use_vision=False
)

# Run the agent
import asyncio
result = asyncio.run(agent.run())
```

### Using Different Models

```python
# Use a larger model for better performance
llm = create_mlx_chat_model(
    model_name="qwen2.5-14b-instruct",
    temperature=0.7,
    max_tokens=2048
)

# Use a custom model path
llm = create_mlx_chat_model(
    model_name="custom-model",
    model_path="/path/to/your/model",
    temperature=0.7,
    max_tokens=2048
)
```

### Running the Example

```bash
# Run the MLX-Swift example
python examples/mlx_local_inference.py

# Run the standard example (will use MLX if no API keys are set)
python examples/try.py
```

## Configuration

### Model Parameters

You can customize the model behavior:

```python
llm = create_mlx_chat_model(
    model_name="qwen2.5-7b-instruct",
    temperature=0.7,        # Creativity (0.0-2.0)
    max_tokens=2048,        # Maximum response length
    top_p=0.9,             # Nucleus sampling
    repetition_penalty=1.1  # Avoid repetition
)
```

### Environment Variables

You can set default MLX-Swift options:

```bash
# Set in your .env file
MLX_SWIFT_MODEL=qwen2.5-7b-instruct
MLX_SWIFT_TEMPERATURE=0.7
MLX_SWIFT_MAX_TOKENS=2048
```

## Integration with Gradio App

The MLX-Swift integration works seamlessly with the Gradio web interface:

1. Start the Gradio app:
```bash
python gradio_app/app.py
```

2. Select "MLX-Swift (Local)" as your provider
3. Choose your preferred model
4. No API key required!

## Troubleshooting

### Common Issues

**Issue**: `MLX Swift executable not found`
**Solution**: Make sure mlx-swift is installed and in your PATH:
```bash
pip install mlx-swift
which mlx-swift  # Should show path to executable
```

**Issue**: `Model not found`
**Solution**: Download the model first:
```bash
mlx-swift download qwen2.5-7b-instruct
```

**Issue**: `Out of memory`
**Solution**: Use a smaller model or close other applications:
```python
# Use smaller model
llm = create_mlx_chat_model(model_name="mistral-7b-instruct")
```

**Issue**: `Slow inference`
**Solution**: 
- Ensure you're on Apple Silicon
- Close other memory-intensive applications
- Try a smaller model
- Reduce max_tokens

### Performance Tips

1. **Use Apple Silicon**: MLX-Swift is optimized for M1/M2/M3/M4 chips
2. **Sufficient RAM**: 8GB minimum, 16GB recommended
3. **Close Other Apps**: Free up memory for better performance
4. **Choose Right Model**: Balance between quality and speed
5. **Adjust Parameters**: Lower temperature and max_tokens for speed

## Comparison with API-based Models

| Aspect | MLX-Swift | API-based |
|--------|-----------|-----------|
| **Privacy** | 🟢 Complete | 🔴 Limited |
| **Cost** | 🟢 Free | 🔴 Per-use |
| **Speed** | 🟡 Good | 🟢 Excellent |
| **Quality** | 🟡 Good | 🟢 Excellent |
| **Reliability** | 🟢 Offline | 🔴 Internet required |
| **Setup** | 🔴 Complex | 🟢 Simple |

## Contributing

To contribute to the MLX-Swift integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with local MLX models
5. Submit a pull request

## Support

For issues related to:
- **MLX-Swift integration**: Open an issue in the macOS-use repository
- **MLX-Swift itself**: Visit the [MLX-Swift GitHub page](https://github.com/ml-explore/mlx-swift)
- **Model downloads**: Check the [MLX model repository](https://huggingface.co/mlx-community)

## Future Enhancements

- [ ] Vision model support with MLX-VLM
- [ ] Model quantization options
- [ ] Batch processing support
- [ ] Custom model fine-tuning
- [ ] Performance monitoring
- [ ] Auto model selection based on task complexity

---

With MLX-Swift integration, you can now enjoy the full power of macOS-use with complete privacy and no ongoing costs. The local inference runs entirely on your Mac, making it perfect for sensitive tasks or when you want to avoid API limits.
