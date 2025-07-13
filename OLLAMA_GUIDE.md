# Ollama Integration with macOS-use

## ✅ **Yes! macOS-use works perfectly with Ollama local models!**

You already have Ollama installed and 8 models available. Here's how to use them:

## 🚀 **Quick Start**

### Available Models in Your Setup:
- `qwen3:14b` - High-quality general purpose model
- `qwen2.5vl:7b` - Vision-language model  
- `gemma3:1b` - Fast, lightweight model
- `gemma3:4b` - Balanced performance
- `gemma3:12b` - High capability model
- `deepseek-r1:14b` - Reasoning-focused model
- `qwen2.5-coder:7b` - Code-specialized model
- `llama3.2-vision:11b` - Vision capabilities

### 🔧 **Management Commands**

```bash
# Check Ollama status
./manage_services.sh ollama-status

# List available models
./manage_services.sh ollama-models

# Test with specific model
./manage_services.sh test-ollama "gemma3:1b"
./manage_services.sh test-ollama "qwen3:14b"
```

## 📡 **API Usage**

### REST API Examples:

**Simple task with Ollama:**
```bash
curl -X POST "http://localhost:8080/quick-task?task=open%20calculator&provider=ollama&model=gemma3:1b"
```

**Full task request:**
```bash
curl -X POST "http://localhost:8080/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "open Safari and go to google.com",
    "llm_provider": "ollama",
    "llm_model": "qwen3:14b",
    "max_steps": 30,
    "max_actions": 10
  }'
```

### 🌐 **Gradio Web UI**

In the Gradio interface at http://localhost:7860:
1. **LLM Provider**: Select "ollama"
2. **Model**: Choose from your available models
3. **API Key**: Leave blank (not needed for Ollama)

## 🎯 **Model Recommendations**

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Quick tasks** | `gemma3:1b` | Fastest response |
| **Complex automation** | `qwen3:14b` | Best reasoning |
| **Vision tasks** | `llama3.2-vision:11b` | Sees screenshots |
| **Code tasks** | `qwen2.5-coder:7b` | Code-optimized |
| **Balanced use** | `gemma3:4b` | Good speed/quality |

## ⚡ **Performance Tips**

1. **Start with lightweight models** like `gemma3:1b` for testing
2. **Use vision models** for complex UI interactions
3. **Ollama runs locally** - no API costs!
4. **Models stay loaded** - subsequent requests are faster

## 🔧 **Configuration**

### Environment Setup:
- ✅ Ollama is running on `http://localhost:11434`
- ✅ 8 models are available and ready
- ✅ API v2 has full Ollama integration
- ✅ No API keys needed

### API Endpoints:
- `/ollama/status` - Check Ollama server status
- `/ollama/models` - List available models
- `/config` - Shows Ollama integration status

## 🚨 **Troubleshooting**

### If Ollama isn't working:

```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve

# Pull a model if needed
ollama pull gemma3:1b

# Check API integration
./manage_services.sh ollama-status
```

### Common Issues:

1. **"Ollama server not available"**: Run `ollama serve`
2. **Model not found**: Run `ollama pull [model-name]`
3. **Slow responses**: Use smaller models like `gemma3:1b`

## 💰 **Benefits of Local Ollama**

✅ **No API costs** - Run unlimited tasks locally  
✅ **Privacy** - Data never leaves your machine  
✅ **Speed** - No network latency once loaded  
✅ **Reliability** - Works offline  
✅ **Control** - Choose your preferred models  

## 🎮 **Example Tasks**

Try these with Ollama:

```bash
# Basic automation
./manage_services.sh test-ollama "gemma3:1b"

# More complex task
curl -X POST "http://localhost:8080/quick-task?task=open%20notes%20and%20create%20a%20new%20note&provider=ollama&model=qwen3:14b"

# Vision-enabled task
curl -X POST "http://localhost:8080/quick-task?task=take%20a%20screenshot%20and%20describe%20what%20you%20see&provider=ollama&model=llama3.2-vision:11b"
```

## 🔄 **Model Management**

```bash
# Add more models
ollama pull llama3.1:8b
ollama pull mistral:7b

# Remove unused models
ollama rm [model-name]

# Update models
ollama pull [model-name]
```

Your setup is ready to go! Ollama provides a fantastic local alternative to cloud-based LLMs for macOS automation. 🚀
