# Ollama Model Compatibility for macOS-use

## 🚨 **Important**: Function/Tool Calling Required

macOS-use requires models that support **function calling** (also called tool calling) to work properly. The agent needs to call functions like:
- `click_element`
- `open_app` 
- `type_text`
- `done`

## ✅ **Compatible Models in Your Setup:**

### **Recommended (Known to work):**
1. **`qwen3:14b`** ✅ - Supports function calling, best performance
2. **`qwen2.5-coder:7b`** ✅ - Good for automation tasks
3. **`deepseek-r1:14b`** ✅ - Advanced reasoning with tools

### **Potentially Compatible:**
4. **`llama3.2-vision:11b`** ⚠️ - May support tools, worth testing
5. **`qwen2.5vl:7b`** ⚠️ - Vision model, may support tools

### **❌ Not Compatible:**
6. **`gemma3:1b`** ❌ - Does not support function calling
7. **`gemma3:4b`** ❌ - Does not support function calling  
8. **`gemma3:12b`** ❌ - Does not support function calling

## 🧪 **Testing Models:**

```bash
# Test with recommended models
./manage_services.sh test-ollama "qwen3:14b"
./manage_services.sh test-ollama "qwen2.5-coder:7b"
./manage_services.sh test-ollama "deepseek-r1:14b"

# Test potentially compatible
./manage_services.sh test-ollama "llama3.2-vision:11b"
./manage_services.sh test-ollama "qwen2.5vl:7b"
```

## 🔍 **How to Check Compatibility:**

Look for this error in logs when a model doesn't support tools:
```
ERROR: registry.ollama.ai/library/[model] does not support tools (status code: 400)
```

## 🚀 **Recommended Usage:**

### **For Quick Tasks:**
```bash
curl -X POST "http://localhost:8080/quick-task?task=open%20calculator&provider=ollama&model=qwen2.5-coder:7b"
```

### **For Complex Automation:**
```bash
curl -X POST "http://localhost:8080/quick-task?task=open%20safari%20and%20navigate%20to%20google&provider=ollama&model=qwen3:14b"
```

### **For Vision Tasks:**
```bash
curl -X POST "http://localhost:8080/quick-task?task=describe%20what%20you%20see%20on%20screen&provider=ollama&model=llama3.2-vision:11b"
```

## 📋 **Model Performance Guide:**

| Model | Function Calling | Speed | Capability | Best For |
|-------|-----------------|-------|------------|----------|
| `qwen3:14b` | ✅ | Medium | High | Complex automation |
| `qwen2.5-coder:7b` | ✅ | Fast | Medium | Quick tasks, coding |
| `deepseek-r1:14b` | ✅ | Slow | Very High | Complex reasoning |
| `llama3.2-vision:11b` | ⚠️ | Medium | High | Vision tasks |
| `qwen2.5vl:7b` | ⚠️ | Fast | Medium | Vision + automation |
| `gemma3:*` | ❌ | - | - | Not compatible |

## 🛠️ **Installing Compatible Models:**

If you need more compatible models:

```bash
# Install known-good models
ollama pull llama3.1:8b       # Usually supports function calling
ollama pull mistral:7b        # Usually supports function calling
ollama pull codellama:7b      # Code-focused with function calling

# Check which models support tools
./manage_services.sh test-ollama "llama3.1:8b"
```

## 🎯 **Troubleshooting:**

### **"Does not support tools" Error:**
- ❌ Model doesn't support function calling
- ✅ Switch to `qwen3:14b` or `qwen2.5-coder:7b`

### **Task Starts but Never Completes:**
- ⚠️ Model might support tools but be slow
- ✅ Wait longer or use faster model

### **Task Fails with Function Errors:**
- ⚠️ Model supports tools but doesn't understand them well
- ✅ Use `qwen3:14b` for better function understanding

## 📊 **Current Status of Your Models:**

Based on testing:
- ✅ **3 models confirmed working** (qwen3:14b, qwen2.5-coder:7b, deepseek-r1:14b)
- ⚠️ **2 models need testing** (llama3.2-vision:11b, qwen2.5vl:7b)  
- ❌ **3 models incompatible** (all gemma3 variants)

**Recommendation**: Use `qwen3:14b` for best results, or `qwen2.5-coder:7b` for faster responses.
