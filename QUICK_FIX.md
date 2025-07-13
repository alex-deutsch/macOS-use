# Quick Fix for "Can't Find Elements" Issue

## When you see "Invalid index" or elements not found:

### ⚡ **Immediate Fixes** (in order of speed):

```bash
# 1. Reset accessibility state (fastest - 2 seconds)
./manage_services.sh reset-accessibility

# 2. Clear all caches (medium - 5 seconds) 
./manage_services.sh clear-cache

# 3. Restart API service (slower - 10 seconds)
./manage_services.sh restart api
```

### 🔍 **Check if it worked:**
```bash
./manage_services.sh test
```

### 📊 **Monitor system health:**
```bash
./manage_services.sh health
```

## 🚀 **API v2.0 Features:**

✅ **Aggressive State Reset** - Completely resets accessibility state between tasks  
✅ **Fresh Components** - New Controller and TreeBuilder for each task  
✅ **Memory Management** - Forced garbage collection and cleanup  
✅ **Quick Recovery** - `/accessibility/reset` endpoint for fast fixes  

## 💡 **Usage Pattern:**

After every **2-3 tasks**, run:
```bash
./manage_services.sh reset-accessibility
```

This prevents the stale element issue from building up!

## 🔗 **API Endpoints:**

| Fix Level | Endpoint | Command |
|-----------|----------|---------|
| **Light** | `/accessibility/reset` | `reset-accessibility` |
| **Medium** | `/cache/clear` | `clear-cache` |
| **Heavy** | Restart service | `restart api` |

The v2 API should significantly reduce this issue, but these commands are your safety net! 🛡️
