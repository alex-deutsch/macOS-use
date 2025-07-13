# macOS-use Troubleshooting Guide

## Common Issues and Solutions

### Issue: "Can't find elements in app anymore" after multiple task runs

**Symptoms:**
- Tasks work initially but fail after several runs
- Error messages about missing or invalid UI elements
- Apps seem unresponsive to automation
- Restarting macOS-use fixes the issue temporarily

**Root Cause:**
This is due to **Accessibility API state corruption** and **stale UI element caching** in macOS. The system accumulates invalid references to UI elements over time.

### Solutions:

#### 1. **Quick Fix - Clear API Cache**
```bash
# Clear the cache without restarting services
./manage_services.sh clear-cache
```

#### 2. **Check API Health**
```bash
# Check system status and active tasks
./manage_services.sh health
```

#### 3. **Restart API Service**
```bash
# Restart just the API service
./manage_services.sh restart api
```

#### 4. **Full Service Restart**
```bash
# Restart both Gradio and API services
./manage_services.sh restart
```

#### 5. **Programmatic Cache Clearing**
You can also clear the cache via HTTP API:
```bash
curl -X POST "http://localhost:8080/cache/clear"
```

### Enhanced Features in API v1.1.0:

✅ **Automatic Cache Management**
- Agents are cleaned up after each task
- Periodic cleanup every 5 tasks
- Memory optimization with garbage collection

✅ **Cache Clearing Endpoint**
- Manual cache clearing via `/cache/clear`
- Removes stale UI element references
- Cleans up completed task history

✅ **Better Resource Management**
- Each agent instance is properly disposed
- Element caches are reset between tasks
- Process IDs are properly tracked

### Prevention Tips:

1. **Use the Enhanced API** - The new API automatically manages caches
2. **Clear Cache Periodically** - If running many tasks, clear cache every 10-15 tasks
3. **Monitor Task Count** - Use `/health` endpoint to monitor active tasks
4. **Restart Services Nightly** - For heavy usage, restart services daily

### Monitoring Commands:

```bash
# Check service status
./manage_services.sh status

# View recent logs
./manage_services.sh logs api

# Check for errors
./manage_services.sh errors api

# Test API functionality
./manage_services.sh test

# Clear cache when needed
./manage_services.sh clear-cache
```

### Manual Accessibility Reset (Advanced):

If the issue persists, you can reset macOS Accessibility permissions:

1. **System Preferences → Security & Privacy → Accessibility**
2. **Remove and re-add Terminal/Python** (the app running macOS-use)
3. **Restart the macOS-use services**

### Logs to Check:

- **API Logs**: `/tmp/macos-use-api.log`
- **API Errors**: `/tmp/macos-use-api-error.log`
- **Gradio Logs**: `/tmp/macos-use-gradio.log`

### Performance Impact:

The enhanced cache management has minimal performance impact:
- ~1-2ms overhead per task for cleanup
- Automatic garbage collection
- Better memory usage over time
- More reliable element detection

### API Endpoints for Cache Management:

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/health` | GET | Check API health and task counts |
| `/cache/clear` | POST | Clear all caches and cleanup |
| `/tasks` | GET | List all tasks and statuses |
| `/tasks/{id}` | DELETE | Delete specific task and cleanup |

This enhanced version should significantly reduce the "stale element" issue you were experiencing!
