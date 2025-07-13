#!/bin/bash

# macOS-use Services Management Script

GRADIO_SERVICE="com.macos-use.gradio"
API_SERVICE="com.macos-use.api"
GRADIO_PLIST="$HOME/Library/LaunchAgents/${GRADIO_SERVICE}.plist"
API_PLIST="$HOME/Library/LaunchAgents/${API_SERVICE}.plist"

case "$1" in
    start)
        echo "Starting macOS-use Gradio service..."
        launchctl load "$PLIST_PATH"
        echo "Service started. Web UI available at: http://localhost:7860"
        ;;
    stop)
        echo "Stopping macOS-use Gradio service..."
        launchctl unload "$PLIST_PATH"
        echo "Service stopped."
        ;;
    restart)
        echo "Restarting macOS-use Gradio service..."
        launchctl unload "$PLIST_PATH" 2>/dev/null
        launchctl load "$PLIST_PATH"
        echo "Service restarted. Web UI available at: http://localhost:7860"
        ;;
    status)
        if launchctl list | grep -q "$SERVICE_NAME"; then
            PID=$(launchctl list | grep "$SERVICE_NAME" | awk '{print $1}')
            echo "✅ Service is running (PID: $PID)"
            echo "🌐 Web UI: http://localhost:7860"
            echo "📄 Logs: /tmp/macos-use-gradio.log"
            echo "❌ Errors: /tmp/macos-use-gradio-error.log"
        else
            echo "❌ Service is not running"
        fi
        ;;
    logs)
        echo "=== Recent logs ==="
        tail -20 /tmp/macos-use-gradio.log
        ;;
    errors)
        echo "=== Recent errors ==="
        if [ -f /tmp/macos-use-gradio-error.log ]; then
            tail -20 /tmp/macos-use-gradio-error.log
        else
            echo "No error log found."
        fi
        ;;
    open)
        echo "Opening web UI in browser..."
        open http://localhost:7860
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|errors|open}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the service"
        echo "  stop     - Stop the service"
        echo "  restart  - Restart the service"
        echo "  status   - Check service status"
        echo "  logs     - Show recent logs"
        echo "  errors   - Show recent errors"
        echo "  open     - Open web UI in browser"
        exit 1
        ;;
esac
